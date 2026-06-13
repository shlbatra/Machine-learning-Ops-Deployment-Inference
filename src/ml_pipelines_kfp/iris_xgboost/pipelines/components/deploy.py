from kfp.dsl import Input, Model, component, Artifact, Output
import ml_pipelines_kfp.iris_xgboost.constants as _constants


@component(
    base_image=_constants.IMAGE_NAME,
)
def deploy_blessed_model_to_fastapi(
    project_id: str,
    location: str,
    model_name: str,
    service_name: str,
    fastapi_image_name: str,
    service_endpoint: Output[Artifact],
):
    from google.cloud import aiplatform, aiplatform_v1, run_v2, storage
    from google.auth import default
    from google.cloud.run_v2 import ServicesClient
    from google.iam.v1 import iam_policy_pb2
    from google.iam.v1.iam_policy_pb2 import SetIamPolicyRequest
    from google.iam.v1 import policy_pb2
    import joblib
    import tempfile
    import os
    import requests
    import time
    from ml_pipelines_kfp.log import get_logger

    logger = get_logger(__name__)

    logger.info(f"Starting FastAPI deployment for blessed model: {model_name}")

    aiplatform.init(project=project_id, location=location)

    credentials, _ = default()

    client = aiplatform_v1.ModelServiceClient(
        credentials=credentials,
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"},
    )

    logger.info(f"Searching for blessed model with name: {model_name}")

    request = {
        "parent": f"projects/{project_id}/locations/{location}",
        "filter": f"display_name={model_name}",
    }

    models = list(client.list_models(request=request))
    blessed_model = None

    logger.info(f"Found {len(models)} model versions with name {model_name}")

    for parent_model in models:
        versions_request = {"name": parent_model.name}
        versions = list(client.list_model_versions(request=versions_request))

        for version in versions:
            if "blessed" in version.version_aliases:
                blessed_model = version
                logger.info(f"Found blessed version: {version.version_id}")
                break

        if blessed_model:
            break

    if not blessed_model:
        available_versions = [
            (m.resource_name, m.version_id, list(m.version_aliases)) for m in models
        ]
        raise ValueError(
            f"No blessed version found for model {model_name}. Available versions: {available_versions}"
        )

    logger.info(f"Found blessed model: {blessed_model.name}, URI: {blessed_model.artifact_uri}")

    gcs_uri = blessed_model.artifact_uri
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected GCS URI, got: {gcs_uri}")

    bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
    model_path = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])

    logger.info(f"Downloading model from gs://{bucket_name}/{model_path}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    model_blob_path = f"{model_path}/model.joblib"
    blob = bucket.blob(model_blob_path)

    if not blob.exists():
        raise ValueError(
            f"Model file not found at gs://{bucket_name}/{model_blob_path}"
        )

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        local_model_path = temp_file.name

    deployment_model_path = f"deployed-models/{service_name}/model.joblib"
    deployment_blob = bucket.blob(deployment_model_path)

    logger.info(f"Copying model to deployment location: gs://{bucket_name}/{deployment_model_path}")
    deployment_blob.upload_from_filename(local_model_path)

    model_gcs_path = f"gs://{bucket_name}/{deployment_model_path}"

    logger.info(f"Deploying to Cloud Run service: {service_name}")

    run_client = run_v2.ServicesClient()

    generic_image = fastapi_image_name

    service_config = {
        "parent": f"projects/{project_id}/locations/{location}",
        "service_id": service_name,
        "service": {
            "template": {
                "containers": [
                    {
                        "image": generic_image,
                        "ports": [{"container_port": 8080}],
                        "resources": {"limits": {"memory": "2Gi", "cpu": "2"}},
                        "env": [
                            {"name": "MODEL_GCS_PATH", "value": model_gcs_path},
                            {"name": "MODEL_NAME", "value": model_name},
                            {"name": "GOOGLE_CLOUD_PROJECT", "value": project_id},
                        ],
                    }
                ],
                "scaling": {"min_instance_count": 0, "max_instance_count": 10},
                "service_account": f"kfp-mlops@{project_id}.iam.gserviceaccount.com",
            },
            "traffic": [
                {"percent": 100, "type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"}
            ],
        },
    }

    try:
        try:
            existing_service = run_client.get_service(
                name=f"projects/{project_id}/locations/{location}/services/{service_name}"
            )
            logger.info(f"Service {service_name} already exists, updating...")

            update_service = service_config["service"]
            update_service["name"] = existing_service.name

            operation = run_client.update_service(service=update_service)
            result = operation.result(timeout=600)

        except Exception as get_error:
            logger.info(f"Service doesn't exist, creating new one: {get_error}")
            operation = run_client.create_service(request=service_config)
            result = operation.result(timeout=600)

        run_client = ServicesClient()

        policy = policy_pb2.Policy()
        binding = policy_pb2.Binding()
        binding.role = "roles/run.invoker"
        binding.members.append("allUsers")
        policy.bindings.append(binding)

        iam_request = SetIamPolicyRequest(
            resource=result.name, policy=policy
        )
        run_client.set_iam_policy(request=iam_request)

        service_url = result.uri
        logger.info(f"Service deployed successfully to: {service_url}")

        logger.info("Testing deployment...")
        time.sleep(30)

        test_payload = {
            "instances": [
                {
                    "SepalLengthCm": 5.1,
                    "SepalWidthCm": 3.5,
                    "PetalLengthCm": 1.4,
                    "PetalWidthCm": 0.2,
                }
            ]
        }

        try:
            health_response = requests.get(f"{service_url}/health/live", timeout=30)
            logger.info(f"Health check status: {health_response.status_code}")

            response = requests.post(
                f"{service_url}/predict", json=test_payload, timeout=30
            )
            if response.status_code == 200:
                logger.info("Deployment test successful")
            else:
                logger.error(
                    f"Prediction test failed: {response.status_code} - {response.text}"
                )

        except Exception as test_e:
            logger.error(f"Test request failed: {test_e}")

        service_endpoint.uri = service_url
        service_endpoint.metadata = {
            "service_name": service_name,
            "model_version": blessed_model.version_id,
            "model_name": model_name,
            "deployment_type": "cloud_run_fastapi",
            "model_gcs_path": model_gcs_path,
            "image": generic_image,
        }

        logger.info(f"Deployment completed. Service URL: {service_url}")

    except Exception as deploy_e:
        logger.error(f"Cloud Run deployment failed: {deploy_e}")
        raise
    finally:
        try:
            os.unlink(local_model_path)
        except Exception as cleanup_e:
            logger.error(f"Cleanup warning: {cleanup_e}")
