from kfp.dsl import Input, Model, component, Artifact, Output
from ml_pipelines_kfp.iris_xgboost.constants import IMAGE_NAME

@component(
    base_image=IMAGE_NAME, 
    packages_to_install=[
        "google-cloud-aiplatform>=1.59.0",
        "google-cloud-run>=0.10.0",
        "google-cloud-storage>=2.10.0",
        "requests>=2.31.0",
        "joblib>=1.4.2",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "grpcio-status>=1.62.3"
    ]
)
def deploy_blessed_model_to_fastapi(
    project_id: str,
    location: str,
    model_name: str,
    service_name: str,
    fastapi_image_name: str,
    service_endpoint: Output[Artifact]
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

    print(f"Starting FastAPI deployment for blessed model: {model_name}")
    print(f"Service name: {service_name}")

    # 1. Initialize Vertex AI and get credentials
    aiplatform.init(project=project_id, location=location)
    
    # Get default credentials
    credentials, _ = default()
    print(credentials)
    
    # Create client with explicit credentials
    client = aiplatform_v1.ModelServiceClient(
        credentials=credentials,
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )

    print(f"Searching for blessed model with name: {model_name}")
    
    # Use the high-level aiplatform library to list all model versions
    # models = aiplatform.Model.list(filter=f"display_name={model_name}")
    # blessed_model = None

    request = {
            "parent": f"projects/{project_id}/locations/{location}",
            "filter": f"display_name={model_name}"
        }

    models = list(client.list_models(request=request))
    blessed_model = None
    
    print(f"Found {len(models)} model versions with name {model_name}")
    
    # Search through all model versions (each item in models is already a version)
    for parent_model in models:
        print(f"Checking parent model: {parent_model.name}")
        
        # List all versions of this model
        versions_request = {"name": parent_model.name}
        versions = list(client.list_model_versions(request=versions_request))
        
        print(f"Found {len(versions)} versions for this model")
        
        for version in versions:
            print(f"Version {version.version_id}: Aliases = {list(version.version_aliases)}")
            if "blessed" in version.version_aliases:
                blessed_model = version
                print(f"Found blessed version: {version.version_id}")
                break
        
        if blessed_model:
            break
    
    if not blessed_model:
        available_versions = [(m.resource_name, m.version_id, list(m.version_aliases)) for m in models]
        raise ValueError(f"No blessed version found for model {model_name}. Available versions: {available_versions}")
        
    print(f"Found blessed model: {blessed_model.name}")
    print(f"Model URI: {blessed_model.artifact_uri}")
    
    # 2. Download joblib model from blessed version
    gcs_uri = blessed_model.artifact_uri
    if not gcs_uri.startswith('gs://'):
        raise ValueError(f"Expected GCS URI, got: {gcs_uri}")
    
    bucket_name = gcs_uri.replace('gs://', '').split('/')[0]
    model_path = '/'.join(gcs_uri.replace('gs://', '').split('/')[1:])
    
    print(f"Downloading model from gs://{bucket_name}/{model_path}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Download and validate the model
    model_blob_path = f"{model_path}/model.joblib"
    blob = bucket.blob(model_blob_path)
    
    if not blob.exists():
        raise ValueError(f"Model file not found at gs://{bucket_name}/{model_blob_path}")
    
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        local_model_path = temp_file.name
    
    print(f"Downloaded model to: {local_model_path}")
    
    # 3. Validate model can be loaded
    try:
        model_obj = joblib.load(local_model_path)
        print(f"Model type: {type(model_obj)}")
        print(f"Model validation successful")
    except Exception as e:
        os.unlink(local_model_path)
        raise ValueError(f"Model validation failed: {e}")
    
    # 4. Copy model to standard deployment location
    deployment_model_path = f"deployed-models/{service_name}/model.joblib"
    deployment_blob = bucket.blob(deployment_model_path)
    
    print(f"Copying model to deployment location: gs://{bucket_name}/{deployment_model_path}")
    deployment_blob.upload_from_filename(local_model_path)
    
    model_gcs_path = f"gs://{bucket_name}/{deployment_model_path}"
    print(f"Model available at: {model_gcs_path}")
    
    # 5. Deploy to Cloud Run using pre-built generic image
    print(f"Deploying to Cloud Run service: {service_name}")
    
    run_client = run_v2.ServicesClient()
    
    # Use pre-built generic FastAPI image from CI/CD
    generic_image = fastapi_image_name
    
    service_config = {
        "parent": f"projects/{project_id}/locations/{location}",
        "service_id": service_name,
        "service": {
            "template": {
                "containers": [{
                    "image": generic_image,
                    "ports": [{"container_port": 8080}],
                    "resources": {
                        "limits": {
                            "memory": "2Gi",
                            "cpu": "2"
                        }
                    },
                    "env": [
                        {"name": "MODEL_GCS_PATH", "value": model_gcs_path},
                        {"name": "MODEL_NAME", "value": model_name},
                        {"name": "GOOGLE_CLOUD_PROJECT", "value": project_id}
                    ]
                }],
                "scaling": {
                    "min_instance_count": 0,
                    "max_instance_count": 10
                },
                "service_account": f"kfp-mlops@{project_id}.iam.gserviceaccount.com"
            },
            "traffic": [{"percent": 100, "type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"}]
        }
    }
    
    try:
        # Check if service already exists
        try:
            existing_service = run_client.get_service(
                name=f"projects/{project_id}/locations/{location}/services/{service_name}"
            )
            print(f"Service {service_name} already exists, updating...")
            
            # Update existing service
            update_service = service_config["service"]
            update_service["name"] = existing_service.name
            
            operation = run_client.update_service(service=update_service)
            result = operation.result(timeout=600)
            
        except Exception as get_error:
            print(f"Service doesn't exist, creating new one: {get_error}")
            # Create new service
            operation = run_client.create_service(request=service_config)
            result = operation.result(timeout=600)
            
        
        run_client = ServicesClient()
            
        # Create policy to allow public access
        policy = policy_pb2.Policy()
        binding = policy_pb2.Binding()
        binding.role = "roles/run.invoker"
        binding.members.append("allUsers")
        policy.bindings.append(binding)

        # Apply the policy
        iam_request = SetIamPolicyRequest(
            resource=result.name,  # This should be the full resource name
            policy=policy
        )
        run_client.set_iam_policy(request=iam_request)

        service_url = result.uri
        print(f"Service deployed successfully to: {service_url}")
        
        # 6. Test deployment
        print("Testing deployment...")
        time.sleep(30)  # Wait for service to be ready
        
        test_payload = {
            "instances": [
               {"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2}
            ]
        }
        
        try:
            # Test health endpoint first
            health_response = requests.get(f"{service_url}/health", timeout=30)
            print(f"Health check status: {health_response.status_code}")
            if health_response.status_code == 200:
                print(f"Health check response: {health_response.json()}")
            
            # Test prediction endpoint
            response = requests.post(
                f"{service_url}/predict", 
                json=test_payload,
                timeout=30
            )
            if response.status_code == 200:
                print("Deployment test successful!")
                print(f"Prediction: {response.json()}")
            else:
                print(f"Prediction test failed: {response.status_code} - {response.text}")
                
        except Exception as test_e:
            print(f"Test request failed: {test_e}")
        
        # 7. Set output artifact
        service_endpoint.uri = service_url
        service_endpoint.metadata = {
            "service_name": service_name,
            "model_version": blessed_model.version_id,
            "model_name": model_name,
            "deployment_type": "cloud_run_fastapi",
            "model_gcs_path": model_gcs_path,
            "image": generic_image
        }
        
        print(f"Deployment completed successfully!")
        print(f"Service URL: {service_url}")
        print(f"Health check: {service_url}/health")
        print(f"Prediction endpoint: {service_url}/predict")
        print(f"Vertex AI compatible endpoint: {service_url}/v1/models/model:predict")
        
    except Exception as deploy_e:
        print(f"Cloud Run deployment failed: {deploy_e}")
        raise
    finally:
        # 8. Cleanup temporary file
        try:
            os.unlink(local_model_path)
            print("Temporary model file cleaned up")
        except Exception as cleanup_e:
            print(f"Cleanup warning: {cleanup_e}")