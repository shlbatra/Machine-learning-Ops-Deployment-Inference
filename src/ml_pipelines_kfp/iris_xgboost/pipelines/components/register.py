from kfp.dsl import Input, Model, component, Output, Artifact
import ml_pipelines_kfp.iris_xgboost.constants as _constants


@component(base_image=_constants.IMAGE_NAME)
def upload_model(
    project_id: str,
    location: str,
    model: Input[Model],
    schema: Input[Artifact],
    model_name: str,
    image_name: str,
    vertex_model: Output[Artifact],
):
    from google.cloud import aiplatform, aiplatform_v1
    import fsspec, gcsfs
    from ml_pipelines_kfp.log import get_logger

    logger = get_logger(__name__)

    aiplatform.init(project=project_id, location=location)

    client = aiplatform_v1.ModelServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )

    request = {
        "parent": f"projects/{project_id}/locations/{location}",
        "filter": f"display_name={model_name}",
    }
    results = list(client.list_models(request=request))

    if results:
        parent_model = results[0]
    else:
        parent_model = None

    container_spec = aiplatform_v1.types.model.ModelContainerSpec(
        image_uri=image_name,
        ports=[{"container_port": 8080}],
        predict_route="/predict",
        health_route="/health/live",
    )

    artifact_uri = schema.path.replace("/gcs/", "gs://")
    instance_schema_filename = "instance.yaml"
    prediction_schema_filename = "prediction.yaml"
    parameters_schema_filename = "parameters.yaml"
    fs, _ = fsspec.core.url_to_fs(artifact_uri)
    if isinstance(fs, gcsfs.GCSFileSystem):
        instance_schema_uri = f"{artifact_uri}/{instance_schema_filename}"
        prediction_schema_uri = f"{artifact_uri}/{prediction_schema_filename}"
        parameters_schema_uri = f"{artifact_uri}/{parameters_schema_filename}"

        predict_schemata = aiplatform_v1.PredictSchemata(
            instance_schema_uri=(
                instance_schema_uri if fs.exists(instance_schema_uri) else None
            ),
            parameters_schema_uri=(
                parameters_schema_uri if fs.exists(parameters_schema_uri) else None
            ),
            prediction_schema_uri=(
                prediction_schema_uri if fs.exists(prediction_schema_uri) else None
            ),
        )
    else:
        predict_schemata = None

    new_model = aiplatform_v1.Model(
        display_name=model_name,
        container_spec=container_spec,
        artifact_uri=model.path.replace("/gcs/", "gs://"),
        predict_schemata=predict_schemata,
        version_aliases=["blessed"],
    )

    result = client.upload_model(
        request=dict(
            parent=f"projects/{project_id}/locations/{location}",
            parent_model=parent_model.name if parent_model else None,
            model=new_model,
        ),
        timeout=1800,
    ).result()

    vertex_model.metadata["registered"] = True
    vertex_model.metadata["alias"] = "blessed"

    logger.info(f"Model uploaded successfully: {result}")
