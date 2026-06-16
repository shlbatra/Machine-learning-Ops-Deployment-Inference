from kfp.dsl import Input, Model, component, Output
import ml_pipelines_kfp.iris_xgboost.constants as _constants


@component(base_image=_constants.IMAGE_NAME)
def get_model(
    project_id: str,
    location: str,
    model_name: str,
    latest_model: Output[Model],
):
    from google.cloud import aiplatform, aiplatform_v1
    import fsspec
    from ml_pipelines_kfp.log import get_logger

    logger = get_logger(__name__)

    aiplatform.init(project=project_id, location=location)

    client = aiplatform_v1.ModelServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )

    parent_models = list(client.list_models(request={
        "parent": f"projects/{project_id}/locations/{location}",
        "filter": f"display_name={model_name}",
    }))

    if not parent_models:
        logger.error(f"Could not find model: {model_name}")
        return

    blessed = client.get_model(name=parent_models[0].name + "@blessed")

    logger.info(f"Found blessed model: {blessed.name}, version: {blessed.version_id}, artifact_uri: {blessed.artifact_uri}")

    latest_model_path = latest_model.path.replace("/gcs/", "gs://")
    fs, _ = fsspec.core.url_to_fs(blessed.artifact_uri)
    fs.copy(
        blessed.artifact_uri + "/",
        latest_model_path,
        recursive=True,
    )
