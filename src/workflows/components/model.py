from typing import Dict, List, Optional

from src.workflows.components.constants import BASE_IMAGE

from kfp.dsl import Artifact, Input, Output, component, Model
from google_cloud_pipeline_components.types import artifact_types

# HACK This unused import allows mocking during unit testing
import google.cloud.aiplatform as aip  # noqa: F401


@component(base_image=BASE_IMAGE)
def upload_model_version(
    project_id: str,
    location: str,
    display_name: str,
    serving_container_image_uri: str,
    serving_container_predict_route: str,
    serving_container_health_route: str,
    serving_container_ports: List[int],
    serving_container_environment_variables: Dict[str, str],
    model: Output[Artifact],
):
    """Uploads a new custom model version.
    If model_display_name can't be found, a new model is created in the registry.
    Returns an artifact compatible with google.VertexModel.

    This is a workaround for this issue: https://github.com/kubeflow/pipelines/issues/8882
    Right now, when uploading a new model version by specify a `parent_model` in
    the ModelUploadOp, the default model is not updated, which causes the Deploy
    step to redeploy the old version.
    """
    import google.cloud.aiplatform as aip  # noqa: F811
    from typing import Optional

    aip.init(project=project_id, location=location)

    def parent_model(display_name: str) -> Optional[str]:
        models = aip.Model.list(filter=f"display_name={display_name}", project=project_id, location=location)
        if len(models) == 0:
            return None
        elif len(models) == 1:
            return models[0].resource_name
        else:
            raise RuntimeError(f"Multiple models have the name {display_name}")

    aip_model = aip.Model.upload(
        project=project_id,
        location=location,
        is_default_version=True,
        parent_model=parent_model(display_name),
        display_name=display_name,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route=serving_container_predict_route,
        serving_container_health_route=serving_container_health_route,
        serving_container_ports=serving_container_ports,
        serving_container_environment_variables=serving_container_environment_variables,
    )
    aip_model.wait()
    model.uri = f"https://{location}-aiplatform.googleapis.com/v1/" + aip_model.resource_name
    model.metadata["resourceName"] = aip_model.resource_name


@component(base_image=BASE_IMAGE)
def copy_model(
    project_id: str, location: str, destination_location: str, model: Input[Artifact], copied_model: Output[Artifact]
):
    import google.cloud.aiplatform as aip  # noqa: F811

    if location == destination_location:
        copied_model.uri = model.uri
        copied_model.metadata["resourceName"] = model.metadata["resourceName"]
        return

    aip.init(project=project_id, location=location)
    model_to_copy = aip.Model(model.metadata["resourceName"])
    new_model = model_to_copy.copy(destination_location=destination_location)
    copied_model.uri = f"https://{location}-aiplatform.googleapis.com/v1/" + new_model.resource_name
    copied_model.metadata["resourceName"] = new_model.resource_name


@component(base_image=BASE_IMAGE)
def train_xgboost_model(
    features: List[str],
    categorical_features: List[str],
    target: str,
    model: Output[Model],
    project_id: str,
    table_name: Optional[str] = None,
    query_string: Optional[str] = None,
    hyperparameter_optimization: Optional[bool] = False,
):

    from src.workflows.utils.models.xgboost import XGBoostTrainer
    from src.workflows.utils.data_handling.pandas import PandasDataHandler, DatasetFeaturesConfig

    features_config = DatasetFeaturesConfig(features=features, categorical_features=categorical_features, target=target)
    data_handler = PandasDataHandler()
    data_handler.set_dataset_features_config(features_config)

    if table_name and query_string:
        raise ValueError("Only one of table_name or query_string should be provided")
    elif table_name:
        data_handler.set_dataset_from_BQ_table(project_id, table_name)
    elif query_string:
        data_handler.set_dataset_from_BQ_query(project_id, query_string)
    else:
        raise ValueError("Either table_name or query_string should be provided")

    X, y = data_handler.get_splits()

    trainer = XGBoostTrainer(X, y)
    if hyperparameter_optimization:
        trainer.fit()
    else:
        trainer.train()

    _, _metrics, _best_params = trainer.export()
    trainer.save_model(model.path)
    for v in [_metrics, _best_params]:
        for key, value in v.items():
            model.metadata[key] = value

    model.metadata["features"] = features
    model.metadata["categorical_features"] = categorical_features
    model.metadata["target"] = target


@component(base_image=BASE_IMAGE)
def batch_predict_xgboost_model(
    model: Input[Model],
    project_id: str,
    features: List[str],
    categorical_features: List[str],
    output_table_name: str,
    output_table: Output[artifact_types.BQTable],
    table_name: Optional[str] = None,
    query_string: Optional[str] = None,
    target: str = "time_in_transit_days",
):
    from src.workflows.utils.models.xgboost import XGBoostTrainer
    from src.workflows.utils.data_handling.pandas import PandasDataHandler, DatasetFeaturesConfig

    features_config = DatasetFeaturesConfig(features=features, categorical_features=categorical_features, target=target)
    data_handler = PandasDataHandler()
    data_handler.set_dataset_features_config(features_config)

    if table_name and query_string:
        raise ValueError("Only one of table_name or query_string should be provided")
    elif table_name:
        data_handler.set_dataset_from_BQ_table(project_id, table_name)
    elif query_string:
        data_handler.set_dataset_from_BQ_query(project_id, query_string)
    else:
        raise ValueError("Either table_name or query_string should be provided")

    X, y = data_handler.get_splits()

    trainer = XGBoostTrainer(X, y)
    trainer.load_model(model.uri)
    y_pred = trainer.predict()
    data_handler.dataset["predicted_time_in_transit_days"] = y_pred

    data_handler.save_to_bigquery(project_id, output_table_name, if_exists="replace")

    output_table_ids = output_table_name.split(".")

    output_table.uri = f"https://www.googleapis.com/bigquery/v2/projects/{output_table_ids[0]}/datasets/{output_table_ids[1]}/tables/{output_table_ids[2]}"
    output_table.metadata["datasetId"] = output_table_ids[1]
    output_table.metadata["tableId"] = output_table_ids[2]
    output_table.metadata["projectId"] = output_table_ids[0]
