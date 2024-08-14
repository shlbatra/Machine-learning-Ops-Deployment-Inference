from typing import List, Dict, Union, Optional

from datetime import datetime
from functools import cached_property

from pydantic import BaseModel, computed_field

from src.workflows.components.constants import GCR_IMAGE_TAG


### Generalized Components
class PipelineConfig(BaseModel):
    project_id: str
    bucket_name: str
    pipeline_service_account: str
    pipeline_name: str
    pipeline_filename: str
    git_commit_hash: str = GCR_IMAGE_TAG
    _now: datetime = datetime.now()

    @computed_field  # type: ignore[misc]
    @cached_property
    def bucket_uri(self) -> str:
        return f"gs://{self.bucket_name}/vertex-ai-pipelines"

    @computed_field  # type: ignore[misc]
    @cached_property
    def pipeline_root(self) -> str:
        return f"{self.bucket_uri}/pipeline_root/shop_promise/"

    @computed_field  # type: ignore[misc]
    @cached_property
    def pipeline_timestamp(self) -> str:
        return self._now.strftime("%Y%m%d%H%M%S")

    @computed_field  # type: ignore[misc]
    @cached_property
    def pipeline_date(self) -> str:
        return self._now.strftime("%Y_%m_%d")


class ExperimentPipelineConfig(PipelineConfig):
    experiment_name: str


class BigqueryClientConfig(BaseModel):
    project_id: str
    bigquery_location: str


class ModifiedGoldenDatasetConfig(BigqueryClientConfig):
    golden_dataset_name: str
    prototype_table_name: str
    prototype_columns: List[str]


class RegisterDatasetConfig(BaseModel):
    project: str
    display_name: str
    location: str
    bq_source: str


class TestDatasetConfig(BigqueryClientConfig):
    table_name: str


class TrainAutoMLModelConfig(BaseModel):
    optimization_prediction_type: str
    project: str
    display_name: str
    target_column: str
    location: str
    column_transformations: List[Dict[str, Dict[str, Union[str, bool]]]]
    predefined_split_column_name: str
    budget_milli_node_hours: int
    disable_early_stopping: bool


class FeatureValidationAutoMLModelConfig(BaseModel):
    project: str
    tnt_features: List[str]


class BatchPredictXGboostModelConfig(BaseModel):
    # model: Model # TODO: Do I add a model here knowing that we're passing in the model if it is trained?
    project_id: str
    features: List[str]
    categorical_features: List[str]
    output_table_name: str
    table_name: Optional[str] = None
    query_string: Optional[str] = None
    target: str


class BatchPredictAutoMLModelConfig(BaseModel):
    project: str
    job_display_name: str
    location: str
    bigquery_source_input_uri: str
    bigquery_destination_output_uri: str
    instances_format: str
    machine_type: str
    starting_replica_count: int
    max_replica_count: int
    predictions_format: str

    @computed_field  # type: ignore[misc]
    @cached_property
    def bigquery_source_input(self) -> str:
        return self.bigquery_source_input_uri.replace("bq://", "")


class MonitoringConfig(BaseModel):
    sampling_rate: float
    monitor_interval: int
    emails: List[str]
    slack_channels: List[str]
    drift_thresholds: Dict


class CreateMonitoringConfig(BaseModel):
    project_id: str
    location: str
    endpoint_name: str
    monitoring_config: MonitoringConfig


class UpdateMonitoringConfig(BaseModel):
    project_id: str
    location: str
    monitoring_name: str
    monitoring_config: MonitoringConfig


class DeleteMonitoringJob(BaseModel):
    project_id: str
    location: str
    monitoring_name: str


class ImportBQTableConfig(BaseModel):
    project_id: str
    dataset_id: str
    table_id: str

    @computed_field  # type: ignore[misc]
    @cached_property
    def full_table_name(self) -> str:
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"

    @computed_field  # type: ignore[misc]
    @cached_property
    def bigquery_table_uri(self) -> str:
        return f"https://www.googleapis.com/bigquery/v2/projects/{self.project_id}/datasets/{self.dataset_id}/tables/{self.table_id}"


class ImportVertexModelConfig(BaseModel):
    project_id: str
    location: str
    ml_model_id: str

    @computed_field  # type: ignore[misc]
    @cached_property
    def resource_name(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}/models/{self.ml_model_id}"

    @computed_field  # type: ignore[misc]
    @cached_property
    def vertex_model_uri(self) -> str:
        return f"https://us-central1-aiplatform.googleapis.com/v1/{self.resource_name}"


### Specific Components


class TrainXGBoostModelConfig(BaseModel):
    features: List[str]
    categorical_features: List[str]
    target: str
    project_id: str
    table_name: Optional[str] = None
    query_string: Optional[str] = None
    hyperparameter_optimization: Optional[bool] = False


class GetCumulativeSumConfig(BigqueryClientConfig):
    output_table_name: str
    input_data_table: str
    cumulative_confidence_threshold: float


class DeliveryTimeFromSlaConfig(BigqueryClientConfig):
    table_name: str
    output_table_name: str


class EvaluatorConfig(BaseModel):
    project_id: str
    table_name: str
    evaluation_type: str


class ValidatePreprocessingConfig(BaseModel):
    base_table: str
    output_table_name: str
    join_column: str
    columns: List[str]
    project_id: str
