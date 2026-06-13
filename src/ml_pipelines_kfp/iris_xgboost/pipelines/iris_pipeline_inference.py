import argparse
import sys
import os
import kfp
import google.cloud.aiplatform as aip
from google.oauth2 import service_account

from ml_pipelines_kfp.iris_xgboost.constants import (
    PIPELINE_NAME,
    PIPELINE_ROOT,
    MODEL_NAME,
    PROJECT_ID,
    REGION,
    SERVICE_ACCOUNT,
    BQ_DATASET,
    BQ_TABLE,
    BQ_TABLE_PREDICTIONS,
    SERVICE_ACCOUNT_PATH,
)


@kfp.dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(
    project_id: str,
    location: str,
    bq_dataset: str,
    bq_table: str,
    bq_table_predictions: str,
):

    # Import components
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.get_model import get_model
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.inference import (
        inference_model,
    )

    # Start pipeline definition

    get_model_op = get_model(
        project_id=project_id, location=location, model_name=MODEL_NAME
    ).set_display_name("Get Model")

    inference_op = (
        inference_model(
            project_id=project_id,
            location=location,
            model=get_model_op.outputs["latest_model"],
            bq_dataset=bq_dataset,
            bq_table=bq_table,
            bq_table_predictions=bq_table_predictions,
        )
        .set_display_name("Inference Model")
        .after(get_model_op)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile and submit the Iris inference pipeline to Vertex AI"
    )
    parser.add_argument("--project-id", default=PROJECT_ID)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--bq-dataset", default=BQ_DATASET)
    parser.add_argument("--bq-table", default=BQ_TABLE)
    parser.add_argument("--bq-table-predictions", default=BQ_TABLE_PREDICTIONS)
    parser.add_argument("--service-account-path", default=SERVICE_ACCOUNT_PATH)
    cli = parser.parse_args()

    credentials = service_account.Credentials.from_service_account_file(
        filename=cli.service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    aip.init(project=cli.project_id, credentials=credentials)

    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.yaml",
        pipeline_name=PIPELINE_NAME,
    )
    job = aip.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path="pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
        parameter_values={
            "bq_dataset": cli.bq_dataset,
            "bq_table": cli.bq_table,
            "bq_table_predictions": cli.bq_table_predictions,
            "location": cli.region,
            "project_id": cli.project_id,
        },
        credentials=credentials,
    )
    job.submit(service_account=SERVICE_ACCOUNT)
