import sys
import kfp
import google.cloud.aiplatform as aip
from google.oauth2 import service_account

from ml_pipelines_kfp.iris_xgboost.constants import (
    PIPELINE_NAME, PIPELINE_ROOT, MODEL_NAME, PROJECT_ID, REGION, SERVICE_ACCOUNT, BQ_DATASET, BQ_TABLE, BQ_TABLE_PREDICTIONS, SERVICE_ACCOUNT_PATH)


@kfp.dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(project_id: str, location: str, bq_dataset: str, bq_table: str, bq_table_predictions: str):
    
    # Import components
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.get_model import get_model
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.inference import inference_model

    # Start pipeline definition

    get_model_op = get_model(
        project_id=project_id,
        location=location,
        model_name=MODEL_NAME
    ).set_display_name("Get Model")

    inference_op = inference_model(
        project_id=project_id,
        location=location,
        model=get_model_op.outputs["latest_model"],
        bq_dataset=bq_dataset,
        bq_table=bq_table,
        bq_table_predictions=bq_table_predictions
    ).set_display_name("Inference Model").after(get_model_op)

if __name__ == "__main__":
    # Pipeline compilation
    sys.path.append("src")

        # Set up authentication using service account
    credentials = service_account.Credentials.from_service_account_file(
        filename=SERVICE_ACCOUNT_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    aip.init(project=PROJECT_ID, credentials=credentials)

    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="pipeline.yaml", pipeline_name=PIPELINE_NAME
    )
    job=aip.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path="pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
        parameter_values={
            "bq_dataset":BQ_DATASET,
            "bq_table":BQ_TABLE,
            "bq_table_predictions":BQ_TABLE_PREDICTIONS,
            "location":REGION,
            "project_id":PROJECT_ID
        },
        credentials=credentials
    )
    job.submit(service_account=SERVICE_ACCOUNT)