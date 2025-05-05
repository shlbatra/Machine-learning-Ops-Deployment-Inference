import sys
import kfp
import google.cloud.aiplatform as aip

# Project settings
BUCKET = "gs://sb-vertex"
PIPELINE_NAME = "pipeline-iris"
PIPELINE_ROOT = f"{BUCKET}/pipeline_root"
REGION = "us-central1"
PROJECT_ID = "deeplearning-sahil"
SERVICE_ACCOUNT = "kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"
MODEL_NAME = "Iris-Classifier-XGBoost"


@kfp.dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(project_id: str, location: str, bq_dataset: str, bq_table: str):
    
    # Import components
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.get_model import get_model
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.inference import inference_model

    # Start pipeline definition

    get_model_op = get_model(
        project_id=PROJECT_ID,
        location=REGION,
        model_name=MODEL_NAME
    ).set_display_name("Get Model")

    inference_op = inference_model(
        project_id=PROJECT_ID,
        location=REGION,
        model=get_model_op.outputs["latest_model"],
        bq_dataset="ml_dataset",
        bq_table="iris_inference"
    ).set_display_name("Inference Model").after(get_model_op)

if __name__ == "__main__":
    # Pipeline compilation
    sys.path.append("src")

    aip.init(project=PROJECT_ID)

    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="pipeline.yaml", pipeline_name=PIPELINE_NAME
    )
    job=aip.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path="pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
        parameter_values={
            "bq_dataset":"ml_dataset",
            "bq_table":"iris",
            "location":REGION,
            "project_id":PROJECT_ID
        }
    )
    job.run(service_account=SERVICE_ACCOUNT)