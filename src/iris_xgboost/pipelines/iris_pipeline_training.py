import sys
import kfp
import google.cloud.aiplatform as aip

# Project settings
BUCKET = "gs://ml-pipelines-kfp"
PIPELINE_NAME = "pipeline-iris"
PIPELINE_ROOT = f"{BUCKET}/pipeline_root"
REGION = "us-east1"
PROJECT_ID = "ml-pipelines-project-433602"
SERVICE_ACCOUNT = "ml-pipelines-sa@ml-pipelines-project-433602.iam.gserviceaccount.com"
MODEL_NAME = "Iris-Classifier-XGBoost-2"

@kfp.dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(project_id: str, location: str, bq_dataset: str, bq_table: str):
    
    # Import components
    from src.iris_xgboost.pipelines.components.data import load_data
    from src.iris_xgboost.pipelines.components.evaluation import choose_best_model
    from src.iris_xgboost.pipelines.components.models import decision_tree, random_forest
    from src.iris_xgboost.pipelines.components.register import upload_model
    from src.iris_xgboost.pipelines.components.deploy import deploy_model

    # Start pipeline definition
    data_op = load_data(
        project_id=project_id, bq_dataset=bq_dataset, bq_table=bq_table
    ).set_display_name("Load data from BigQuery")

    dt_op = decision_tree(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Decision Tree")

    rf_op = random_forest(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Random Forest")

    choose_model_op = choose_best_model(
        test_dataset=data_op.outputs["test_dataset"],
        decision_tree_model=dt_op.outputs["output_model"],
        random_forest_model=rf_op.outputs["output_model"],
    ).set_display_name("Select best Model")

    upload_model_op = upload_model(
        project_id=project_id,
        location=location,
        model=choose_model_op.outputs["best_model"],
        model_name=MODEL_NAME
    ).set_display_name("Register Model")

    deploy_model_op = deploy_model(
        project_id=project_id,
        location=location,
        model=choose_model_op.outputs["best_model"],
        endpoint_name="iris-model-endpoint",
        model_name=MODEL_NAME
    ).set_display_name("Deploy Model").after(upload_model_op)


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