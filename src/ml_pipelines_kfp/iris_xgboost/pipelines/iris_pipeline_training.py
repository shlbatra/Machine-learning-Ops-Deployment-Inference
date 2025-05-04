import sys
import kfp
import google.cloud.aiplatform as aip
from src.iris_xgboost.constants import PIPELINE_NAME, PIPELINE_ROOT, MODEL_NAME, IMAGE_NAME, PROJECT_ID, REGION, SERVICE_ACCOUNT, ENDPOINT_NAME

@kfp.dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(project_id: str, location: str, bq_dataset: str, bq_table: str):
    
    # Import components
    from src.iris_xgboost.pipelines.components.data import load_data
    from src.iris_xgboost.pipelines.components.schema import load_schema
    from src.iris_xgboost.pipelines.components.evaluation import choose_best_model
    from src.iris_xgboost.pipelines.components.models import decision_tree, random_forest
    from src.iris_xgboost.pipelines.components.register import upload_model
    from src.iris_xgboost.pipelines.components.deploy import deploy_model

    # Start pipeline definition
    data_op = load_data(
        project_id=project_id, bq_dataset=bq_dataset, bq_table=bq_table
    ).set_display_name("Load data from BigQuery")

    schema_load = load_schema(repo_root="ml_pipelines_kfp").set_display_name("Load schema relevant to model")

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
    ).set_display_name("Select best Model").after(schema_load)

    upload_model_op = upload_model(
        project_id=project_id,
        location=location,
        model=choose_model_op.outputs["best_model"],
        schema=schema_load.outputs["gcs_schema"],
        model_name=MODEL_NAME,
        image_name=IMAGE_NAME
    ).set_display_name("Register Model")

    deploy_model_op = deploy_model(
        project_id=project_id,
        location=location,
        model=choose_model_op.outputs["best_model"],
        vertex_model = upload_model_op.outputs["vertex_model"],
        endpoint_name=ENDPOINT_NAME,
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
    job.submit(service_account=SERVICE_ACCOUNT)