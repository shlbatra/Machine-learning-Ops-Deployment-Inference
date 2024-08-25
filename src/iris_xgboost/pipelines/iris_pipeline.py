import sys
import kfp

sys.path.append("src")

# Project settings
BUCKET = "gs://ml-pipelines-kfp"
PIPELINE_NAME = "pipeline_iris"
PIPELINE_ROOT = f"{BUCKET}/pipeline_root"

@kfp.dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(project_id: str, location: str, bq_dataset: str, bq_table: str):
    
    # Import components
    from components.data import load_data
    from components.evaluation import choose_best_model
    from components.models import decision_tree, random_forest
    from components.register import upload_model

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

    upload_model(
        project_id=project_id,
        location=location,
        model=choose_model_op.outputs["best_model"],
    ).set_display_name("Register Model")


if __name__ == "__main__":
    # Pipeline compilation

    aip.init(project=PROJECT_ID)

    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=f"pipeline.yaml", pipeline_name=PIPELINE_NAME
    )
    job=aip.PipelineJob(
        display_name=PIPELINE_NAME,
        template_name=,
        pipeline_root=PIPELINE_ROOT
    )
    job.run(service_account=)