import argparse
import sys
import os
import kfp
import google.cloud.aiplatform as aip
import google.auth
import ml_pipelines_kfp.iris_xgboost.constants as _constants
from ml_pipelines_kfp.iris_xgboost.constants import (
    ENV,
    PIPELINE_NAME,
    REPO_ROOT,
    PIPELINE_ROOT,
    MODEL_NAME,
    IMAGE_NAME,
    PROJECT_ID,
    REGION,
    SERVICE_ACCOUNT,
    ENDPOINT_NAME,
    BQ_DATASET,
    BQ_TABLE,
    BQ_FEATURE_TABLE,
    FASTAPI_IMAGE_NAME,
)
def coalesce(*args):
    return next((a for a in args if a is not None), None)


@kfp.dsl.pipeline(name=f"{PIPELINE_NAME}-training", pipeline_root=PIPELINE_ROOT)
def pipeline(project_id: str, location: str, bq_dataset: str, bq_feature_table: str):

    from ml_pipelines_kfp.iris_xgboost.pipelines.components.data import (
        load_data_from_feature_store,
    )
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.schema import load_schema
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.evaluation import (
        choose_best_model,
    )
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.models import (
        decision_tree,
        random_forest,
    )
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.register import upload_model
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.deploy import (
        deploy_blessed_model_to_fastapi,
    )

    data_op = load_data_from_feature_store(
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_feature_table=bq_feature_table,
    ).set_display_name("Load data from Feature Store")

    schema_load = load_schema(repo_root=REPO_ROOT).set_display_name(
        "Load schema relevant to model"
    )

    dt_op = decision_tree(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Decision Tree")

    rf_op = random_forest(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Random Forest")

    choose_model_op = (
        choose_best_model(
            test_dataset=data_op.outputs["test_dataset"],
            decision_tree_model=dt_op.outputs["output_model"],
            random_forest_model=rf_op.outputs["output_model"],
        )
        .set_display_name("Select best Model")
        .after(schema_load)
    )

    upload_model_op = upload_model(
        project_id=project_id,
        location=location,
        model=choose_model_op.outputs["best_model"],
        schema=schema_load.outputs["gcs_schema"],
        model_name=MODEL_NAME,
        image_name=FASTAPI_IMAGE_NAME,
    ).set_display_name("Register Model")

    deploy_model_op = (
        deploy_blessed_model_to_fastapi(
            project_id=project_id,
            location=location,
            model_name=MODEL_NAME,
            service_name=f"iris-classifier-xgboost-service{'' if ENV == 'prod' else '-staging'}",
            fastapi_image_name=FASTAPI_IMAGE_NAME,
        )
        .set_display_name("Deploy Blessed Model to FastAPI")
        .after(upload_model_op)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile and submit the Iris training pipeline to Vertex AI"
    )
    parser.add_argument("--project-id")
    parser.add_argument("--region")
    parser.add_argument("--pipeline-name")
    parser.add_argument("--pipeline-root")
    parser.add_argument("--model-name")
    parser.add_argument("--image-name")
    parser.add_argument("--fastapi-image-name")
    parser.add_argument("--bq-dataset")
    parser.add_argument("--bq-table")
    parser.add_argument("--bq-feature-table")
    parser.add_argument("--service-account")
    cli = parser.parse_args()

    # Resolve each param: CLI > env var > constants.py
    # _constants.IMAGE_NAME needed: deploy.py/schema.py use it as base_image in @component decorator
    IMAGE_NAME = _constants.IMAGE_NAME = coalesce(cli.image_name, _constants.IMAGE_NAME)
    FASTAPI_IMAGE_NAME = coalesce(cli.fastapi_image_name, FASTAPI_IMAGE_NAME)
    pipeline_name = coalesce(cli.pipeline_name, f"{PIPELINE_NAME}-training")
    pipeline_root = coalesce(cli.pipeline_root, PIPELINE_ROOT)
    bq_dataset = coalesce(cli.bq_dataset, BQ_DATASET)
    bq_table = coalesce(cli.bq_table, BQ_TABLE)
    bq_feature_table = coalesce(cli.bq_feature_table, BQ_FEATURE_TABLE)
    MODEL_NAME = coalesce(cli.model_name, os.getenv("MODEL_NAME"), MODEL_NAME)
    project_id = coalesce(cli.project_id, os.getenv("PROJECT_ID"), PROJECT_ID)
    region = coalesce(cli.region, os.getenv("REGION"), REGION)
    sa_email = coalesce(cli.service_account, os.getenv("SERVICE_ACCOUNT"), SERVICE_ACCOUNT)
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    aip.init(project=project_id, location=region, credentials=credentials)

    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.yaml",
        pipeline_name=pipeline_name,
    )

    job = aip.PipelineJob(
        display_name=pipeline_name,
        template_path="pipeline.yaml",
        pipeline_root=pipeline_root,
        enable_caching=False,
        parameter_values={
            "bq_dataset": bq_dataset,
            "bq_feature_table": bq_feature_table,
            "location": region,
            "project_id": project_id,
        },
        credentials=credentials,
    )
    job.submit(service_account=sa_email)
    job.wait()
