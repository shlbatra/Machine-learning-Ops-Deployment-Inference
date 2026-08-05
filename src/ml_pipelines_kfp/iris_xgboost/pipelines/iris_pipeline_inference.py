import argparse
import sys
import os
import kfp
import google.cloud.aiplatform as aip
import google.auth

from ml_pipelines_kfp.iris_xgboost.constants import (
    PIPELINE_NAME,
    PIPELINE_ROOT,
    MODEL_NAME,
    PROJECT_ID,
    REGION,
    SERVICE_ACCOUNT,
    BQ_DATASET,
    BQ_FEATURE_TABLE,
    BQ_TABLE_PREDICTIONS,
)


def coalesce(*args):
    return next((a for a in args if a is not None), None)


# GPU resource config for the inference step. Module-level so the pipeline body reads
# them when the graph is built. KFP's @dsl.pipeline decorator builds the graph eagerly
# at decoration time, so these are reassigned from CLI args in __main__ *before* the
# decorator is applied there. ACCELERATOR_TYPE == "" means CPU-only.
ACCELERATOR_TYPE = ""
ACCELERATOR_COUNT = 0


def pipeline(
    project_id: str,
    location: str,
    bq_dataset: str,
    bq_feature_table: str,
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
            bq_feature_table=bq_feature_table,
            bq_table_predictions=bq_table_predictions,
        )
        .set_display_name("Inference Model")
        .after(get_model_op)
    )

    # Attach a GPU to the inference step when configured. Vertex AI auto-selects a
    # compatible (n1-*) machine for the accelerator.
    if ACCELERATOR_TYPE:
        inference_op.set_accelerator_type(ACCELERATOR_TYPE)
        inference_op.set_accelerator_limit(ACCELERATOR_COUNT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile and submit the Iris inference pipeline to Vertex AI"
    )
    parser.add_argument("--project-id", default=PROJECT_ID)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--bq-dataset", default=BQ_DATASET)
    parser.add_argument("--bq-feature-table", default=BQ_FEATURE_TABLE)
    parser.add_argument("--bq-table-predictions", default=BQ_TABLE_PREDICTIONS)
    parser.add_argument("--accelerator-type", default="",
                        help="GPU type to attach to the inference step "
                             "(e.g., NVIDIA_TESLA_T4, NVIDIA_L4). "
                             "Blank = CPU-only. Vertex auto-selects a compatible machine.")
    parser.add_argument("--accelerator-count", default="0",
                        help="Number of GPUs per step (one of 0, 1, 2, 4, 8, 16)")
    cli = parser.parse_args()

    # Resolve GPU config into the module-level globals read by pipeline() at build time.
    ACCELERATOR_TYPE = coalesce(cli.accelerator_type, "")
    ACCELERATOR_COUNT = int(coalesce(cli.accelerator_count, "0"))
    if ACCELERATOR_TYPE and ACCELERATOR_COUNT not in (1, 2, 4, 8, 16):
        parser.error("--accelerator-count must be one of 1, 2, 4, 8, 16 when a GPU is set")

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    aip.init(project=cli.project_id, credentials=credentials)

    # Apply the KFP pipeline decorator here (not at module top) so the graph is built
    # AFTER the GPU config above is resolved — KFP builds the graph eagerly at
    # decoration time, so decorating at import would freeze it with CPU-only config.
    pipeline_func = kfp.dsl.pipeline(
        name=f"{PIPELINE_NAME}-inference", pipeline_root=PIPELINE_ROOT
    )(pipeline)

    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path="pipeline.yaml",
        pipeline_name=f"{PIPELINE_NAME}-inference",
    )
    job = aip.PipelineJob(
        display_name=f"{PIPELINE_NAME}-inference",
        template_path="pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
        parameter_values={
            "bq_dataset": cli.bq_dataset,
            "bq_feature_table": cli.bq_feature_table,
            "bq_table_predictions": cli.bq_table_predictions,
            "location": cli.region,
            "project_id": cli.project_id,
        },
        credentials=credentials,
    )
    job.submit(service_account=SERVICE_ACCOUNT)
    job.wait()
