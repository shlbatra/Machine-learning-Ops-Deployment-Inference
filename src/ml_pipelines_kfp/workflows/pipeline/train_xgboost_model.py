import google.cloud.aiplatform as aip


from kfp import dsl, compiler
from src.workflows.config import get_base_pipeline_config
from src.workflows.components.model import train_xgboost_model
from src.training.common_config import TNT_FEATURES, TNT_CATEGORICAL_FEATURES

from typing import List


@dsl.pipeline(name="xgboost-training", description="xgboost Training Pipeline")
def pipeline(
    features: List[str],
    categorical_features: List[str],
    target: str,
    project_id: str,
    query_string: str,
):
    train_xgboost_model(
        features=features,
        categorical_features=categorical_features,
        target=target,
        project_id=project_id,
        query_string=query_string,
    )


def main():
    pipeline_config = get_base_pipeline_config(
        pipeline_name="xgboost-training",
        pipeline_filename="xgboost_training_pipeline.json",
    )

    pipeline_parms = {
        "features": TNT_FEATURES,
        "categorical_features": TNT_CATEGORICAL_FEATURES,
        "target": "time_in_transit_days",
        "project_id": pipeline_config.project_id,
        "query_string": "SELECT * FROM `test` where mod(order_id, 30000) = 0",
    }

    aip.init(
        project=pipeline_config.project_id, staging_bucket=pipeline_config.bucket_uri
    )

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=pipeline_config.pipeline_filename,
        pipeline_name=pipeline_config.pipeline_name,
        pipeline_parameters=pipeline_parms,
    )
    job = aip.PipelineJob(
        display_name=pipeline_config.pipeline_name,
        template_path=pipeline_config.pipeline_filename,
        pipeline_root=pipeline_config.pipeline_root,
        enable_caching=True,
    )
    job.run(service_account=pipeline_config.pipeline_service_account)


if __name__ == "__main__":
    main()
