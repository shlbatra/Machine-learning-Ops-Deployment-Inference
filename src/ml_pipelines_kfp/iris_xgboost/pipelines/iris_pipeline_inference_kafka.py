import sys
import kfp
import google.cloud.aiplatform as aip
from google.oauth2 import service_account

from ml_pipelines_kfp.iris_xgboost.constants import (
    PIPELINE_ROOT, MODEL_NAME, PROJECT_ID, REGION, SERVICE_ACCOUNT, 
    BQ_DATASET, BQ_TABLE_PREDICTIONS, SERVICE_ACCOUNT_PATH)


@kfp.dsl.pipeline(name="pipeline-iris-kafka-inference", pipeline_root=PIPELINE_ROOT)
def kafka_inference_pipeline(
    project_id: str, 
    location: str, 
    bq_dataset: str, 
    bq_table_predictions: str,
    kafka_servers: str = "localhost:9092",
    kafka_topic: str = "iris-inference-data",
    kafka_batch_size: int = 100,
    kafka_timeout_seconds: int = 300
):
    
    # Import components
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.get_model import get_model
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.inference import inference_model
    from ml_pipelines_kfp.iris_xgboost.pipelines.components.kafka_data_source import kafka_data_source

    # Step 1: Consume data from Kafka topic
    kafka_data_op = kafka_data_source(
        kafka_servers=kafka_servers,
        topic=kafka_topic,
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_table="iris_kafka_data",  # Temporary table for Kafka data
        batch_size=kafka_batch_size,
        timeout_seconds=kafka_timeout_seconds
    ).set_display_name("Consume Kafka Data")

    # Step 2: Get the latest model
    get_model_op = get_model(
        project_id=project_id,
        location=location,
        model_name=MODEL_NAME
    ).set_display_name("Get Model")

    # Step 3: Run inference on Kafka data
    inference_op = inference_model(
        project_id=project_id,
        location=location,
        model=get_model_op.outputs["latest_model"],
        bq_dataset=bq_dataset,
        bq_table="iris_kafka_data",  # Use Kafka data as input
        bq_table_predictions=bq_table_predictions
    ).set_display_name("Inference on Kafka Data").after(kafka_data_op, get_model_op)


if __name__ == "__main__":
    # Pipeline compilation and execution
    sys.path.append("src")

    # Set up authentication using service account
    credentials = service_account.Credentials.from_service_account_file(
        filename=SERVICE_ACCOUNT_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    aip.init(project=PROJECT_ID, credentials=credentials)

    # Compile pipeline
    pipeline_name = "pipeline-iris-kafka-inference"
    kfp.compiler.Compiler().compile(
        pipeline_func=kafka_inference_pipeline, 
        package_path=f"{pipeline_name}.yaml", 
        pipeline_name=pipeline_name
    )

    # Submit pipeline job
    job = aip.PipelineJob(
        display_name=pipeline_name,
        template_path=f"{pipeline_name}.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
        parameter_values={
            "bq_dataset": BQ_DATASET,
            "bq_table_predictions": BQ_TABLE_PREDICTIONS,
            "location": REGION,
            "project_id": PROJECT_ID,
            "kafka_servers": "kafka:9092",  # Docker service name
            "kafka_topic": "iris-inference-data",
            "kafka_batch_size": 50,
            "kafka_timeout_seconds": 300
        },
        credentials=credentials
    )
    job.submit(service_account=SERVICE_ACCOUNT)