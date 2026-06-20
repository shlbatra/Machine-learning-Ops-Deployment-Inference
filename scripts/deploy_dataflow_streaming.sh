#!/bin/bash

# Deploy Dataflow streaming job for real-time Iris inference using FastAPI service
set -e

# Environment: staging (default) or prod
ENV=${1:-staging}

# Configuration
PROJECT_ID="deeplearning-sahil"
REGION="us-central1"
PUBSUB_TOPIC="projects/$PROJECT_ID/topics/iris-inference-data"
TEMP_LOCATION="gs://sb-vertex/temp"
STAGING_LOCATION="gs://sb-vertex/staging"
SERVICE_ACCOUNT="kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"

if [ "$ENV" = "prod" ]; then
  SERVICE_URL="https://iris-classifier-xgboost-service-zoxyfmo73q-uc.a.run.app"
  JOB_PREFIX="iris-streaming-inference"
  OUTPUT_TABLE="$PROJECT_ID:ml_dataset.iris_predictions_streaming"
else
  SERVICE_URL="<staging-cloud-run-url>"
  JOB_PREFIX="iris-streaming-inference-staging"
  OUTPUT_TABLE="$PROJECT_ID:ml_dataset.iris_predictions_streaming_staging"
fi

JOB_NAME="$JOB_PREFIX-$(date +%Y%m%d-%H%M%S)"

echo "Deploying Dataflow streaming job ($ENV) for real-time inference using FastAPI service..."
echo "Service URL: $SERVICE_URL"

# Run the Dataflow job
echo "Starting Dataflow streaming job: $JOB_NAME"
python src/ml_pipelines_kfp/dataflow/iris_inference_pipeline.py \
    --input_topic $PUBSUB_TOPIC \
    --output_table $OUTPUT_TABLE \
    --project_id $PROJECT_ID \
    --region $REGION \
    --service_url $SERVICE_URL \
    --runner DataflowRunner \
    --job_name $JOB_NAME \
    --temp_location $TEMP_LOCATION \
    --staging_location $STAGING_LOCATION \
    --service_account_email $SERVICE_ACCOUNT \
    --use_public_ips \
    --max_num_workers 3 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --streaming \
    --enable_streaming_engine \
    --experiments use_runner_v2

echo "Dataflow job submitted successfully!"
echo "Job name: $JOB_NAME"
echo "Environment: $ENV"
echo "Monitor at: https://console.cloud.google.com/dataflow/jobs/$REGION/$JOB_NAME?project=$PROJECT_ID"
echo ""
echo "To test the pipeline:"
echo "1. Run the producer: python src/ml_pipelines_kfp/iris_xgboost/pubsub_producer.py --project-id=$PROJECT_ID"
echo "2. Check BigQuery table: $OUTPUT_TABLE"
echo "3. Monitor Dataflow job in the console"
echo ""
echo "To stop the job:"
echo "gcloud dataflow jobs cancel $JOB_NAME --region=$REGION --project=$PROJECT_ID"
