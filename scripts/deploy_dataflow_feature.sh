#!/bin/bash

# Deploy Dataflow streaming job for ingesting Pub/Sub data into the Feature Store
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
ONLINE_BATCH_SIZE=100

if [ "$ENV" = "prod" ]; then
  JOB_PREFIX="iris-streaming-features"
  OUTPUT_TABLE="$PROJECT_ID:ml_dataset.iris_features"
else
  JOB_PREFIX="iris-streaming-features-staging"
  OUTPUT_TABLE="$PROJECT_ID:ml_dataset.iris_features"
fi

JOB_NAME="$JOB_PREFIX-$(date +%Y%m%d-%H%M%S)"

echo "Deploying Dataflow feature pipeline ($ENV)..."
echo "Output table: $OUTPUT_TABLE"

python src/dataflow/iris_feature_pipeline.py \
    --input_topic $PUBSUB_TOPIC \
    --output_table $OUTPUT_TABLE \
    --project_id $PROJECT_ID \
    --region $REGION \
    --online_batch_size $ONLINE_BATCH_SIZE \
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
    --experiments use_runner_v2 \
    --no_wait

echo "Dataflow job submitted successfully!"
echo "Job name: $JOB_NAME"
echo "Environment: $ENV"
echo "Monitor at: https://console.cloud.google.com/dataflow/jobs/$REGION/$JOB_NAME?project=$PROJECT_ID"
echo ""
echo "To test the pipeline:"
echo "1. Publish messages: ./scripts/run_pubsub_producer.sh"
echo "2. Check feature table: SELECT * FROM ml_dataset.iris_features WHERE source = 'streaming'"
echo ""
echo "To stop the job:"
echo "gcloud dataflow jobs cancel $JOB_NAME --region=$REGION --project=$PROJECT_ID"
