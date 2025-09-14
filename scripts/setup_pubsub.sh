#!/bin/bash

# Setup script for Cloud Pub/Sub
# This script creates the necessary Pub/Sub topics and subscriptions for the ML pipeline

set -e

# Configuration from constants.py
PROJECT_ID="deeplearning-sahil"
TOPIC_NAME="iris-inference-data"
SUBSCRIPTION_NAME="iris-inference-data-sub"
SERVICE_ACCOUNT="kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"

echo "Setting up Cloud Pub/Sub..."

# Enable the Pub/Sub API
echo "Enabling Cloud Pub/Sub API..."
gcloud services enable pubsub.googleapis.com --project=$PROJECT_ID

# Create the topic
echo "Creating Pub/Sub topic: $TOPIC_NAME"
gcloud pubsub topics create $TOPIC_NAME --project=$PROJECT_ID || echo "Topic already exists"

# Create the subscription
echo "Creating Pub/Sub subscription: $SUBSCRIPTION_NAME"
gcloud pubsub subscriptions create $SUBSCRIPTION_NAME \
    --topic=$TOPIC_NAME \
    --project=$PROJECT_ID \
    --ack-deadline=60 || echo "Subscription already exists"

echo "Pub/Sub setup complete!"
echo ""
echo "Resources created:"
echo "- Topic: $TOPIC_NAME"
echo "- Subscription: $SUBSCRIPTION_NAME"
echo ""
echo "Test the setup:"
echo "python test_pubsub.py"
echo ""
echo "Run the producer:"
echo "python src/ml_pipelines_kfp/iris_xgboost/pubsub_producer.py --project-id=$PROJECT_ID --topic=$TOPIC_NAME"
echo ""
echo "Monitor in GCP Console:"
echo "https://console.cloud.google.com/cloudpubsub/topic/detail/$TOPIC_NAME?project=$PROJECT_ID"