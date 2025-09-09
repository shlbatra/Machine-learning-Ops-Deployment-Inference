#!/bin/bash

# Setup script for GCP Managed Service for Apache Kafka
# This script creates the necessary Kafka cluster and topics for the ML pipeline

set -e

# Configuration from constants.py
PROJECT_ID="deeplearning-sahil"
REGION="us-central1"
CLUSTER_NAME="iris-ml-cluster"
TOPIC_NAME="iris-inference-data"

echo "Setting up GCP Managed Service for Apache Kafka..."

# Enable the Managed Kafka API
echo "Enabling Managed Service for Apache Kafka API..."
gcloud services enable managedkafka.googleapis.com --project=$PROJECT_ID

# Create the Kafka cluster
echo "Creating Kafka cluster: $CLUSTER_NAME"
gcloud managed-kafka clusters create $CLUSTER_NAME \
    --location=$REGION \
    --project=$PROJECT_ID \
    --cpu=3 \
    --memory=3GiB \
    --subnets=projects/$PROJECT_ID/regions/$REGION/subnetworks/default

echo "Waiting for cluster to be ready..."
gcloud managed-kafka clusters describe $CLUSTER_NAME \
    --location=$REGION \
    --project=$PROJECT_ID \
    --format="value(state)"

# Create the topic
echo "Creating Kafka topic: $TOPIC_NAME"
gcloud managed-kafka topics create $TOPIC_NAME \
    --cluster=$CLUSTER_NAME \
    --location=$REGION \
    --project=$PROJECT_ID \
    --partitions=3 \
    --replication-factor=3

# Get cluster connection details
echo "Getting cluster connection details..."
BOOTSTRAP_SERVERS=$(gcloud managed-kafka clusters describe $CLUSTER_NAME \
    --location=$REGION \
    --project=$PROJECT_ID \
    --format="value(gcpConfig.bootstrapServers)")
# bootstrap.iris-ml-cluster.us-central1.managedkafka.deeplearning-sahil.cloud.goog:9092
echo "python src/ml_pipelines_kfp/iris_xgboost/kafka_producer.py --kafka-servers="bootstrap.iris-ml-cluster.us-central1.managedkafka.deeplearning-sahil.cloud.goog:9092""
echo "Kafka cluster setup complete!"
echo "Bootstrap servers: $BOOTSTRAP_SERVERS"
echo ""
echo "Update your constants.py with the actual bootstrap server endpoint:"
echo "KAFKA_BOOTSTRAP_SERVERS = \"$BOOTSTRAP_SERVERS\""
echo ""
echo "To test the producer, run:"
echo "python src/ml_pipelines_kfp/iris_xgboost/kafka_producer.py --kafka-servers=\"$BOOTSTRAP_SERVERS\""