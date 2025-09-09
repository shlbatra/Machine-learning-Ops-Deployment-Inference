#!/bin/bash

# Setup bastion host for GCP Managed Kafka external access
set -e

PROJECT_ID="deeplearning-sahil"
REGION="us-central1"
ZONE="us-central1-a"
BASTION_NAME="kafka-bastion"
KAFKA_CLUSTER="iris-ml-cluster"
SERVICE_ACCOUNT="kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"

echo "Setting up bastion host for Kafka external access..."

# Create bastion host
echo "Creating bastion host: $BASTION_NAME"
gcloud compute instances create $BASTION_NAME \
    --zone=$ZONE \
    --machine-type=e2-micro \
    --network-interface=network-tier=PREMIUM,subnet=default \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=10GB \
    --boot-disk-type=pd-balanced \
    --metadata=enable-oslogin=true \
    --tags=kafka-bastion \
    --project=$PROJECT_ID

# Create firewall rule for SSH access (0.0.0.0/0 \)
echo "Creating firewall rule for SSH access..."
gcloud compute firewall-rules create allow-ssh-kafka-bastion \
    --allow=tcp:22 \
    --source-ranges=10.0.0.224/32 \
    --target-tags=kafka-bastion \
    --project=$PROJECT_ID \
    --description="Allow SSH access to Kafka bastion host"


echo "Bastion host setup complete!"
echo ""
echo "To connect via SSH tunnel:"
echo "gcloud compute ssh $BASTION_NAME --zone=$ZONE --project=$PROJECT_ID -- -L 9092:bootstrap.$KAFKA_CLUSTER.$REGION.managedkafka.$PROJECT_ID.cloud.goog:9092 -N"
echo "gcloud compute ssh kafka-bastion --zone=us-central1-a --project=deeplearning-sahil -- -L 9092:bootstrap.iris-ml-cluster.us-central1.managedkafka.deeplearning-sahil.cloud.goog:9092 -N"
echo ""
echo "Then connect your Kafka client to localhost:9092"
echo "Using existing service account key: deeplearning-sahil-e50332de6687.json"