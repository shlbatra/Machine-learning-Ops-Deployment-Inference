#!/bin/bash

# Setup script for Google Artifact Registry
# Creates the Docker repository and grants push/pull access to the CI/CD service account

set -e

# Configuration from constants.py and cicd.yaml
PROJECT_ID="deeplearning-sahil"
LOCATION="us"
REPOSITORY="sahil-experiment-docker-images"
SERVICE_ACCOUNT="kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"

echo "Setting up Artifact Registry..."

# Enable the Artifact Registry API
echo "Enabling Artifact Registry API..."
gcloud services enable artifactregistry.googleapis.com --project=$PROJECT_ID

# Create the Docker repository
echo "Creating Artifact Registry repository: $REPOSITORY"
gcloud artifacts repositories create $REPOSITORY \
    --repository-format=docker \
    --location=$LOCATION \
    --project=$PROJECT_ID \
    --description="Docker images for ML pipelines and FastAPI inference" \
    || echo "Repository already exists"

# Grant the service account write access to push images
echo "Granting Artifact Registry Writer role to $SERVICE_ACCOUNT..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/artifactregistry.writer" \
    --condition=None \
    --quiet

# Grant the service account read access to pull images (for Cloud Run / KFP)
echo "Granting Artifact Registry Reader role to $SERVICE_ACCOUNT..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/artifactregistry.reader" \
    --condition=None \
    --quiet

# Configure Docker to authenticate with Artifact Registry
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${LOCATION}-docker.pkg.dev --quiet

echo ""
echo "Artifact Registry setup complete!"
echo ""
echo "Repository: ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}"
echo ""
echo "Images built by CI/CD:"
echo "  - ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/ml-pipelines-kfp-image:<branch>"
echo "  - ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/fastapi-ml-generic:<branch>"
echo ""
echo "To push manually:"
echo "  docker build -t ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/ml-pipelines-kfp-image:latest -f Dockerfile ."
echo "  docker push ${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/ml-pipelines-kfp-image:latest"
