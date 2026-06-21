#!/bin/bash

# Sync all DAGs from dags/ to the Composer DAGs bucket

set -e

PROJECT_ID="deeplearning-sahil"
REGION="us-central1"
ENVIRONMENT_NAME="ml-pipelines-composer"

echo "Fetching Composer DAGs bucket..."
DAGS_BUCKET=$(gcloud composer environments describe $ENVIRONMENT_NAME \
    --location=$REGION \
    --project=$PROJECT_ID \
    --format="value(config.dagGcsPrefix)")

echo "Syncing dags/ to $DAGS_BUCKET/"
gsutil -m rsync -r -d dags/ "$DAGS_BUCKET/"

echo ""
echo "DAGs synced. They should appear in the Airflow UI within ~30 seconds."
