# Plan: GitHub Action for Dataflow Streaming Deployment

## Context

The Dataflow streaming pipeline (`iris_streaming_pipeline.py`) is currently deployed via a manual script (`scripts/deploy_dataflow_streaming.sh`). This requires someone to run the script locally with GCP credentials. The `service_url` for the FastAPI Cloud Run endpoint is hardcoded and must be manually updated after each Cloud Run deployment.

The goal is to automate this as a GitHub Action that:
- Can be triggered manually (workflow_dispatch) or after a successful Cloud Run deployment
- Dynamically discovers the Cloud Run service URL instead of hardcoding it
- Drains the existing Dataflow job before deploying a new one (streaming jobs run indefinitely)

## Approach

### 1. Create a new workflow file

**New file:** `.github/workflows/deploy-dataflow.yaml`

```yaml
name: Deploy Dataflow Streaming

on:
  workflow_dispatch:
    inputs:
      drain_existing:
        description: "Drain existing streaming job before deploying"
        type: boolean
        default: true

env:
  PROJECT_ID: "deeplearning-sahil"
  REGION: "us-central1"
  PUBSUB_TOPIC: "projects/deeplearning-sahil/topics/iris-inference-data"
  OUTPUT_TABLE: "deeplearning-sahil:ml_dataset.iris_predictions_streaming"
  TEMP_LOCATION: "gs://sb-vertex/temp"
  STAGING_LOCATION: "gs://sb-vertex/staging"
  SERVICE_ACCOUNT_EMAIL: "kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"
  CLOUD_RUN_SERVICE: "iris-classifier-xgboost-service"

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -e .

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}

      - name: Set up gcloud CLI
        uses: google-github-actions/setup-gcloud@v1
        with:
          project_id: ${{ env.PROJECT_ID }}

      - name: Discover Cloud Run service URL
        id: service_url
        run: |
          URL=$(gcloud run services describe ${{ env.CLOUD_RUN_SERVICE }} \
            --region=${{ env.REGION }} \
            --project=${{ env.PROJECT_ID }} \
            --format='value(status.url)')
          echo "url=$URL" >> $GITHUB_OUTPUT
          echo "Discovered service URL: $URL"

      - name: Drain existing Dataflow job
        if: ${{ inputs.drain_existing }}
        run: |
          # Find active streaming jobs with the iris-streaming prefix
          ACTIVE_JOBS=$(gcloud dataflow jobs list \
            --region=${{ env.REGION }} \
            --project=${{ env.PROJECT_ID }} \
            --status=active \
            --filter="name:iris-streaming-inference" \
            --format='value(id)')

          if [ -n "$ACTIVE_JOBS" ]; then
            for JOB_ID in $ACTIVE_JOBS; do
              echo "Draining job: $JOB_ID"
              gcloud dataflow jobs drain $JOB_ID \
                --region=${{ env.REGION }} \
                --project=${{ env.PROJECT_ID }}
            done
            echo "Waiting 60s for drain to start..."
            sleep 60
          else
            echo "No active streaming jobs found"
          fi

      - name: Deploy Dataflow streaming job
        run: |
          JOB_NAME="iris-streaming-inference-$(date +%Y%m%d-%H%M%S)"
          echo "Submitting job: $JOB_NAME"

          python src/ml_pipelines_kfp/dataflow/iris_streaming_pipeline.py \
            --input_topic ${{ env.PUBSUB_TOPIC }} \
            --output_table ${{ env.OUTPUT_TABLE }} \
            --project_id ${{ env.PROJECT_ID }} \
            --region ${{ env.REGION }} \
            --service_url ${{ steps.service_url.outputs.url }} \
            --runner DataflowRunner \
            --job_name $JOB_NAME \
            --temp_location ${{ env.TEMP_LOCATION }} \
            --staging_location ${{ env.STAGING_LOCATION }} \
            --service_account_email ${{ env.SERVICE_ACCOUNT_EMAIL }} \
            --use_public_ips \
            --max_num_workers 3 \
            --autoscaling_algorithm THROUGHPUT_BASED \
            --streaming \
            --enable_streaming_engine \
            --experiments use_runner_v2

          echo "Job submitted: $JOB_NAME"
          echo "Monitor: https://console.cloud.google.com/dataflow/jobs/${{ env.REGION }}?project=${{ env.PROJECT_ID }}"
```

### 2. Key design decisions

**Trigger: `workflow_dispatch` (manual)**
- Dataflow streaming jobs run indefinitely — you don't want every push to redeploy
- Manual trigger lets you control when to update the running job
- The `drain_existing` input (default: true) handles the old job gracefully. Draining lets the existing job finish processing in-flight messages before shutting down, avoiding data loss. Cancel is faster but drops messages.

**Dynamic service URL discovery**
- Uses `gcloud run services describe` to get the current Cloud Run URL at deploy time
- Eliminates the hardcoded URL problem in the manual script
- If Cloud Run service doesn't exist yet, this step will fail with a clear error

**Drain vs Cancel**
- The workflow uses `drain` (not `cancel`) for existing jobs. Drain finishes processing buffered messages, then stops. This avoids data loss.
- Waits 60s after issuing drain to let the job start draining before submitting the new one. Pub/Sub handles the gap — unacked messages will be redelivered to the new job.

### 3. Optional: trigger after Cloud Run deploy

If you want to also auto-trigger this after the training pipeline deploys a new model to Cloud Run, add to the `on:` block:

```yaml
on:
  workflow_dispatch:
    inputs:
      drain_existing:
        description: "Drain existing streaming job before deploying"
        type: boolean
        default: true
  workflow_run:
    workflows: ["CI/CD Pipeline"]
    branches: [main]
    types: [completed]
```

This would trigger after the CI/CD pipeline completes on main. However, this only makes sense if the Dataflow pipeline code itself changed — the streaming job doesn't need redeployment just because the model changed (the model is served by Cloud Run, not baked into Dataflow).

### 4. Secrets required

The workflow reuses the existing `GCP_SERVICE_ACCOUNT_KEY` secret already configured for the CI/CD pipeline. No new secrets needed.

The service account (`kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com`) needs these IAM roles:
- `roles/dataflow.admin` — submit and manage Dataflow jobs
- `roles/pubsub.subscriber` — read from Pub/Sub topic
- `roles/bigquery.dataEditor` — write to BigQuery
- `roles/storage.objectAdmin` — read/write temp and staging buckets
- `roles/run.viewer` — discover Cloud Run service URL

Most of these are likely already granted since the same service account runs the KFP pipelines.

## Files to create/modify

| File | Action |
|---|---|
| `.github/workflows/deploy-dataflow.yaml` | **New** — the workflow above |

No changes needed to `iris_streaming_pipeline.py` or any other existing code. The workflow calls the pipeline script the same way the manual script does.

## Verification

1. Push the workflow file to any branch
2. Go to Actions tab in GitHub → "Deploy Dataflow Streaming" → "Run workflow"
3. Check the workflow logs for:
   - Service URL discovery succeeds
   - Drain step finds/skips existing jobs correctly
   - Job submission succeeds and prints job name
4. Verify in GCP Console → Dataflow → Jobs that the new streaming job is running
5. Test end-to-end: run the pub/sub producer locally and check BigQuery for predictions
