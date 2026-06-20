# Plan: GitHub Action for Dataflow Streaming Deployment

## Context

The Dataflow streaming pipeline (`iris_inference_pipeline.py`) is currently deployed via a manual script (`scripts/deploy_dataflow_streaming.sh`). This requires someone to run the script locally with GCP credentials.

The project now supports **two environments** (staging and prod) controlled by the `ENVIRONMENT` env var, with environment-specific Cloud Run services, BigQuery tables, Dataflow job names, and GCS paths. The deploy script accepts an environment argument (`./scripts/deploy_dataflow_streaming.sh staging|prod`).

The goal is to automate this as a GitHub Action that:
- Can be triggered manually (workflow_dispatch) with environment selection
- Dynamically discovers the environment-specific Cloud Run service URL
- Drains the existing Dataflow job for that environment before deploying a new one

## Environment-Specific Resources

| Resource | Staging | Production |
|---|---|---|
| Cloud Run service | `iris-classifier-xgboost-service-staging` | `iris-classifier-xgboost-service` |
| BQ streaming table | `iris_predictions_streaming_staging` | `iris_predictions_streaming` |
| Dataflow job prefix | `iris-streaming-inference-staging` | `iris-streaming-inference` |
| GCS pipeline root | `gs://sb-vertex/staging/pipeline_root` | `gs://sb-vertex/prod/pipeline_root` |

Shared across both environments:
- Project: `deeplearning-sahil`
- Region: `us-central1`
- Pub/Sub topic: `projects/deeplearning-sahil/topics/iris-inference-data`
- GCS temp/staging: `gs://sb-vertex/temp`, `gs://sb-vertex/staging`
- Service account: `kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com`

## Approach

### 1. Create a new workflow file

**New file:** `.github/workflows/deploy-dataflow.yaml`

```yaml
name: Deploy Dataflow Streaming

on:
  workflow_dispatch:
    inputs:
      environment:
        description: "Target environment"
        type: choice
        options:
          - staging
          - prod
        default: staging
env:
  PROJECT_ID: "deeplearning-sahil"
  REGION: "us-central1"
  PUBSUB_TOPIC: "projects/deeplearning-sahil/topics/iris-inference-data"
  TEMP_LOCATION: "gs://sb-vertex/temp"
  STAGING_LOCATION: "gs://sb-vertex/staging"
  SERVICE_ACCOUNT_EMAIL: "kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set environment-specific variables
        run: |
          if [ "${{ inputs.environment }}" = "prod" ]; then
            echo "CLOUD_RUN_SERVICE=iris-classifier-xgboost-service" >> $GITHUB_ENV
            echo "OUTPUT_TABLE=${{ env.PROJECT_ID }}:ml_dataset.iris_predictions_streaming" >> $GITHUB_ENV
            echo "JOB_PREFIX=iris-streaming-inference" >> $GITHUB_ENV
          else
            echo "CLOUD_RUN_SERVICE=iris-classifier-xgboost-service-staging" >> $GITHUB_ENV
            echo "OUTPUT_TABLE=${{ env.PROJECT_ID }}:ml_dataset.iris_predictions_streaming_staging" >> $GITHUB_ENV
            echo "JOB_PREFIX=iris-streaming-inference-staging" >> $GITHUB_ENV
          fi

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
          echo "Discovered service URL for ${{ inputs.environment }}: $URL"

      - name: Deploy Dataflow streaming job
        run: |
          JOB_NAME="${{ env.JOB_PREFIX }}-$(date +%Y%m%d-%H%M%S)"
          echo "Submitting ${{ inputs.environment }} job: $JOB_NAME"

          python src/ml_pipelines_kfp/dataflow/iris_inference_pipeline.py \
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
          echo "Environment: ${{ inputs.environment }}"
          echo "Monitor: https://console.cloud.google.com/dataflow/jobs/${{ env.REGION }}?project=${{ env.PROJECT_ID }}"
```

### 2. Key design decisions

**Trigger: `workflow_dispatch` (manual) with environment selector**
- Dataflow streaming jobs run indefinitely — you don't want every push to redeploy
- Manual trigger lets you control when to update the running job
- The `environment` input (default: staging) ensures you explicitly choose which environment to deploy to. Defaults to staging to prevent accidental prod deployments.
- User should drain/cancel existing jobs manually before triggering a new deployment
- **Data loss implications:** Drain before redeploying = no duplicates, no data loss. Cancel/crash = some duplicates, no data loss (at-least-once). Pub/Sub redelivers unacked messages to the new job.
- **Model redeployment:** No Dataflow redeployment needed when a new model is deployed to Cloud Run (retrained or new version). Dataflow calls the Cloud Run URL via HTTP — Cloud Run does a rolling update to the new revision behind the same URL. Only redeploy Dataflow if the model's input/output schema changes (different features or response format).

**Environment-specific variable resolution**
- A dedicated step sets `CLOUD_RUN_SERVICE`, `OUTPUT_TABLE`, and `JOB_PREFIX` based on the selected environment
- Mirrors the branching logic in `scripts/deploy_dataflow_streaming.sh` and `src/ml_pipelines_kfp/iris_xgboost/constants.py`

**Dynamic service URL discovery**
- Uses `gcloud run services describe` to get the current Cloud Run URL at deploy time
- Queries the environment-specific service name (`-staging` suffix for staging)
- Eliminates the hardcoded URL problem in the manual script
- If the Cloud Run service doesn't exist yet, this step will fail with a clear error

### 3. Optional: trigger after Cloud Run deploy

If you want to also auto-trigger this after the training pipeline deploys a new model to Cloud Run, add to the `on:` block:

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: "Target environment"
        type: choice
        options:
          - staging
          - prod
        default: staging
  workflow_run:
    workflows: ["CI/CD Pipeline"]
    branches: [main]
    types: [completed]
```

This would trigger after the CI/CD pipeline completes on main. However, this only makes sense if the Dataflow pipeline code itself changed — the streaming job doesn't need redeployment just because the model changed (the model is served by Cloud Run, not baked into Dataflow).

Note: a `workflow_run` trigger would need to decide which environment to deploy to. You'd likely default to staging for branch pushes and prod for main merges, or skip auto-trigger entirely and keep it manual.

### 4. Secrets required

The workflow reuses the existing `GCP_SERVICE_ACCOUNT_KEY` secret already configured for the CI/CD pipeline. No new secrets needed.

The service account (`kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com`) needs these IAM roles:
- `roles/dataflow.admin` — submit and manage Dataflow jobs
- `roles/pubsub.subscriber` — read from Pub/Sub topic
- `roles/bigquery.dataEditor` — write to BigQuery
- `roles/storage.objectAdmin` — read/write temp and staging buckets
- `roles/run.viewer` — discover Cloud Run service URL

Most of these are likely already granted since the same service account runs the KFP pipelines.

### 5. Deployment flow

**Staging:**
1. Push feature branch → CI builds branch-tagged images
2. Run training pipeline locally: `ENVIRONMENT=staging python iris_pipeline_training.py`
3. Once staging Cloud Run service is up, trigger the Dataflow workflow with `environment=staging`
4. Validate streaming predictions in `iris_predictions_streaming_staging`

**Production:**
1. Merge to main → CI builds `main`-tagged images
2. Run training pipeline locally: `ENVIRONMENT=prod python iris_pipeline_training.py`
3. Once prod Cloud Run service is up, trigger the Dataflow workflow with `environment=prod`
4. Validate streaming predictions in `iris_predictions_streaming`

## Files to create/modify

| File | Action |
|---|---|
| `.github/workflows/deploy-dataflow.yaml` | **New** — the workflow above |

No changes needed to `iris_inference_pipeline.py` or any other existing code. The workflow calls the pipeline script the same way the manual deploy script does.

## Verification

1. Push the workflow file to any branch
2. Go to Actions tab in GitHub → "Deploy Dataflow Streaming" → "Run workflow"
3. Select the target environment (staging first)
4. Check the workflow logs for:
   - Environment variables resolve correctly (service name, table, job prefix)
   - Service URL discovery succeeds for the selected environment
   - Job submission succeeds and prints job name with correct prefix
5. Verify in GCP Console → Dataflow → Jobs that the new streaming job is running with the correct environment prefix
6. Test end-to-end: run the Pub/Sub producer locally and check the environment-specific BigQuery table for predictions
7. Repeat for prod after staging is validated
