# Staging + Production Environments Plan

Single GCP project (`deeplearning-sahil`), two logical environments separated by naming conventions, GCS paths, BQ datasets, Cloud Run services, and Vertex AI model aliases.

---

## 1. Environment Naming Convention

Every GCP resource gets an environment suffix/prefix:

| Resource | Staging | Production |
|---|---|---|
| Docker image tag | `<branch-name>` (e.g. `fix-logging`) | `main` |
| KFP pipeline display name | `pipeline-iris-staging` | `pipeline-iris-prod` |
| GCS pipeline root | `gs://sb-vertex/staging/pipeline_root` | `gs://sb-vertex/prod/pipeline_root` |
| GCS deployed models | `gs://sb-vertex/staging/deployed-models/` | `gs://sb-vertex/prod/deployed-models/` |
| BQ dataset | `ml_dataset_staging` | `ml_dataset` (keep current as prod) |
| BQ predictions table | `ml_dataset_staging.iris_predictions` | `ml_dataset.iris_predictions` |
| Cloud Run service | `iris-classifier-xgboost-service-staging` | `iris-classifier-xgboost-service` |
| Vertex AI model name | `Iris-Classifier-XGBoost-staging` | `Iris-Classifier-XGBoost` |
| Vertex AI model alias | `blessed` (within staging model) | `blessed` (within prod model) |
| Pub/Sub topic | `iris-inference-data-staging` | `iris-inference-data` |
| Dataflow job | `iris-streaming-inference-staging-*` | `iris-streaming-inference-*` |

---

## 2. constants.py Changes

Add an `ENV` parameter that drives all resource names. The environment is resolved from: CLI arg > `ENVIRONMENT` env var > default `"staging"`.

```python
# New constants.py structure

import os
from pathlib import Path

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(Path(PACKAGE_ROOT).parent.parent.parent.absolute())
SERVICE_ACCOUNT_PATH = os.path.join(REPO_ROOT, "deeplearning-sahil-e50332de6687.json")

# --- Environment ---
ENV = os.getenv("ENVIRONMENT", "staging")  # "staging" or "prod"

# --- Project settings (shared) ---
PROJECT_ID = "deeplearning-sahil"
REGION = "us-central1"
SERVICE_ACCOUNT = "kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"

# --- Environment-specific settings ---
BUCKET = "gs://sb-vertex"

if ENV == "prod":
    PIPELINE_NAME = "pipeline-iris-prod"
    PIPELINE_ROOT = f"{BUCKET}/prod/pipeline_root"
    MODEL_NAME = "Iris-Classifier-XGBoost"
    BQ_DATASET = "ml_dataset"
    BQ_TABLE = "iris"
    BQ_TABLE_PREDICTIONS = "iris_predictions"
    PUBSUB_TOPIC = "iris-inference-data"
    PUBSUB_SUBSCRIPTION = "iris-inference-data-sub"
    _DEFAULT_IMAGE_TAG = "main"
else:  # staging
    PIPELINE_NAME = "pipeline-iris-staging"
    PIPELINE_ROOT = f"{BUCKET}/staging/pipeline_root"
    MODEL_NAME = "Iris-Classifier-XGBoost-staging"
    BQ_DATASET = "ml_dataset_staging"
    BQ_TABLE = "iris"
    BQ_TABLE_PREDICTIONS = "iris_predictions"
    PUBSUB_TOPIC = "iris-inference-data-staging"
    PUBSUB_SUBSCRIPTION = "iris-inference-data-staging-sub"
    _DEFAULT_IMAGE_TAG = os.getenv("BUILD_BRANCH", "staging")

IMAGE_NAME = os.getenv(
    "PIPELINE_BASE_IMAGE",
    f"us-docker.pkg.dev/{PROJECT_ID}/sahil-experiment-docker-images/ml-pipelines-kfp-image:{_DEFAULT_IMAGE_TAG}",
)
FASTAPI_IMAGE_NAME = os.getenv(
    "PIPELINE_FASTAPI_IMAGE",
    f"us-docker.pkg.dev/{PROJECT_ID}/sahil-experiment-docker-images/fastapi-ml-generic:{_DEFAULT_IMAGE_TAG}",
)

ENDPOINT_NAME = MODEL_NAME
MODEL_FILENAME = "model.joblib"
PUBSUB_REGION = REGION
```

Key behavior:
- Staging defaults to the branch-specific image tag (from `BUILD_BRANCH` env var set by CI)
- Prod always uses the `main` tag
- Safe default: if you forget to set `ENVIRONMENT`, you get staging (can't accidentally pollute prod)

---

## 3. CI/CD Workflow Changes (.github/workflows/cicd.yaml)

### 3a. Keep current behavior: every push builds branch-tagged images

No change needed to the build step. Branch pushes already tag images with the branch name (e.g. `fix-logging`), which is exactly what staging needs.

### 3b. Add staging pipeline trigger (on non-main branches)

Add a new job that compiles and submits the training + inference pipelines in staging mode after the image build succeeds:

```yaml
  deploy-staging:
    needs: build
    if: github.ref_name != 'main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}

      - name: Install dependencies
        run: pip install uv && uv pip install -e . --system

      - name: Submit training pipeline (staging)
        env:
          ENVIRONMENT: staging
          BUILD_BRANCH: ${{ env.IMAGE_TAG }}
        run: |
          python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py \
            --image-name "us-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPOSITORY }}/${{ env.IMAGE_NAME }}:${{ env.IMAGE_TAG }}" \
            --fastapi-image-name "us-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPOSITORY }}/${{ env.FASTAPI_IMAGE_NAME }}:${{ env.IMAGE_TAG }}"
```

### 3c. Add production pipeline trigger (on merge to main)

```yaml
  deploy-prod:
    needs: build
    if: github.ref_name == 'main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}

      - name: Install dependencies
        run: pip install uv && uv pip install -e . --system

      - name: Submit training pipeline (prod)
        env:
          ENVIRONMENT: prod
        run: |
          python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py
```

### 3d. Full trigger summary

| Event | Images built | Pipeline submitted | Environment |
|---|---|---|---|
| Push to feature branch | `<branch>` tag | Training pipeline | staging |
| PR opened/updated | `<branch>` tag | None (build-only validation) | - |
| Merge to main | `main` tag | Training pipeline | prod |

---

## 4. Deploy Component Changes (deploy.py)

The deploy component needs to produce environment-isolated Cloud Run services.

### Service naming

Cloud Run service names already come from the pipeline parameter `service_name`. Update the training pipeline to append the env suffix:

```python
# In iris_pipeline_training.py
env_suffix = "-staging" if ENV != "prod" else ""
service_name = f"{MODEL_NAME.lower().replace('_', '-')}-service{env_suffix}"
```

### GCS model path isolation

The deploy component copies the blessed model to `deployed-models/{service_name}/model.joblib`. Since the service name now includes the env suffix, model artifacts are automatically isolated:
- Staging: `gs://sb-vertex/deployed-models/iris-classifier-xgboost-service-staging/model.joblib`
- Prod: `gs://sb-vertex/deployed-models/iris-classifier-xgboost-service/model.joblib`

No changes needed in `deploy.py` itself.

---

## 5. Vertex AI Model Registry Isolation

Staging and prod models are separate entries in the Model Registry:
- `Iris-Classifier-XGBoost-staging` - staging model, multiple versions, `blessed` alias on current best
- `Iris-Classifier-XGBoost` - prod model, promoted from staging

Both live in the same project. The `blessed` alias works independently on each model entry.

### Promotion flow (staging -> prod)

Promotion is simply: merge the branch to `main`. The `deploy-prod` CI job runs the training pipeline with `ENVIRONMENT=prod`, which:
1. Trains on `ml_dataset.iris` (prod BQ dataset)
2. Registers the model as `Iris-Classifier-XGBoost` (prod model name)
3. Deploys to `iris-classifier-xgboost-service` (prod Cloud Run)

This is a full retrain-on-prod-data approach, not a model copy. Staging validates the code/pipeline; prod retrains on prod data.

---

## 6. BigQuery Dataset Isolation

### One-time setup: create the staging dataset

```bash
bq mk --dataset \
  --location=us-central1 \
  --description="Staging dataset for ML pipelines" \
  deeplearning-sahil:ml_dataset_staging
```

Copy the iris table schema (or a subset of data) into staging:

```bash
bq cp ml_dataset.iris ml_dataset_staging.iris
```

### Table layout

| Dataset | Tables | Purpose |
|---|---|---|
| `ml_dataset` | `iris`, `iris_predictions`, `iris_predictions_streaming` | Production |
| `ml_dataset_staging` | `iris`, `iris_predictions`, `iris_predictions_streaming` | Staging |

---

## 7. Pub/Sub + Dataflow (Streaming Inference)

### Staging topic/subscription setup

```bash
gcloud pubsub topics create iris-inference-data-staging --project=deeplearning-sahil
gcloud pubsub subscriptions create iris-inference-data-staging-sub \
  --topic=iris-inference-data-staging --project=deeplearning-sahil
```

### Dataflow job isolation

Update `scripts/deploy_dataflow_streaming.sh` to accept an environment parameter:

```bash
ENV=${1:-staging}

if [ "$ENV" = "prod" ]; then
  TOPIC="iris-inference-data"
  BQ_TABLE="ml_dataset.iris_predictions_streaming"
  SERVICE_URL="<prod-cloud-run-url>"
  JOB_PREFIX="iris-streaming-inference"
else
  TOPIC="iris-inference-data-staging"
  BQ_TABLE="ml_dataset_staging.iris_predictions_streaming"
  SERVICE_URL="<staging-cloud-run-url>"
  JOB_PREFIX="iris-streaming-inference-staging"
fi
```

---

## 8. Inference Pipeline Changes (iris_pipeline_inference.py)

The inference pipeline already reads from constants. With the `ENV`-driven constants.py, it automatically targets the right BQ dataset and model name. Add `--environment` CLI support matching the training pipeline pattern:

```python
parser.add_argument("--environment", choices=["staging", "prod"])
# ... then set os.environ["ENVIRONMENT"] before importing constants
```

Or simpler: just set `ENVIRONMENT=prod` when invoking:

```bash
# Staging inference
ENVIRONMENT=staging python iris_pipeline_inference.py

# Prod inference
ENVIRONMENT=prod python iris_pipeline_inference.py
```

---

## 9. Implementation Order

### Phase 1: Resource setup (one-time, manual)
1. Create `ml_dataset_staging` BQ dataset + copy iris table
2. Create staging Pub/Sub topic and subscription
3. Verify both staging and prod images exist in Artifact Registry

### Phase 2: Code changes
1. Refactor `constants.py` with `ENV` parameter (as shown in section 2)
2. Update `iris_pipeline_training.py` to use env-suffixed service name
3. Update `iris_pipeline_inference.py` to accept environment
4. Update `deploy_dataflow_streaming.sh` to accept environment

### Phase 3: CI/CD
1. Add `deploy-staging` job to `cicd.yaml` (non-main branches)
2. Add `deploy-prod` job to `cicd.yaml` (main branch only)
3. Optionally: add a PR comment step that links to the staging pipeline run

### Phase 4: Validation
1. Push a feature branch, verify staging pipeline runs with branch-tagged image
2. Verify staging Cloud Run service is created with `-staging` suffix
3. Verify staging BQ predictions land in `ml_dataset_staging`
4. Merge to main, verify prod pipeline runs with `main`-tagged image
5. Verify prod Cloud Run service is unchanged
6. Test streaming inference on both staging and prod topics

---

## 10. Architecture Diagram

```
                    Push to feature branch          Merge to main
                           |                              |
                    [GitHub Actions]                [GitHub Actions]
                           |                              |
                  Build images:fix-logging        Build images:main
                           |                              |
                  Submit KFP pipeline             Submit KFP pipeline
                  ENV=staging                     ENV=prod
                           |                              |
              +------------+----------+       +-----------+-----------+
              |                       |       |                       |
        [Training]              [Inference]  [Training]           [Inference]
              |                       |       |                       |
    Model Registry:             BQ:staging   Model Registry:       BQ:prod
    XGBoost-staging                          XGBoost
              |                                    |
    Cloud Run:                              Cloud Run:
    ...-service-staging                     ...-service
              |                                    |
    Dataflow:staging                        Dataflow:prod
    PubSub:staging                          PubSub:prod
```

---

## 11. What Stays Shared (Single GCP Project)

These resources are intentionally shared across both environments:
- **GCP project**: `deeplearning-sahil`
- **Service account**: `kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com`
- **Artifact Registry repo**: `sahil-experiment-docker-images` (images for both envs live here, separated by tag)
- **GCS bucket**: `gs://sb-vertex` (paths isolated by `staging/` vs `prod/` prefix)
- **Region**: `us-central1`

---

## 12. Cost Considerations

Since both environments share a single project:
- Cloud Run staging services can scale to 0 (already configured with `min_instance_count: 0`)
- Staging Dataflow jobs can be manually started/stopped (not always-on)
- Staging BQ storage is minimal (same schema, can use smaller data subsets)
- Vertex AI Pipeline runs are the main cost driver - staging pipelines run on every branch push, so consider adding a path filter to only trigger when `src/` files change
