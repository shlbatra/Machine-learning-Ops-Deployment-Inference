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
| BQ dataset | `ml_dataset` | `ml_dataset` |
| BQ training table | `ml_dataset.iris` | `ml_dataset.iris` |
| BQ predictions table | `ml_dataset.iris_predictions_staging` | `ml_dataset.iris_predictions` |
| BQ streaming predictions table | `ml_dataset.iris_predictions_streaming_staging` | `ml_dataset.iris_predictions_streaming` |
| Cloud Run service | `iris-classifier-xgboost-service-staging` | `iris-classifier-xgboost-service` |
| Vertex AI model name | `Iris-Classifier-XGBoost-staging` | `Iris-Classifier-XGBoost` |
| Vertex AI model alias | `blessed` (within staging model) | `blessed` (within prod model) |
| Pub/Sub topic | `iris-inference-data` | `iris-inference-data` |
| Pub/Sub subscription | `iris-inference-data-sub` | `iris-inference-data-sub` |
| Dataflow job | `iris-streaming-inference-staging-*` | `iris-streaming-inference-*` |

---

## 2. Constants Changes

Split constants into two files: a root-level file for shared GCP settings reusable across ML projects, and the existing project-specific file for iris pipeline settings.

### 2a. New root-level constants: `src/ml_pipelines_kfp/constants.py`

```python
import os
from pathlib import Path

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(Path(PACKAGE_ROOT).parent.parent.absolute())
SERVICE_ACCOUNT_PATH = os.path.join(REPO_ROOT, "deeplearning-sahil-e50332de6687.json")

PROJECT_ID = "deeplearning-sahil"
REGION = "us-central1"
BUCKET = "gs://sb-vertex"
SERVICE_ACCOUNT = "kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"

ENV = os.getenv("ENVIRONMENT", "staging")  # "staging" or "prod"
```

### 2b. Updated project constants: `src/ml_pipelines_kfp/iris_xgboost/constants.py`

Import shared settings from root. Environment-specific resource names are driven by `ENV`.

```python
import os

from ml_pipelines_kfp.constants import (  # noqa: F401
    BUCKET, ENV, PROJECT_ID, REGION, REPO_ROOT, SERVICE_ACCOUNT, SERVICE_ACCOUNT_PATH,
)

# --- Shared across environments ---
BQ_DATASET = "ml_dataset"
BQ_TABLE = "iris"
PUBSUB_TOPIC = "iris-inference-data"
PUBSUB_SUBSCRIPTION = "iris-inference-data-sub"

# --- Environment-specific ---
if ENV == "prod":
    PIPELINE_NAME = "pipeline-iris-prod"
    PIPELINE_ROOT = f"{BUCKET}/prod/pipeline_root"
    MODEL_NAME = "Iris-Classifier-XGBoost"
    BQ_TABLE_PREDICTIONS = "iris_predictions"
    BQ_TABLE_PREDICTIONS_STREAMING = "iris_predictions_streaming"
    _DEFAULT_IMAGE_TAG = "main"
else:  # staging
    PIPELINE_NAME = "pipeline-iris-staging"
    PIPELINE_ROOT = f"{BUCKET}/staging/pipeline_root"
    MODEL_NAME = "Iris-Classifier-XGBoost-staging"
    BQ_TABLE_PREDICTIONS = "iris_predictions_staging"
    BQ_TABLE_PREDICTIONS_STREAMING = "iris_predictions_streaming_staging"
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

The `# noqa: F401` re-export means all existing `from ml_pipelines_kfp.iris_xgboost.constants import PROJECT_ID, ...` imports continue to work with no downstream changes.

Key behavior:
- Staging defaults to the branch-specific image tag (from `BUILD_BRANCH` env var set by CI)
- Prod always uses the `main` tag
- Safe default: if you forget to set `ENVIRONMENT`, you get staging (can't accidentally pollute prod)
- Root constants (`ml_pipelines_kfp.constants`) can be imported directly by future ML projects without depending on `iris_xgboost`

---

## 3. CI/CD Workflow Changes (.github/workflows/cicd.yaml)

### 3a. Keep current behavior: every push builds branch-tagged images

No change needed to the build step. Branch pushes already tag images with the branch name (e.g. `fix-logging`), which is exactly what staging needs.

### 3b. Staging pipeline runs (local)

No CI job needed for staging. Once images are built by CI, run the staging pipeline locally:

```bash
ENVIRONMENT=staging \
PIPELINE_BASE_IMAGE=us-docker.pkg.dev/deeplearning-sahil/sahil-experiment-docker-images/ml-pipelines-kfp-image:<branch> \
PIPELINE_FASTAPI_IMAGE=us-docker.pkg.dev/deeplearning-sahil/sahil-experiment-docker-images/fastapi-ml-generic:<branch> \
  python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py
```

### 3c. Production pipeline runs (local)

No CI job needed for production either. After merging to main and CI builds the `main`-tagged images, run the production pipeline locally:

```bash
ENVIRONMENT=prod \
  python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py
```

### 3d. Full trigger summary

| Event | Images built | Pipeline submitted | Environment |
|---|---|---|---|
| Push to feature branch | `<branch>` tag | None (run locally) | staging |
| Merge to main | `main` tag | None (run locally) | prod |

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

Promotion is simply: merge the branch to `main`, wait for CI to build the `main`-tagged images, then run the pipeline locally:

```bash
ENVIRONMENT=prod python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py
```

This will:
1. Train on `ml_dataset.iris` (shared BQ dataset)
2. Register the model as `Iris-Classifier-XGBoost` (prod model name)
3. Deploy to `iris-classifier-xgboost-service` (prod Cloud Run)

This is a full retrain-on-prod-data approach, not a model copy. Staging validates the code/pipeline; prod retrains on prod data.

---

## 6. BigQuery Dataset

Staging and production share the same BQ dataset (`ml_dataset`) and training table (`iris`). Predictions tables are environment-specific to avoid staging writes polluting production data.

| Dataset | Table | Used by |
|---|---|---|
| `ml_dataset` | `iris` | Both (shared training data) |
| `ml_dataset` | `iris_predictions` | Production |
| `ml_dataset` | `iris_predictions_staging` | Staging |
| `ml_dataset` | `iris_predictions_streaming` | Production |
| `ml_dataset` | `iris_predictions_streaming_staging` | Staging |

Staging predictions tables are auto-created on first write — no one-time setup needed.

---

## 7. Pub/Sub + Dataflow (Streaming Inference)

Staging and production share the same Pub/Sub topic (`iris-inference-data`) and subscription (`iris-inference-data-sub`). No separate topic setup needed.

### Dataflow job isolation

Environment isolation for streaming is at the Dataflow job and Cloud Run service layer. Update `scripts/deploy_dataflow_streaming.sh` to accept an environment parameter:

```bash
ENV=${1:-staging}

TOPIC="iris-inference-data"

if [ "$ENV" = "prod" ]; then
  SERVICE_URL="<prod-cloud-run-url>"
  JOB_PREFIX="iris-streaming-inference"
  BQ_TABLE="ml_dataset.iris_predictions_streaming"
else
  SERVICE_URL="<staging-cloud-run-url>"
  JOB_PREFIX="iris-streaming-inference-staging"
  BQ_TABLE="ml_dataset.iris_predictions_streaming_staging"
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
1. Verify both staging and prod images exist in Artifact Registry
2. BQ dataset and Pub/Sub topic/subscription are shared — no setup needed

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
3. Verify staging BQ predictions land in `ml_dataset.iris_predictions_staging`
4. Merge to main, verify prod pipeline runs with `main`-tagged image
5. Verify prod Cloud Run service is unchanged
6. Test streaming inference on shared Pub/Sub topic with both staging and prod Dataflow jobs

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
    Model Registry:                          Model Registry:
    XGBoost-staging                          XGBoost
              |                                    |
    Cloud Run:                              Cloud Run:
    ...-service-staging                     ...-service
              |                                    |
    Dataflow:staging                        Dataflow:prod
              \                                   /
               \                                 /
                +--- Shared: BQ:ml_dataset -----+
                +--- Shared: PubSub:iris-inference-data ---+
```

---

## 11. What Stays Shared (Single GCP Project)

These resources are intentionally shared across both environments:
- **GCP project**: `deeplearning-sahil`
- **Service account**: `kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com`
- **Artifact Registry repo**: `sahil-experiment-docker-images` (images for both envs live here, separated by tag)
- **GCS bucket**: `gs://sb-vertex` (paths isolated by `staging/` vs `prod/` prefix)
- **BQ dataset**: `ml_dataset` (shared dataset; training table shared, predictions tables are env-specific)
- **Pub/Sub topic**: `iris-inference-data` and subscription `iris-inference-data-sub`
- **Region**: `us-central1`

---

## 12. Cost Considerations

Since both environments share a single project:
- Cloud Run staging services can scale to 0 (already configured with `min_instance_count: 0`)
- Staging Dataflow jobs can be manually started/stopped (not always-on)
- BQ dataset is shared — only predictions tables are duplicated (minimal storage overhead)
- Pub/Sub topic is shared — no duplicate infrastructure
- Vertex AI Pipeline runs are the main cost driver - staging pipelines run on every branch push, so consider adding a path filter to only trigger when `src/` files change
