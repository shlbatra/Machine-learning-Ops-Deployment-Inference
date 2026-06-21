# Plan: Model Training & Batch Inference via Cloud Composer (Airflow + KubernetesPodOperator)

## Context

The repo currently runs training and batch inference as KFP pipelines on Vertex AI, triggered manually via `python -m` commands. This plan adds Cloud Composer (managed Airflow on GKE) as the orchestration layer, using `KubernetesPodOperator` to run the existing pipeline scripts as containers on Kubernetes. Each DAG has a single KPO task that executes the full pipeline end-to-end — no decomposition into individual steps.

### Why Composer + KubernetesPodOperator

| Concern | Current (KFP on Vertex AI) | Proposed (Composer + KPO) |
|---|---|---|
| Scheduling | Manual trigger or cron via external system | Native Airflow `schedule_interval` |
| Retries & alerting | KFP has limited retry config | Airflow retries, SLA misses, email/Slack alerts |
| Cross-pipeline dependencies | None — training and inference are independent | Both on independent schedules; sensors available if needed later |
| Visibility | Vertex AI console (pipeline-specific) | Airflow UI (all workflows in one place) |
| Cost | Vertex AI Pipelines pricing per run | Composer environment (fixed) + GKE pod costs |
| Portability | Locked to Vertex AI | Runs on any K8s cluster |

### What stays the same

- Docker images built by CI/CD (same `Dockerfile`, same Artifact Registry)
- ML logic in `src/ml_pipelines_kfp/` — **zero code changes** to existing pipeline scripts
- GCS bucket, BigQuery tables, Feature Store, Model Registry
- Staging/prod environment split via `ENVIRONMENT` env var
- Each pipeline compiles KFP YAML and submits a `PipelineJob` to Vertex AI (same as today, just triggered by Airflow instead of manually)

---

## Architecture

```
GitHub Actions (CI/CD)
  ├── Build & push Docker images → Artifact Registry
  └── Sync DAGs → Composer DAGs bucket (gs://COMPOSER_BUCKET/dags/)

Cloud Composer 2 (Airflow on GKE)
  │
  ├── DAG: iris_training
  │   └── Task: run_training_pipeline   (KubernetesPodOperator)
  │         → runs: python -m ml_pipelines_kfp.iris_xgboost.pipelines.iris_pipeline_training
  │         → compiles KFP pipeline → submits PipelineJob to Vertex AI
  │
  └── DAG: iris_batch_inference
      └── Task: run_inference_pipeline  (KubernetesPodOperator)
            → runs: python -m ml_pipelines_kfp.iris_xgboost.pipelines.iris_pipeline_inference
            → compiles KFP pipeline → submits PipelineJob to Vertex AI

Each KPO task:
  - Pulls ml-pipelines-kfp-image from Artifact Registry
  - Mounts service account key as K8s secret (or uses Workload Identity)
  - Receives config via CLI args + env vars (project, region, env, image tags)
  - Compiles and submits the full KFP pipeline to Vertex AI
  - Writes structured logs to stdout → Cloud Logging
```

### How it works

1. Airflow schedule triggers the DAG
2. KPO spins up a pod using the existing `ml-pipelines-kfp-image`
3. The pod runs the pipeline's `__main__` block — same as running it locally today
4. The `__main__` block compiles the KFP pipeline to YAML, then calls `job.submit()` to Vertex AI
5. Vertex AI runs the pipeline (data load → train → evaluate → register → deploy)
6. Pod exits after successful submission; Airflow marks the task as complete

The existing `iris_pipeline_training.py` and `iris_pipeline_inference.py` already accept CLI arguments for all config (project-id, region, image names, BQ tables, etc.), so no code changes are needed — just pass the right args from the DAG.

---

## 1. Composer Environment Setup

### 1a. Create Composer 2 Environment

```bash
gcloud composer environments create ml-pipelines-composer \
  --location us-central1 \
  --environment-size small \
  --image-version composer-2.17.3-airflow-2.10.5 \
  --service-account kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com
```

Key flags:
- `--environment-size small` — sufficient for 2 DAGs; scale up later
- Same service account as existing pipelines (already has BQ, GCS, Vertex AI, Artifact Registry roles)
- Composer 2 runs on GKE Autopilot — no node pool management

### 1b. Grant Composer Service Account Roles

The existing `kfp-mlops@` SA already has most roles. Verify these are present:

```
roles/composer.worker            # Required for Composer
roles/container.developer        # Pull images, create pods on GKE
roles/artifactregistry.reader    # Pull Docker images
roles/bigquery.dataEditor        # Read/write BQ tables
roles/storage.objectAdmin        # Read/write GCS
roles/aiplatform.user            # Model registry, feature store, pipeline submission
roles/run.admin                  # Deploy to Cloud Run (training pipeline)
```

### 1c. Workload Identity (pod authentication)

KPO pods authenticate via GKE Workload Identity — no key files or K8s secrets needed. The pod's Kubernetes service account is mapped to the GCP service account, and `google.auth.default()` picks up credentials automatically from the metadata server.

```bash
gcloud iam service-accounts add-iam-policy-binding \
  kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:deeplearning-sahil.svc.id.goog[composer-user-workloads/default]"
```

Annotate the K8s service account so GKE knows the mapping:

```bash
kubectl annotate serviceaccount default \
  --namespace composer-user-workloads \
  iam.gke.io/gcp-service-account=kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com
```

---

## 2. Docker Image Strategy

**Reuse existing `ml-pipelines-kfp-image`** — it already has all dependencies (scikit-learn, xgboost, kfp, google-cloud-aiplatform, etc.) and the full `src/` package installed. The KPO task invokes the same `python -m` entrypoints that work today.

No new images needed. Same image CI/CD already builds and pushes.

---

## 3. DAG Definitions

### New directory: `dags/`

```
dags/
├── iris_training_dag.py
└── iris_batch_inference_dag.py
```

### 3a. Training DAG

The existing `iris_pipeline_training.py` `__main__` block accepts CLI args for all config. The DAG passes these via `arguments` on the KPO. All config has hardcoded defaults but can be overridden per-run via the Airflow UI "Trigger DAG w/ config" form using `Param`:

```python
# dags/iris_training_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.models.param import Param
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# --- Defaults ---
PROJECT_ID = "deeplearning-sahil"
REGION = "us-central1"
BUCKET = "gs://sb-vertex"
SERVICE_ACCOUNT = "kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"
REGISTRY = f"us-docker.pkg.dev/{PROJECT_ID}/sahil-experiment-docker-images"

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["shlbatra123bot@gmail.com"],
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="iris_training",
    default_args=default_args,
    description="Train Iris classifier, register and deploy best model",
    schedule_interval="0 6 * * 1",  # Weekly Monday 6am UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "training", "iris"],
    params={
        "environment": Param("staging", enum=["staging", "prod"], description="Target environment"),
        "project_id": Param(PROJECT_ID, type="string", description="GCP project ID"),
        "region": Param(REGION, type="string", description="GCP region"),
        "bq_dataset": Param("ml_dataset", type="string"),
        "bq_table": Param("iris", type="string"),
        "bq_feature_table": Param("iris_features", type="string"),
        "service_account": Param(SERVICE_ACCOUNT, type="string"),
    },
) as dag:

    run_training_pipeline = KubernetesPodOperator(
        task_id="run_training_pipeline",
        name="iris-training-pipeline",
        namespace="composer-user-workloads",
        image=(
            f"{REGISTRY}/ml-pipelines-kfp-image:"
            "{{ 'main' if params.environment == 'prod' else 'staging' }}"
        ),
        cmds=["python", "-m", "ml_pipelines_kfp.iris_xgboost.pipelines.iris_pipeline_training"],
        arguments=[
            "--project-id", "{{ params.project_id }}",
            "--region", "{{ params.region }}",
            "--pipeline-name",
            "{{ 'pipeline-iris-prod' if params.environment == 'prod' else 'pipeline-iris-staging' }}",
            "--pipeline-root",
            "{{ params.environment == 'prod' and '%s/prod/pipeline_root' % '%s' or '%s/staging/pipeline_root' % '%s' }}"
                .replace("%s", BUCKET),
            "--model-name",
            "{{ 'Iris-Classifier-XGBoost' if params.environment == 'prod' else 'Iris-Classifier-XGBoost-staging' }}",
            "--image-name",
            f"{REGISTRY}/ml-pipelines-kfp-image:"
            "{{ 'main' if params.environment == 'prod' else 'staging' }}",
            "--fastapi-image-name",
            f"{REGISTRY}/fastapi-ml-generic:"
            "{{ 'main' if params.environment == 'prod' else 'staging' }}",
            "--bq-dataset", "{{ params.bq_dataset }}",
            "--bq-table", "{{ params.bq_table }}",
            "--bq-feature-table", "{{ params.bq_feature_table }}",
            "--service-account", "{{ params.service_account }}",
        ],
        env_vars=[
            k8s.V1EnvVar(name="ENVIRONMENT", value="{{ params.environment }}"),
        ],
        startup_timeout_seconds=300,
        resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "1Gi"},
            limits={"cpu": "1", "memory": "2Gi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )
```

When triggered on schedule, defaults are used. When triggered manually via the UI, Airflow shows a form with all params pre-filled — override any value for that specific run.

### 3b. Batch Inference DAG

```python
# dags/iris_batch_inference_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.models.param import Param
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# --- Defaults ---
PROJECT_ID = "deeplearning-sahil"
REGION = "us-central1"
REGISTRY = f"us-docker.pkg.dev/{PROJECT_ID}/sahil-experiment-docker-images"

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["shlbatra123bot@gmail.com"],
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="iris_batch_inference",
    default_args=default_args,
    description="Batch inference using blessed Iris model",
    schedule_interval="0 8 * * *",  # Daily 8am UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "inference", "iris", "batch"],
    params={
        "environment": Param("staging", enum=["staging", "prod"], description="Target environment"),
        "project_id": Param(PROJECT_ID, type="string", description="GCP project ID"),
        "region": Param(REGION, type="string", description="GCP region"),
        "bq_dataset": Param("ml_dataset", type="string"),
        "bq_feature_table": Param("iris_features", type="string"),
        "bq_table_predictions": Param("iris_predictions_staging", type="string",
                                      description="BQ table for predictions (staging default)"),
    },
) as dag:

    run_inference_pipeline = KubernetesPodOperator(
        task_id="run_inference_pipeline",
        name="iris-inference-pipeline",
        namespace="composer-user-workloads",
        image=(
            f"{REGISTRY}/ml-pipelines-kfp-image:"
            "{{ 'main' if params.environment == 'prod' else 'staging' }}"
        ),
        cmds=["python", "-m", "ml_pipelines_kfp.iris_xgboost.pipelines.iris_pipeline_inference"],
        arguments=[
            "--project-id", "{{ params.project_id }}",
            "--region", "{{ params.region }}",
            "--bq-dataset", "{{ params.bq_dataset }}",
            "--bq-feature-table", "{{ params.bq_feature_table }}",
            "--bq-table-predictions", "{{ params.bq_table_predictions }}",
        ],
        env_vars=[
            k8s.V1EnvVar(name="ENVIRONMENT", value="{{ params.environment }}"),
        ],
        startup_timeout_seconds=300,
        resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "1Gi"},
            limits={"cpu": "1", "memory": "2Gi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )

```

---

## 4. Pipeline Script Change: Use Application Default Credentials

The existing pipeline scripts use `service_account.Credentials.from_service_account_file()` to authenticate. With Workload Identity, credentials come from the GKE metadata server instead. Switch both scripts to use `google.auth.default()`:

```python
# Before (both iris_pipeline_training.py and iris_pipeline_inference.py):
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(
    filename=sa_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
aip.init(project=project_id, location=region, credentials=credentials)

# After:
import google.auth
credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
aip.init(project=project_id, location=region, credentials=credentials)
```

`google.auth.default()` works everywhere — on GKE (Workload Identity), locally (gcloud auth application-default login), and in CI (service account key via `GOOGLE_APPLICATION_CREDENTIALS`). This also removes the need for `--service-account-path` CLI arg.

---

## 5. CI/CD Integration

### 5a. Manual DAG Sync to Composer

For development and testing, sync DAGs manually from your local machine:

```bash
# Get the Composer DAGs bucket
DAGS_BUCKET=$(gcloud composer environments describe ml-pipelines-composer \
    --location us-central1 \
    --project deeplearning-sahil \
    --format="value(config.dagGcsPrefix)")

# Sync all DAGs to Composer
gsutil -m rsync -r dags/ $DAGS_BUCKET/

# Or upload a single DAG
gsutil cp dags/iris_training_dag.py $DAGS_BUCKET/
gsutil cp dags/iris_batch_inference_dag.py $DAGS_BUCKET/
```

After uploading, DAGs appear in the Airflow UI within ~30 seconds (Composer polls the bucket automatically).

### 5b. CI/CD DAG Sync

Add GitHub Actions steps to sync DAGs after Docker image push:

```yaml
# .github/workflows/cicd.yaml — add after existing image push steps

      - name: Get Composer DAGs bucket
        id: composer
        run: |
          DAGS_BUCKET=$(gcloud composer environments describe ml-pipelines-composer \
            --location us-central1 \
            --format="value(config.dagGcsPrefix)")
          echo "dags_bucket=$DAGS_BUCKET" >> $GITHUB_OUTPUT

      - name: Sync DAGs to Composer
        run: |
          gsutil -m rsync -r -d dags/ ${{ steps.composer.outputs.dags_bucket }}/
```

### 5c. Environment Promotion

Image tags are deterministic from `ENV` — the DAG computes the right tag (`main` for prod, `staging` for staging) without any Airflow variable update. CI just needs to push images with those well-known tags.

| Event | Action |
|---|---|
| PR push | Build images with branch tag, sync DAGs |
| Merge to main | Build images with `main` tag, sync DAGs |

---

## 6. Monitoring & Alerting

### Airflow-native

- **Email on failure**: `default_args["email_on_failure"] = True` → sends to `shlbatra123bot@gmail.com`
- **SLA misses**: Add `sla=timedelta(hours=1)` to the KPO task if the pipeline should complete within a time bound
- **Task retries**: 2 retries with 5-minute delay; covers transient Vertex AI submission failures

### Logging

- KPO pods write to stdout → Cloud Logging (automatic on GKE)
- Filter: `resource.type="k8s_container" AND resource.labels.namespace_name="composer-user-workloads"`
- Existing `ml_pipelines_kfp.log` JSON logger works unchanged inside the pod

---

## 7. Resource Configuration

The KPO pod only compiles the pipeline YAML and submits it to Vertex AI — the actual training/inference runs on Vertex AI's infrastructure. The pod is lightweight:

| Task | CPU request/limit | Memory request/limit | Typical duration |
|---|---|---|---|
| run_training_pipeline | 500m / 1 | 1Gi / 2Gi | ~30s (compile + submit) |
| run_inference_pipeline | 500m / 1 | 1Gi / 2Gi | ~30s (compile + submit) |

### GPU for pipeline steps

GPU resources are configured on the Vertex AI pipeline components, not on the KPO pod. See `docs/gpu_vertex_pipelines.md` for the full plan.

---

## 8. Wait-for-Completion

Both pipeline scripts call `job.submit()` followed by `job.wait()` so the KPO pod blocks until the Vertex AI pipeline finishes. This means:

- Airflow task duration matches actual pipeline duration (10-20 min)
- Airflow accurately reflects pipeline success/failure
- Airflow is the single source of truth for pipeline status

```python
# Both iris_pipeline_training.py and iris_pipeline_inference.py:
job.submit(service_account=sa_email)
job.wait()  # blocks until Vertex AI pipeline completes or fails
```

---

## 9. Implementation Order

### Phase 1: Infrastructure (Day 1-2)
1. Create Composer 2 environment (`scripts/setup_composer.sh`)
2. Configure Workload Identity binding for `kfp-mlops@` SA
3. Set Airflow variables via `gcloud composer environments run`
4. Update pipeline scripts to use `google.auth.default()` instead of `from_service_account_file()`

### Phase 2: DAGs (Day 2-3)
4. Write `dags/iris_training_dag.py`
5. Write `dags/iris_batch_inference_dag.py`
6. Upload DAGs manually to Composer for testing: `gsutil cp dags/*.py gs://COMPOSER_BUCKET/dags/`
7. Trigger DAGs from Airflow UI, verify Vertex AI pipelines run successfully

### Phase 3: CI/CD (Day 3-4)
8. Add DAG sync step to `.github/workflows/cicd.yaml`
9. Add Airflow variable update step for image tags
10. Test end-to-end: push branch → images build → DAGs sync → trigger run

### Phase 4: Validation (Day 4-5)
11. Run training DAG in staging, verify model in Vertex AI Registry
12. Run inference DAG in staging, verify predictions in BQ
13. Compare results with manual pipeline runs
14. Enable scheduled runs

---

## 10. File Changes Summary

### New files

| File | Purpose |
|---|---|
| `dags/iris_training_dag.py` | Training DAG — single KPO running `iris_pipeline_training` |
| `dags/iris_batch_inference_dag.py` | Inference DAG — single KPO running `iris_pipeline_inference` |
| `scripts/setup_composer.sh` | One-time Composer environment + secrets + variables setup |

### Modified files

| File | Change |
|---|---|
| `.github/workflows/cicd.yaml` | Add DAG sync + Airflow variable update steps |

### Unchanged (zero code changes)

- `src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py`
- `src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_inference.py`
- All KFP components, constants, models
- All Docker images (reuse `ml-pipelines-kfp-image` as-is)
- Feature Store, Dataflow, FastAPI code
