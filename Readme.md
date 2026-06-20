# End to end ML Pipeline with training, deployment and real time inference.

![CI/CD Pipeline](https://github.com/shlbatra/Machine-learning-Ops-Deployment-Inference/actions/workflows/cicd.yaml/badge.svg)

A production-grade ML pipeline implementation using Kubeflow Pipelines (KFP) on Google Cloud Vertex AI. This project demonstrates MLOps best practices for automating end-to-end ML workflows.

## Overview

This repository implements a complete ML pipeline for the Iris dataset classification problem, showcasing:
- Automated data ingestion from BigQuery
- Parallel model training (Decision Tree, Random Forest, XGBoost)
- Automatic model evaluation and selection
- Model registration and versioning in Vertex AI
- Automated deployment to FastAPI services on Cloud Run
- Batch inference capabilities
- Feature Store with offline (BQ) and online (Bigtable) serving
- Streaming feature ingestion via Dataflow (Pub/Sub → dual-write BQ + Bigtable)
- Real-time streaming inference via Dataflow (online store lookup → FastAPI → BQ)
- REST API serving with FastAPI

## Key Features

- **Component-based Architecture**: Modular, reusable KFP pipeline components
- **Multi-model Training**: Trains multiple models in parallel and selects the best performer
- **Feature Store**: Vertex AI Feature Store V2 with offline (BQ) and online (Bigtable) serving
- **Dual Streaming Pipelines**: Independent feature ingestion and real-time inference via Dataflow
- **Cloud-native**: Deep integration with Google Cloud (Vertex AI, BigQuery, GCS, Cloud Run, Dataflow, Pub/Sub)
- **Production-ready**: Model versioning, blessed-model deployment, Pydantic validation, structured logging
- **Containerized**: Three Docker images (KFP components, FastAPI serving, Beam SDK workers)

## Project Structure

```
src/
├── ml_pipelines_kfp/               # ML pipeline components and serving
│   ├── constants.py                # Shared GCP settings (project, region, bucket, env)
│   ├── log.py                      # Shared JSON logging helper
│   ├── schemas/                    # Input/output schemas for Vertex AI
│   │   └── iris_xgboost/vertex/    # instance.yaml, prediction.yaml
│   ├── notebooks/                  # Jupyter notebooks for exploration/verification
│   │   ├── verify_model_features.ipynb
│   │   ├── test_notebook_feature_platform.ipynb
│   │   └── test_inference.ipynb
│   └── iris_xgboost/               # Iris classification implementation
│       ├── pipelines/              # KFP pipeline definitions
│       │   ├── components/         # Reusable pipeline components
│       │   │   └── fastapi/        # FastAPI server component
│       │   ├── iris_pipeline_training.py
│       │   └── iris_pipeline_inference.py
│       ├── models/                 # Pydantic models for API (Instance, Prediction)
│       └── constants.py            # Iris-specific constants (model name, BQ tables)
├── dataflow/                       # Dataflow streaming pipelines
│   ├── iris_feature_pipeline.py    # Pub/Sub → Feature Store (dual-write BQ + Bigtable)
│   ├── iris_inference_pipeline.py  # Pub/Sub → online store lookup → FastAPI → BQ
│   ├── models/                     # Pydantic schemas for Pub/Sub messages
│   │   └── iris_schema.py
│   └── utils/                      # Reusable Beam DoFns
│       ├── online_store_reader.py  # Sync fetch from Feature Store online store
│       └── online_store_writer.py  # Direct write to online store via v1beta1 API
├── feature_store/                  # Feature Store definitions and scripts
│   ├── schema.py                   # Shared FeatureConfig dataclass
│   ├── ingest.py                   # Raw BQ → canonical feature table
│   ├── setup.py                    # One-time online store + feature view creation
│   ├── sync.py                     # Trigger FeatureView sync (offline → online)
│   └── iris/                       # Iris-specific feature definitions
│       └── feature_definitions.py
scripts/
├── load_data.sh                    # Load Iris data to BigQuery
├── setup_feature_store.sh          # One-time Feature Store setup
├── deploy_dataflow_feature.sh      # Deploy feature ingestion Dataflow job
├── deploy_dataflow_streaming.sh    # Deploy inference Dataflow job
├── run_pubsub_producer.sh          # Publish test events to Pub/Sub
├── setup_pubsub.sh                 # Create Pub/Sub topic/subscription
├── setup_artifact_registry.sh      # Create Artifact Registry repo
└── clean_reinstall.sh              # Clean venv and reinstall
test/                               # Unit/integration tests
Dockerfile                          # KFP component container
Dockerfile.fastapi                  # FastAPI serving container
Dockerfile.beam                     # Beam SDK container for Dataflow workers
pyproject.toml                      # Project dependencies (hatchling build)
```

## Prerequisites

- Python 3.9-3.10
- Google Cloud Project with enabled APIs:
  - Vertex AI
  - BigQuery
  - Cloud Storage
- Service account with appropriate permissions
- `uv` package manager (for dependency management)


## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml_pipelines_kfp

# Install dependencies
uv pip install -e .
```

## Environments

The project supports two environments controlled by the `ENVIRONMENT` env var:

| | Staging (default) | Production |
|---|---|---|
| `ENVIRONMENT` | `staging` | `prod` |
| Pipeline name | `pipeline-iris-staging` | `pipeline-iris-prod` |
| Model name | `Iris-Classifier-XGBoost-staging` | `Iris-Classifier-XGBoost` |
| Image tag | `<branch>` | `main` |
| Cloud Run service | `iris-classifier-xgboost-service-staging` | `iris-classifier-xgboost-service` |
| BQ predictions table | `iris_predictions_staging` | `iris_predictions` |
| GCS pipeline root | `gs://sb-vertex/staging/pipeline_root` | `gs://sb-vertex/prod/pipeline_root` |

**Shared across environments:** BQ dataset (`ml_dataset`), training table (`iris`), Pub/Sub topic (`iris-inference-data`).

Safe default: if `ENVIRONMENT` is not set, staging is used — you can't accidentally pollute prod.

## Usage

### 1. Load Data to BigQuery

```bash
# Load the original 150 labeled iris rows (WRITE_TRUNCATE)
./scripts/load_data.sh

# Append N random unlabeled rows for batch inference scoring (WRITE_APPEND)
./scripts/load_data.sh --generate-random 20
```

The base load writes 150 labeled training rows to `ml_dataset.iris`. The `--generate-random` flag writes N unlabeled rows to a separate `ml_dataset.iris_batch_input` table, simulating new data arriving for batch inference scoring. Both tables include `Id` and `load_timestamp` columns for downstream ingestion.

### 2. Feature Store Setup

Set up the Feature Store online store and feature view. Run once per ML project before training.

```bash
# Default (iris project)
./scripts/setup_feature_store.sh

# Specify a different ML project config
./scripts/setup_feature_store.sh --config fraud

# Override GCP project and region
./scripts/setup_feature_store.sh --config iris --project my-project --region us-east1

# See all options
./scripts/setup_feature_store.sh --help
```

### 3. Run Training Pipeline

#### Staging

Run after CI builds branch-tagged images:

```bash
ENVIRONMENT=staging \
PIPELINE_BASE_IMAGE=us-docker.pkg.dev/deeplearning-sahil/sahil-experiment-docker-images/ml-pipelines-kfp-image:<branch> \
PIPELINE_FASTAPI_IMAGE=us-docker.pkg.dev/deeplearning-sahil/sahil-experiment-docker-images/fastapi-ml-generic:<branch> \
  python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py
```

This creates pipeline `pipeline-iris-staging`, registers model `Iris-Classifier-XGBoost-staging`, and deploys to Cloud Run service `iris-classifier-xgboost-service-staging`.

#### Production

Run after merging to main and CI builds `main`-tagged images:

```bash
ENVIRONMENT=prod python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py
```

This creates pipeline `pipeline-iris-prod`, registers model `Iris-Classifier-XGBoost`, and deploys to Cloud Run service `iris-classifier-xgboost-service`.

#### CLI overrides

```bash
python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py \
  --project-id my-other-project \
  --region us-east1 \
  --model-name Iris-Classifier-Test \
  --pipeline-name pipeline-iris-test

# See all available options
python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py --help
```

**Image configuration:** `PIPELINE_BASE_IMAGE` and `PIPELINE_FASTAPI_IMAGE` env vars control which Docker images are baked into the compiled pipeline. KFP resolves `base_image` at compile time, so these must be set before running the script. Staging defaults to the branch tag; production defaults to `main`.

### 4. Run Batch Inference Pipeline

#### Staging

```bash
ENVIRONMENT=staging \
PIPELINE_BASE_IMAGE=us-docker.pkg.dev/deeplearning-sahil/sahil-experiment-docker-images/ml-pipelines-kfp-image:<branch> \
  python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_inference.py
```

Predictions are written to `ml_dataset.iris_predictions_staging`.

#### Production

```bash
ENVIRONMENT=prod python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_inference.py
```

Predictions are written to `ml_dataset.iris_predictions`.

### 5. Streaming Feature Ingestion

Deploy a Dataflow streaming job that ingests Pub/Sub messages into the Feature Store (dual-writes to BQ offline store and Bigtable online store):

```bash
# Staging
./scripts/deploy_dataflow_feature.sh staging

# Production
./scripts/deploy_dataflow_feature.sh prod
```

### 6. Real-time Streaming Inference

Deploy a Dataflow streaming job for real-time inference:

```bash
# Staging — writes to ml_dataset.iris_predictions_streaming_staging
./scripts/deploy_dataflow_streaming.sh staging

# Production — writes to ml_dataset.iris_predictions_streaming
./scripts/deploy_dataflow_streaming.sh prod
```

### 7. Publish Pub/Sub Test Events

Generate random Iris data and publish to Pub/Sub for testing streaming pipelines:

```bash
# Default: batch_size=10, delay=5s, runs indefinitely
./scripts/run_pubsub_producer.sh

# Custom: batch_size=20, delay=2s, duration=60s
./scripts/run_pubsub_producer.sh 20 2 60
```

This can be run from any directory — the script resolves paths automatically.

## Development

### Code Quality

```bash
# Format code
black src/

# Lint
ruff check src/

# Type checking
mypy src/
```


## Architecture

```
                    Push / PR                         Merge to main
                        |                                   |
                 [GitHub Actions CI]                 [GitHub Actions CI]
                        |                                   |
              Build 3 images:<branch>             Build 3 images:main
              (KFP, FastAPI, Beam)                 (KFP, FastAPI, Beam)
                        |                                   |
                 Run KFP pipeline                   Run KFP pipeline
                 ENV=staging                         ENV=prod
                        |                                   |
           +------------+------------+       +--------------+--------------+
           |                         |       |                             |
     [Training]               [Batch Inf]  [Training]               [Batch Inf]
           |                         |       |                             |
   Model Registry:       Get Model: blessed  Model Registry:    Get Model: blessed
   XGBoost-staging      + BQ Feature Store   XGBoost           + BQ Feature Store
                                     |                                     |
                             BQ:iris_predictions                   BQ:iris_predictions
                               _staging
           |                                       |
   Cloud Run:                                Cloud Run:
   ...-service-staging                       ...-service
           |                                       |
           +-------------- Shared ----------------+
                               |
                      PubSub:iris-inference-data
                          /              \
          [Dataflow: Feature Pipeline]  [Dataflow: Inference Pipeline]
                     |                           |
            dual-write to:              online store lookup →
            BQ (offline) +               FastAPI → BQ predictions
            Bigtable (online)
                     |                           ^
                     |     features served via    |
                     +----------------------------+
                     |
              Feature Store
              (offline: BQ  |  online: Bigtable)
```

The project follows a component-based architecture where each ML pipeline step is a self-contained KFP component:

1. **Data Component**: Loads and splits data from BigQuery
2. **Model Components**: Implements various ML algorithms (Decision Tree, Random Forest, XGBoost)
3. **Evaluation Component**: Compares model performance
4. **Registry Component**: Manages model versioning with "blessed" aliases
5. **Deployment Component**: Deploys blessed models to Cloud Run FastAPI services
6. **Batch Inference Component**: Scores unlabeled data using the blessed model
7. **Feature Pipeline** (Dataflow): Pub/Sub → dual-write to BQ offline + Bigtable online store
8. **Inference Pipeline** (Dataflow): Pub/Sub → online store feature lookup → FastAPI → BQ predictions

## Configuration

Configuration is split across two files:

- **`src/ml_pipelines_kfp/constants.py`** — shared GCP settings (project ID, region, bucket, service account, `ENV`)
- **`src/ml_pipelines_kfp/iris_xgboost/constants.py`** — iris-specific settings (model name, BQ tables, image names, env-specific branching)

Set `ENVIRONMENT=staging` or `ENVIRONMENT=prod` to switch all resource names. Defaults to `staging`.

## CI/CD

The repository includes three GitHub Actions workflows:

**`cicd.yaml`** — Triggers on every push to `main` and on all PRs:
- Builds **three** Docker images: KFP component, FastAPI inference, and Beam SDK (Dataflow workers)
- Tags images with the branch name (e.g. `fix-logging` for feature branches, `main` for production)
- Pushes to Google Artifact Registry

**`deploy-dataflow.yaml`** — Manual dispatch (`workflow_dispatch`) for deploying the streaming inference Dataflow job:
- Configurable environment (staging/prod), region, machine type, batch size, and concurrency
- Uses the Beam SDK container image built by CI

**`deploy-dataflow-feature.yaml`** — Manual dispatch for deploying the feature ingestion Dataflow job:
- Same configurability as the inference pipeline
- Deploys the dual-write feature pipeline (Pub/Sub → BQ + Bigtable)

KFP training/inference pipelines are submitted locally after CI builds the images — no automated pipeline deployment in CI.

## Technologies

- **Orchestration**: Kubeflow Pipelines 2.8.0
- **Cloud Platform**: Google Cloud (Vertex AI, BigQuery, GCS, Cloud Run, Dataflow, Pub/Sub)
- **Feature Store**: Vertex AI Feature Store V2 (BigQuery offline + Bigtable online)
- **ML Frameworks**: scikit-learn, XGBoost
- **API Framework**: FastAPI
- **Streaming**: Apache Beam 2.50+, Dataflow (Runner V2, Streaming Engine)
- **Data Validation**: Pydantic
- **Data Processing**: Pandas, Polars, Dask
- **Async HTTP**: aiohttp (micro-batch inference)
- **Package Management**: uv, Hatchling
- **CI/CD**: GitHub Actions (3 workflows: CI build, Dataflow inference deploy, Dataflow feature deploy)

## Deployment Architecture

### Model Deployment Strategy

The project uses a **blessed model pattern** for production deployments:

1. **Training Pipeline**: Trains multiple models and selects the best performer
2. **Model Registry**: Stores the winning model in Vertex AI with "blessed" alias
3. **Deployment Pipeline**: Automatically deploys only "blessed" models to production
4. **Cost Optimization**: Uses FastAPI on Cloud Run

### Streaming Architecture

Two independent Dataflow streaming pipelines share the same Pub/Sub topic:

**Feature Pipeline** (`iris_feature_pipeline.py`):
1. **Pub/Sub** → parse and validate with Pydantic
2. **Rename** raw fields to canonical feature names
3. **Dual-write**: BQ `iris_features` table (offline store) + Bigtable (online store via v1beta1 `feature_view_direct_write`)

**Inference Pipeline** (`iris_inference_pipeline.py`):
1. **Pub/Sub** → extract `entity_id`
2. **Online store lookup**: sync gRPC fetch from Bigtable, sequential per batch with retry and exponential backoff
3. **Micro-batch**: Beam `BatchElements` groups up to 50 messages per `/predict` call (flush after 1s at low traffic)
4. **FastAPI call**: async HTTP (`aiohttp`) with retry and exponential backoff
5. **BigQuery**: predictions written with `entity_id`, features (JSON), class probabilities, and timestamps; failed rows raise an exception

Both pipelines use the **Beam SDK container image** (`Dockerfile.beam`) with all project packages pre-installed, deployed via `--sdk_container_image` and Runner V2.

### Feature Store Architecture

The project uses **Vertex AI Feature Store V2** (BigQuery-backed) to provide a single source of truth for feature schemas and consistent feature serving across all paths:

```
                    Raw BQ Tables
                         |
                    [ingest.py]
                         |
              iris_features (canonical BQ table)
                    /              \
           Offline Store         [sync.py]
          (BQ — bulk reads)         |
           /          \        Online Store
     Training    Batch Inference  (Bigtable — key lookups)
     (point-in-time)  (latest)        |
                                 Real-time Inference
                                 (ms latency)
```

| Path | Store | Query Pattern | Latency |
|------|-------|--------------|---------|
| Training | Offline (BQ) | Point-in-time join on `feature_timestamp` | Seconds |
| Batch inference | Offline (BQ) | Latest per entity | Seconds |
| Real-time inference | Online (Bigtable) | Key lookup by `entity_id` | Milliseconds |

**Two streaming pipelines** run independently:
- **Feature pipeline** (`iris_feature_pipeline.py`): Pub/Sub → dual-write to BQ (offline) + Bigtable (online)
- **Inference pipeline** (`iris_inference_pipeline.py`): Pub/Sub → online store lookup → FastAPI → BQ predictions

Feature definitions live in `src/feature_store/` with a shared `FeatureConfig` contract. Each ML project gets its own sub-package (e.g. `iris/feature_definitions.py`) with canonical column names, source-to-canonical mappings, and resource IDs.

### Key Benefits

- **Cost Effective**: Cloud Run FastAPI services cost ~90% less than Vertex AI endpoints
- **Scalable**: Dataflow auto-scales based on Pub/Sub message volume with Streaming Engine
- **Reliable**: Blessed-model deployments, retry with backoff on online store reads, Pydantic message validation
- **Consistent Features**: Single Feature Store serves training (offline/BQ), batch (offline/BQ), and real-time (online/Bigtable)
- **Observable**: All predictions logged to BigQuery with metadata; structured JSON logging via Cloud Logging

## Logging

All components use structured JSON logging via `ml_pipelines_kfp.log.get_logger()`. Logs are auto-parsed by Cloud Logging, enabling filtering by severity, module, and message content.

### Searching Logs in Cloud Logging

**Filter by severity:**
```
severity="ERROR"
severity>="WARNING"
```

**Search by message content:**
```
jsonPayload.message=~"loading data"
jsonPayload.message=~"ml_dataset"
```

**Filter by module:**
```
jsonPayload.module="ephemeral_component"
```

**Filter by pipeline job labels:**
```
labels.ml_pipelines_run_id="your-run-id"
labels.ml_pipelines_component_name="load-data"
```

**Combined example — find errors in a specific pipeline run:**
```
labels.ml_pipelines_run_id="your-run-id"
severity="ERROR"
jsonPayload.message=~"deploy"
```
