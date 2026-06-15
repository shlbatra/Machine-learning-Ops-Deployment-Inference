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
- Real-time streaming inference with Dataflow
- REST API serving with FastAPI

## Key Features

- **Component-based Architecture**: Modular, reusable pipeline components
- **Multi-model Training**: Trains multiple models in parallel and selects the best performer
- **Cloud-native**: Deep integration with Google Cloud services (Vertex AI, BigQuery, GCS)
- **Production-ready**: Includes model versioning, schema validation, and deployment automation
- **Containerized**: Each component runs in Docker containers with isolated dependencies

## Project Structure

```
src/ml_pipelines_kfp/
├── constants.py            # Shared GCP settings (project, region, bucket, env)
├── log.py                  # Shared JSON logging helper
├── iris_xgboost/           # Main Iris classification implementation
│   ├── pipelines/          # KFP pipeline definitions
│   │   ├── components/     # Reusable pipeline components
│   │   ├── iris_pipeline_training.py
│   │   └── iris_pipeline_inference.py
│   ├── models/             # Pydantic models for API
│   ├── bq_dataloader.py    # BigQuery data loading utility
│   └── constants.py        # Iris-specific constants (model name, BQ tables, env branching)
├── dataflow/               # Dataflow streaming pipelines
│   └── iris_streaming_pipeline.py
└── notebooks/              # Example notebooks and experiments
schemas/                    # Input/output schemas for Vertex AI
Dockerfile                  # Container definition
pyproject.toml              # Project dependencies
pipeline.yaml               # Pipeline configuration
deploy_dataflow_streaming.sh # Dataflow streaming deployment script
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

The base load writes 150 labeled rows with an `Id` (1–150) and `load_timestamp` to `ml_dataset.iris`. The `--generate-random` flag appends N unlabeled rows (no `Species`) with auto-incrementing Ids, simulating new data arriving for scoring. Each load is timestamped so downstream ingestion can preserve the T1/T2 distinction.

### 2. Run Training Pipeline

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

### 3. Run Batch Inference Pipeline

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

### 4. Real-time Streaming Inference

Deploy a Dataflow streaming job for real-time inference:

```bash
# Staging — writes to ml_dataset.iris_predictions_streaming_staging
./scripts/deploy_dataflow_streaming.sh staging

# Production — writes to ml_dataset.iris_predictions_streaming
./scripts/deploy_dataflow_streaming.sh prod
```

Start generating test data:

```bash
python src/ml_pipelines_kfp/iris_xgboost/pubsub_producer.py --project-id=deeplearning-sahil
```

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
                    Push to feature branch          Merge to main
                           |                              |
                    [GitHub Actions]                [GitHub Actions]
                           |                              |
                  Build images:<branch>           Build images:main
                           |                              |
                  Run KFP pipeline              Run KFP pipeline
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

The project follows a component-based architecture where each ML pipeline step is a self-contained KFP component:

1. **Data Component**: Loads and splits data from BigQuery
2. **Model Components**: Implements various ML algorithms
3. **Evaluation Component**: Compares model performance
4. **Registry Component**: Manages model versioning with "blessed" aliases
5. **Deployment Component**: Deploys blessed models to Cloud Run FastAPI services
6. **Inference Component**: Performs batch predictions
7. **Streaming Component**: Real-time inference via Dataflow and Pub/Sub

## Configuration

Configuration is split across two files:

- **`src/ml_pipelines_kfp/constants.py`** — shared GCP settings (project ID, region, bucket, service account, `ENV`)
- **`src/ml_pipelines_kfp/iris_xgboost/constants.py`** — iris-specific settings (model name, BQ tables, image names, env-specific branching)

Set `ENVIRONMENT=staging` or `ENVIRONMENT=prod` to switch all resource names. Defaults to `staging`.

## CI/CD

The repository includes GitHub Actions workflow (`.github/workflows/cicd.yaml`) that:
- Builds Docker images for KFP components and FastAPI inference containers
- Tags images with the branch name (e.g. `fix-logging` for feature branches, `main` for production)
- Pushes to Google Artifact Registry
- Triggers on every push to any branch

Pipelines are submitted locally after CI builds the images — no automated pipeline deployment in CI.

## Technologies

- **Orchestration**: Kubeflow Pipelines 2.8.0
- **Cloud Platform**: Google Cloud (Vertex AI, BigQuery, GCS, Cloud Run, Dataflow)
- **ML Frameworks**: scikit-learn, XGBoost
- **API Framework**: FastAPI
- **Streaming**: Apache Beam, Dataflow, Pub/Sub
- **Data Processing**: Pandas, Polars, Dask
- **Package Management**: uv, Hatchling

## Deployment Architecture

### Model Deployment Strategy

The project uses a **blessed model pattern** for production deployments:

1. **Training Pipeline**: Trains multiple models and selects the best performer
2. **Model Registry**: Stores the winning model in Vertex AI with "blessed" alias
3. **Deployment Pipeline**: Automatically deploys only "blessed" models to production
4. **Cost Optimization**: Uses FastAPI on Cloud Run

### Streaming Architecture

Real-time inference is handled through:

1. **Data Ingestion**: Pub/Sub receives real-time inference requests
2. **Stream Processing**: Dataflow processes messages with micro-batching and calls FastAPI services
3. **Model Serving**: Cloud Run hosts FastAPI containers with blessed models
4. **Results Storage**: Predictions are written to BigQuery for monitoring

Streaming supports **micro-batching** via Beam's `BatchElements` with `max_batch_duration_secs`. Up to 50 messages are grouped into a single `/predict` call, reducing HTTP overhead by ~10-50x. At low traffic, partial batches flush after 1 second so no message waits indefinitely. Both `--batch_size` and `--max_batch_duration_secs` are tunable via CLI args.

For high-volume workloads, the pipeline also uses **async HTTP** (`aiohttp`) to overlap multiple batch calls concurrently within a single worker, providing an additional ~2-4x throughput improvement on top of batching.

### Key Benefits

- **Cost Effective**: Cloud Run FastAPI services cost ~90% less than Vertex AI endpoints
- **Scalable**: Dataflow auto-scales based on Pub/Sub message volume
- **Reliable**: Only production-ready "blessed" models are deployed
- **Observable**: All predictions logged to BigQuery with metadata

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
