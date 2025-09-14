# ML Pipelines with Kubeflow Pipelines (KFP)

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
├── iris_xgboost/           # Main Iris classification implementation
│   ├── pipelines/          # KFP pipeline definitions
│   │   ├── components/     # Reusable pipeline components
│   │   ├── iris_pipeline_training.py
│   │   └── iris_pipeline_inference.py
│   ├── models/             # Pydantic models for API
│   ├── server.py           # FastAPI serving application
│   ├── bq_dataloader.py    # BigQuery data loading utility
│   └── constants.py        # Configuration constants
├── workflows/              # Alternative workflow implementations
└── notebooks/              # Example notebooks and experiments
├── dataflow/               # Dataflow streaming pipelines
│   └── iris_streaming_pipeline.py
schemas/                    # Input/output schemas for Vertex AI
memory-bank/                # Project context and documentation
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

## Setup Kafka 

- Setup Kafka infrastructure
./scripts/setup_kafka.sh

- Start data production
docker-compose -f docker-compose.kafka.yml up iris-data-producer

- Run the Kafka-enabled inference pipeline
python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_inference_kafka.py

- Monitor via Kafka UI
Visit http://localhost:8080

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml_pipelines_kfp

# Install dependencies
uv pip install -e .
```

## Usage

### 1. Load Data to BigQuery

```bash
# Set up credentials and load Iris dataset
./src/ml_pipelines_kfp/iris_xgboost/load_data.sh
```

### 2. Run Training Pipeline

```bash
# Execute the training pipeline on Vertex AI
python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py
```

This will:
- Load data from BigQuery
- Train Decision Tree and Random Forest models in parallel
- Evaluate and select the best model
- Register the model in Vertex AI Model Registry with "blessed" alias
- Deploy the blessed model to FastAPI service on Cloud Run

### 3. Run Inference Pipeline

```bash
# Execute batch inference
python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_inference.py
```

### 4. Real-time Streaming Inference

Deploy a Dataflow streaming job for real-time inference:

```bash
# Deploy streaming pipeline (update SERVICE_URL with actual Cloud Run URL)
./deploy_dataflow_streaming.sh
```

Start generating test data:

```bash
# Run data producer to send samples to Pub/Sub
python src/ml_pipelines_kfp/iris_xgboost/pubsub_producer.py --project-id deeplearning-sahil
```

### 5. Local API Server

```bash
# Run the FastAPI server locally
uvicorn src.ml_pipelines_kfp.iris_xgboost.server:app --reload
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

### Docker

```bash
# Build image
docker build -t ml-pipelines-kfp .
```

## Architecture

The project follows a component-based architecture where each ML pipeline step is a self-contained KFP component:

1. **Data Component**: Loads and splits data from BigQuery
2. **Model Components**: Implements various ML algorithms
3. **Evaluation Component**: Compares model performance
4. **Registry Component**: Manages model versioning with "blessed" aliases
5. **Deployment Component**: Deploys blessed models to Cloud Run FastAPI services
6. **Inference Component**: Performs batch predictions
7. **Streaming Component**: Real-time inference via Dataflow and Pub/Sub

## Configuration

Key configuration is managed in `src/ml_pipelines_kfp/iris_xgboost/constants.py`:
- Project ID: `deeplearning-sahil`
- Region: `us-central1`
- Dataset: `ml_datasets.iris`
- Model naming and versioning

## CI/CD

The repository includes GitHub Actions workflow (`.github/workflows/cicd.yaml`) that:
- Builds Docker images for KFP components
- Builds generic FastAPI inference containers
- Pushes to Google Artifact Registry
- Triggers on pushes to main branch

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
2. **Stream Processing**: Dataflow processes messages and calls FastAPI services
3. **Model Serving**: Cloud Run hosts FastAPI containers with blessed models
4. **Results Storage**: Predictions are written to BigQuery for monitoring

### Key Benefits

- **Cost Effective**: Cloud Run FastAPI services cost ~90% less than Vertex AI endpoints
- **Scalable**: Dataflow auto-scales based on Pub/Sub message volume
- **Reliable**: Only production-ready "blessed" models are deployed
- **Observable**: All predictions logged to BigQuery with metadata