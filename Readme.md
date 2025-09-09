# ML Pipelines with Kubeflow Pipelines (KFP)

A production-grade ML pipeline implementation using Kubeflow Pipelines (KFP) on Google Cloud Vertex AI. This project demonstrates MLOps best practices for automating end-to-end ML workflows.

## Overview

This repository implements a complete ML pipeline for the Iris dataset classification problem, showcasing:
- Automated data ingestion from BigQuery
- Parallel model training (Decision Tree, Random Forest, XGBoost)
- Automatic model evaluation and selection
- Model registration and versioning in Vertex AI
- Automated deployment to Vertex AI endpoints
- Batch inference capabilities
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
schemas/                    # Input/output schemas for Vertex AI
memory-bank/                # Project context and documentation
Dockerfile                  # Container definition
pyproject.toml              # Project dependencies
pipeline.yaml               # Pipeline configuration
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
- Register the model in Vertex AI Model Registry
- Deploy the model to an endpoint

### 3. Run Inference Pipeline

```bash
# Execute batch inference
python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_inference.py
```

### 4. Local API Server

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
4. **Registry Component**: Manages model versioning
5. **Deployment Component**: Handles endpoint deployment
6. **Inference Component**: Performs batch predictions

## Configuration

Key configuration is managed in `src/ml_pipelines_kfp/iris_xgboost/constants.py`:
- Project ID: `deeplearning-sahil`
- Region: `us-central1`
- Dataset: `ml_datasets.iris`
- Model naming and versioning

## CI/CD

The repository includes GitHub Actions workflow (`.github/workflows/cicd.yaml`) that:
- Builds Docker images
- Pushes to Google Artifact Registry
- Triggers on pushes to main branch

## Technologies

- **Orchestration**: Kubeflow Pipelines 2.8.0
- **Cloud Platform**: Google Cloud (Vertex AI, BigQuery, GCS)
- **ML Frameworks**: scikit-learn, XGBoost
- **API Framework**: FastAPI
- **Data Processing**: Pandas, Polars, Dask
- **Package Management**: uv, Hatchling

## Set up Pub Sub

bash
```
gcloud auth application-default print-access-token
export GOOGLE_APPLICATION_CREDENTIALS="./***.json"
gcloud auth application-default print-access-token
python test_pubsub.py
```