# ML Pipelines with Kubeflow

This repository demonstrates the deployment of a simple training and inference pipeline using Kubeflow Pipelines (KFP). The project focuses on training and deploying an Iris classifier model on Google Cloud Platform's Vertex AI.

## Project Overview

The project implements two main pipelines:
1. **Training Pipeline**: Trains multiple models (e.g., Decision Tree, Random Forest) on the Iris dataset, evaluates their performance, and deploys the best model.
2. **Inference Pipeline**: Loads the deployed model and performs batch inference.

## Prerequisites

- Python 3.9 or later
- Google Cloud Platform account with Vertex AI enabled
- Service account with necessary permissions
- Google Cloud SDK installed

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ml_pipelines_kfp
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Set up environment variables:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
   ```

## Project Structure

```
├── schemas/                    # Model schema definitions
│   └── iris_xgboost/
│       └── vertex/            # Vertex AI specific schemas
├── src/
│   └── ml_pipelines_kfp/
│       └── iris_xgboost/
│           ├── data/          # Training data
│           ├── pipelines/     # Pipeline definitions
│           └── models/        # Model implementations
├── notebooks/                  # Example notebooks
├── workflows/                  # Workflow components and configurations
```

## Pipeline Components

### Training Pipeline
- **Data Loading**: Loads data from BigQuery.
- **Model Training**: Trains models like Decision Tree and Random Forest.
- **Model Evaluation**: Evaluates model performance.
- **Model Selection**: Selects the best-performing model.
- **Model Registration**: Registers the model in Vertex AI.
- **Model Deployment**: Deploys the model to Vertex AI.

### Inference Pipeline
- **Model Loading**: Loads the deployed model.
- **Batch Inference**: Performs inference on a batch of data.
- **Results Storage**: Stores inference results.

## Usage

### Training Pipeline

To run the training pipeline:
```bash
python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py
```

### Inference Pipeline

To run the inference pipeline:
```bash
python src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_inference.py
```

## Configuration

Key configurations are stored in `constants.py`:
- Project ID
- Region
- GCS bucket information
- Model and endpoint names
- Image configurations

## Development

### Local Development
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -e .
   ```

### Docker Development
Build the Docker image:
```bash
docker build -t ml-pipelines-kfp .
```

## License

[Add your license information here]