import os
from pathlib import Path

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(Path(PACKAGE_ROOT).parent.parent.parent.absolute())  # "ml_pipelines_kfp"
SERVICE_ACCOUNT_PATH = os.path.join(REPO_ROOT, "deeplearning-sahil-e50332de6687.json")

# Project settings
BUCKET = "gs://sb-vertex"
PIPELINE_NAME = "pipeline-iris"
PIPELINE_ROOT = f"{BUCKET}/pipeline_root"
REGION = "us-central1"
PROJECT_ID = "deeplearning-sahil"
SERVICE_ACCOUNT = "kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"
MODEL_NAME = "Iris-Classifier-XGBoost"
ENDPOINT_NAME = "Iris-Classifier-XGBoost"
IMAGE_NAME = "us-docker.pkg.dev/deeplearning-sahil/sahil-experiment-docker-images/ml-pipelines-kfp-image:main"
MODEL_FILENAME = "model.joblib"

# BigQuery settings
BQ_DATASET = "ml_dataset"
BQ_TABLE = "iris"
BQ_TABLE_PREDICTIONS = "iris_predictions"

# Cloud Pub/Sub settings
PUBSUB_TOPIC = "iris-inference-data"
PUBSUB_SUBSCRIPTION = "iris-inference-data-sub"
PUBSUB_REGION = REGION  # us-central1