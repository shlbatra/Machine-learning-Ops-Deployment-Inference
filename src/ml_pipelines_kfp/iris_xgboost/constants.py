import os
from pathlib import Path

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(Path(PACKAGE_ROOT).parent.parent.parent.absolute())  # "ml_pipelines_kfp"
SERVICE_ACCOUNT_PATH = os.path.join(REPO_ROOT, "deeplearning-sahil-e50332de6687.json")
# print(PACKAGE_ROOT)
# print(REPO_ROOT)
# print(f"Service Account Key Path: {SERVICE_ACCOUNT_KEY}")

# Project settings
BUCKET = "gs://sb-vertex"
PIPELINE_NAME = "pipeline-iris"
PIPELINE_ROOT = f"{BUCKET}/pipeline_root"
REGION = "us-east1"
PROJECT_ID = "deeplearning-sahil"
SERVICE_ACCOUNT = "kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"
MODEL_NAME = "Iris-Classifier-XGBoost"
ENDPOINT_NAME = "Iris-Classifier-XGBoost"
IMAGE_NAME = "us-docker.pkg.dev/deeplearning-sahil/sahil-experiment-docker-images/ml-pipelines-kfp-image:main"


# BigQuery settings
BQ_DATASET = "ml_dataset"
BQ_TABLE = "iris"