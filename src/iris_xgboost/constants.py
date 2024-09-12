import os
from pathlib import Path

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = str(Path(PACKAGE_ROOT).parent.parent.absolute())

# Project settings
BUCKET = "gs://ml-pipelines-kfp"
PIPELINE_NAME = "pipeline-iris"
PIPELINE_ROOT = f"{BUCKET}/pipeline_root"
REGION = "us-east1"
PROJECT_ID = "ml-pipelines-project-433602"
SERVICE_ACCOUNT = "ml-pipelines-sa@ml-pipelines-project-433602.iam.gserviceaccount.com"
MODEL_NAME = "Iris-Classifier-XGBoost-2"
ENDPOINT_NAME = "Iris-Classifier-XGBoost-2"
IMAGE_NAME = "gcr.io/ml-pipelines-project-433602/ml-pipelines-kfp-image:main"