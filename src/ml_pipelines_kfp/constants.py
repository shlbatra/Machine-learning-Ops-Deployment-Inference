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
