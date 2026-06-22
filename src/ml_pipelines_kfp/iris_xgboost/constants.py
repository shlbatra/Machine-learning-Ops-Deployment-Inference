import os

from ml_pipelines_kfp.constants import (  # noqa: F401
    BUCKET, ENV, PROJECT_ID, REGION, REPO_ROOT, SERVICE_ACCOUNT, SERVICE_ACCOUNT_PATH,
)

# --- Shared across environments ---
BQ_DATASET = "ml_dataset"
BQ_TABLE = "iris"
BQ_FEATURE_TABLE = "iris_features"
PUBSUB_TOPIC = "iris-inference-data"
PUBSUB_SUBSCRIPTION = "iris-inference-data-sub"

# --- Environment-specific ---
if ENV == "prod":
    PIPELINE_NAME = "pipeline-iris-prod"
    PIPELINE_ROOT = f"{BUCKET}/prod/pipeline_root"
    MODEL_NAME = "Iris-Classifier-XGBoost"
    BQ_TABLE_PREDICTIONS = "iris_predictions"
    BQ_TABLE_PREDICTIONS_STREAMING = "iris_predictions_streaming"
    _DEFAULT_IMAGE_TAG = "main"
else:  # staging
    PIPELINE_NAME = "pipeline-iris-staging"
    PIPELINE_ROOT = f"{BUCKET}/staging/pipeline_root"
    MODEL_NAME = "Iris-Classifier-XGBoost-staging"
    BQ_TABLE_PREDICTIONS = "iris_predictions_staging"
    BQ_TABLE_PREDICTIONS_STREAMING = "iris_predictions_streaming_staging"
    _DEFAULT_IMAGE_TAG = os.getenv("BUILD_BRANCH", "staging").replace("/", "-")

IMAGE_NAME = os.getenv(
    "PIPELINE_BASE_IMAGE",
    f"us-docker.pkg.dev/{PROJECT_ID}/sahil-experiment-docker-images/ml-pipelines-kfp-image:{_DEFAULT_IMAGE_TAG}",
)
FASTAPI_IMAGE_NAME = os.getenv(
    "PIPELINE_FASTAPI_IMAGE",
    f"us-docker.pkg.dev/{PROJECT_ID}/sahil-experiment-docker-images/fastapi-ml-generic:{_DEFAULT_IMAGE_TAG}",
)

ENDPOINT_NAME = MODEL_NAME
MODEL_FILENAME = "model.joblib"
PUBSUB_REGION = REGION
