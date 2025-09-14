from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
import logging
import tempfile
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model Inference API",
    description="FastAPI server for ML model inference with Vertex AI compatibility",
    version="1.0.0",
)

model = None


class PredictionRequest(BaseModel):
    instances: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_type: str
    model_loaded: bool


def download_model_from_gcs(gcs_path: str, local_path: str):
    """Download model from GCS to local path"""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Expected GCS path starting with gs://, got: {gcs_path}")

    # Parse GCS path
    path_parts = gcs_path.replace("gs://", "").split("/")
    bucket_name = path_parts[0]
    blob_path = "/".join(path_parts[1:])

    # Download from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    logger.info(f"Downloading model from {gcs_path} to {local_path}")
    blob.download_to_filename(local_path)
    logger.info(f"Model downloaded successfully")


@app.on_event("startup")
async def load_model():
    global model

    # Check if model should be loaded from GCS
    model_gcs_path = os.getenv("MODEL_GCS_PATH")
    model_path = os.getenv("MODEL_PATH", "/app/models/model.joblib")

    try:
        if model_gcs_path:
            # Download model from GCS
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            download_model_from_gcs(model_gcs_path, model_path)

        # Load model
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Model type: {type(model)}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "ML Model Inference API",
        "health_check": "/health",
        "prediction": "/predict",
        "vertex_ai_endpoint": "/v1/models/model:predict",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy", model_type=str(type(model)), model_loaded=True
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame(request.instances)
        predictions = model.predict(df)

        results = []
        for pred in predictions:
            if isinstance(pred, np.ndarray):
                results.append({"prediction": pred.tolist()})
            elif isinstance(pred, (np.int64, np.int32)):
                results.append({"prediction": int(pred)})
            elif isinstance(pred, (np.float64, np.float32)):
                results.append({"prediction": float(pred)})
            else:
                results.append({"prediction": pred})

        return PredictionResponse(predictions=results)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
