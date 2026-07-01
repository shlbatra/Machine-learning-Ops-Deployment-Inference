from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os
import time
import uvicorn
from google.cloud import storage

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from models.instance import Instance
from models.prediction import Prediction
from log import get_logger

logger = get_logger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# --- OTel metrics setup (environment-aware exporter) ---
resource = Resource.create({"service.name": "fastapi-inference"})
otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

if otel_endpoint:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    exporter = OTLPMetricExporter(endpoint=otel_endpoint, insecure=True)
    logger.info(f"Using OTLP exporter → {otel_endpoint}")
else:
    from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
    exporter = CloudMonitoringMetricsExporter()
    logger.info("Using Cloud Monitoring exporter")

metric_reader = PeriodicExportingMetricReader(
    exporter,
    export_interval_millis=10_000,
)
metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))
meter = metrics.get_meter("fastapi-inference")

prediction_latency = meter.create_histogram(
    name="fastapi.predict.duration",
    description="Model prediction latency (compute only)",
    unit="s",
)
predictions_total = meter.create_counter(
    name="fastapi.predictions.total",
    description="Total predictions served",
)
batch_size_hist = meter.create_histogram(
    name="fastapi.predict.batch_size",
    description="Number of instances per /predict call",
)
model_load_duration = meter.create_histogram(
    name="fastapi.model.load_duration",
    description="Time taken to load the model at startup",
    unit="s",
)

app = FastAPI(
    title="ML Model Inference API",
    description="FastAPI server for ML model inference with Vertex AI compatibility",
    version="1.0.0",
)

FastAPIInstrumentor.instrument_app(app)

model = None

MODEL_FILENAME = "model.joblib"


class PredictionRequest(BaseModel):
    instances: List[Instance]


class PredictionResponse(BaseModel):
    predictions: List[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_type: str
    model_loaded: bool


def download_model_from_gcs(gcs_path: str, local_path: str):
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Expected GCS path starting with gs://, got: {gcs_path}")

    path_parts = gcs_path.replace("gs://", "").split("/")
    bucket_name = path_parts[0]
    blob_path = "/".join(path_parts[1:])

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    logger.info(f"Downloading model from {gcs_path} to {local_path}")
    blob.download_to_filename(local_path)
    logger.info("Model downloaded successfully")


@app.on_event("startup")
async def load_model():
    global model

    model_gcs_path = os.getenv("MODEL_GCS_PATH") or os.getenv("AIP_STORAGE_URI")
    model_path = os.getenv("MODEL_PATH", "/app/model_artifacts/model.joblib")

    start = time.perf_counter()
    try:
        if model_gcs_path:
            if not model_gcs_path.endswith(".joblib"):
                model_gcs_path = model_gcs_path.rstrip("/") + f"/{MODEL_FILENAME}"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            download_model_from_gcs(model_gcs_path, model_path)

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            duration = time.perf_counter() - start
            model_load_duration.record(duration)
            logger.info(f"Model loaded from {model_path} in {duration:.2f}s")
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
        "health_check": "/health/live",
        "prediction": "/predict",
    }


@app.get("/health/live", response_model=HealthResponse)
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
        start = time.perf_counter()
        df = pd.DataFrame(i.model_dump() for i in request.instances)
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        duration = time.perf_counter() - start

        prediction_latency.record(duration)
        predictions_total.add(len(predictions), {"status": "success"})
        batch_size_hist.record(len(request.instances))

        results = [
            Prediction(
                class_=int(pred),
                class_probabilities=proba.tolist(),
            )
            for pred, proba in zip(predictions, probabilities)
        ]

        return PredictionResponse(predictions=results)

    except Exception as e:
        predictions_total.add(1, {"status": "error"})
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
