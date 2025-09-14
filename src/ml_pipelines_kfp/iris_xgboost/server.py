import os
import sys
from logging import StreamHandler
from logging import getLogger
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import fsspec
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.ml_pipelines_kfp.iris_xgboost.constants import MODEL_FILENAME
from src.ml_pipelines_kfp.iris_xgboost.models.instance import Instance
from src.ml_pipelines_kfp.iris_xgboost.models.prediction import Prediction

log = getLogger()
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
log.addHandler(StreamHandler(sys.stdout))

MODELS = {}


def init_model():
    # model_uri = f'{os.environ["AIP_STORAGE_URI"]}/{MODEL_FILENAME}'
    # log.info(f"Loading model from {model_uri}")
    # with fsspec.open(model_uri, "rb") as f:
    #     MODELS["best_model"] = joblib.load(f)

    log.info("Model loaded")


def build_app() -> FastAPI:
    return FastAPI(
        title="Iris Model",
        on_startup=[init_model],
    )


class PredictRequest(BaseModel):
    instances: List[Instance]
    parameters: Optional[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[Prediction]


app = build_app()


@app.get("/health/live")
async def live():
    return ""


@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    df = pd.DataFrame(i.model_dump() for i in request.instances)
    log.info(df.shape)
    # model = MODELS["best_model"]
    # classes = model.predict(df).tolist()
    # class_probabilities_list = model.predict_proba(df).tolist()

    # return PredictResponse(
    #     predictions=[Prediction(class_=c, class_probabilities=cp) for c, cp in zip(classes, class_probabilities_list)]
    # )

    return PredictResponse(
        predictions=[Prediction(class_=0, class_probabilities=[0.1, 0.5, 0.4])]
    )
