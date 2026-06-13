# Plan: Remove server.py and unify on FastAPI server

## Context

The project has two serving paths that diverged:

1. **Old path (Vertex AI Model Registry):** `register.py` uploads models with a `container_spec` pointing to `server.py` inside the `IMAGE_NAME` Docker image (root `Dockerfile`). But `server.py` has its model loading commented out and returns hardcoded predictions — it's effectively dead code.

2. **New path (Cloud Run):** `deploy.py` deploys the `FASTAPI_IMAGE_NAME` image (`Dockerfile.fastapi` + `fastapi_server.py`) which has fully working model loading from GCS, proper health checks, and real predictions.

The goal is to retire `server.py` and point Vertex AI model registration at the new FastAPI server, so both deployment paths use the same serving code.

## Key difference to bridge

Vertex AI serving automatically sets the `AIP_STORAGE_URI` env var to the model's `artifact_uri` when starting the container. The new FastAPI server reads `MODEL_GCS_PATH` instead. The fix: update `fastapi_server.py` to also check `AIP_STORAGE_URI` as a fallback, so it works in both Vertex AI and Cloud Run contexts.

## Changes

### 1. Update `fastapi_server.py` to support Vertex AI's `AIP_STORAGE_URI`

**File:** `src/ml_pipelines_kfp/iris_xgboost/pipelines/components/fastapi/fastapi_server.py`

In the `load_model()` function (line 61-84), change the env var resolution to:

```python
model_gcs_path = os.getenv("MODEL_GCS_PATH") or os.getenv("AIP_STORAGE_URI")
```

When `AIP_STORAGE_URI` is used, it points to a directory (e.g. `gs://bucket/path/`), not a file. The model file is `model.joblib` inside that directory. So also update the GCS download logic: if the path doesn't end with a file extension, append `/model.joblib`.

Rename the health endpoint from `/health` to `/health/live` so it matches the Vertex AI `health_route="/health/live"` already set in `register.py`:

```python
@app.get("/health/live", response_model=HealthResponse)
async def health_check():
```

Update the root `/` response — change `"health_check": "/health"` to `"health_check": "/health/live"` and remove the dead `"vertex_ai_endpoint": "/v1/models/model:predict"` entry (no such route exists).

Update `deploy.py` line 228 — change the smoke test from `requests.get(f"{service_url}/health", ...)` to `requests.get(f"{service_url}/health/live", ...)`.

### 2. Update `register.py` to use FastAPI image

**File:** `src/ml_pipelines_kfp/iris_xgboost/pipelines/components/register.py`

Change the `container_spec` (lines 46-59):
- Remove the `args` list (the FastAPI Dockerfile already has a CMD)
- Keep `predict_route="/predict"` (same in both servers)
- Keep `health_route="/health/live"` (we're adding the alias in step 1)
- `image_uri` already comes from the `image_name` parameter — no change here

### 3. Update training pipeline to pass FastAPI image to register

**File:** `src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py`

Line 74: change `image_name=IMAGE_NAME` to `image_name=FASTAPI_IMAGE_NAME`

### 4. Update root Dockerfile

**File:** `Dockerfile`

The root Dockerfile's CMD (line 33) currently points to `server.py`. Since this image is still used as `base_image` for some KFP components (`deploy.py`, `schema.py`), it doesn't need a serving CMD at all — those components don't use the CMD. Remove the CMD line since this image is only used as a KFP component base, not as a serving container.

### 5. Use typed Pydantic models in `fastapi_server.py` for strict validation

**Keep these files as-is:**
- `src/ml_pipelines_kfp/iris_xgboost/models/instance.py` — `Instance(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)`
- `src/ml_pipelines_kfp/iris_xgboost/models/prediction.py` — `Prediction(class_, class_probabilities)`
- `src/ml_pipelines_kfp/iris_xgboost/models/__init__.py`

**Modify `src/ml_pipelines_kfp/iris_xgboost/pipelines/components/fastapi/fastapi_server.py`:**

Replace the generic dict-based models with imports from the existing typed models:

```python
from src.ml_pipelines_kfp.iris_xgboost.models.instance import Instance
from src.ml_pipelines_kfp.iris_xgboost.models.prediction import Prediction
```

Replace `PredictionRequest` and `PredictionResponse`:

```python
class PredictionRequest(BaseModel):
    instances: List[Instance]          # was List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[Prediction]      # was List[Dict[str, Any]]
```

Update the `/predict` handler to build `Prediction` objects instead of raw dicts:

```python
results = []
for pred in predictions:
    results.append(Prediction(
        class_=int(pred),
        class_probabilities=model.predict_proba(df)[i].tolist(),
    ))
return PredictionResponse(predictions=results)
```

This gives strict validation — requests missing feature fields or sending wrong types are rejected with a 422 before hitting the model.

**Modify `Dockerfile.fastapi`:**

The FastAPI container needs access to the models package. Either:
- Copy the models directory into the image: `COPY ../../models /app/models`
- Or install the project package in the image

**Modify `requirements.fastapi.txt`:**

No changes needed — `pydantic>=2.0.0` is already listed.

**Delete only:**
- `src/ml_pipelines_kfp/iris_xgboost/server.py`

### 6. Update/delete test files

- **Delete** `test/test_server_locally.py` — tests the old server directly, not salvageable
- **Update** `test/test_model_loading.py` — remove the `test_model_compatibility()` function (lines 17-48) which imports `Instance` and `Prediction` from the old models. The other test functions (`test_joblib_versions`, `test_fsspec_gcs`, `test_environment_variables`, `analyze_pipeline_model`) are server-agnostic and should be kept. Update `test_environment_variables()` to test `MODEL_GCS_PATH` instead of `AIP_STORAGE_URI`

### 7. Recompile pipeline.yaml

Run the pipeline compiler to regenerate `pipeline.yaml` and `src/ml_pipelines_kfp/iris_xgboost/pipelines/pipeline.yaml` so the compiled specs no longer reference `server:app`.

## File summary

| File | Action |
|---|---|
| `components/fastapi/fastapi_server.py` | Add `AIP_STORAGE_URI` fallback, rename `/health` to `/health/live`, remove dead `/v1/models/model:predict` from root, use typed `Instance`/`Prediction` models instead of `Dict[str, Any]` |
| `components/fastapi/Dockerfile.fastapi` | Copy models package into image |
| `components/deploy.py` | Update health check test to `/health/live` |
| `components/register.py` | Remove `args` from container_spec |
| `pipelines/iris_pipeline_training.py` | Pass `FASTAPI_IMAGE_NAME` to `upload_model` |
| `Dockerfile` | Remove CMD line |
| `iris_xgboost/server.py` | **Delete** |
| `iris_xgboost/models/instance.py` | **Keep** — used by `fastapi_server.py` for request validation |
| `iris_xgboost/models/prediction.py` | **Keep** — used by `fastapi_server.py` for response schema |
| `test/test_server_locally.py` | **Delete** |
| `test/test_model_loading.py` | Remove `test_model_compatibility`, update env var test |
| `pipeline.yaml` (root + pipelines/) | **Recompile** |

## Verification

1. `grep -rn "iris_xgboost.server\|server:app" src/ test/ Dockerfile *.yaml` — should return nothing
2. `grep -rn "models.instance\|models.prediction" src/ test/` — should only reference `fastapi_server.py`
3. `python -m py_compile` on all modified files
4. `python -m pytest test/` — tests pass
5. Recompile pipeline: `python -m src.ml_pipelines_kfp.iris_xgboost.pipelines.iris_pipeline_training` (generates updated pipeline.yaml)
6. Verify `pipeline.yaml` no longer references `server:app`
