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

Also add a `/health/live` alias endpoint so Vertex AI's existing health route config works without changes:

```python
@app.get("/health/live")
async def health_live():
    return await health_check()
```

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

### 5. Delete `server.py` and its models

**Delete these files:**
- `src/ml_pipelines_kfp/iris_xgboost/server.py`
- `src/ml_pipelines_kfp/iris_xgboost/models/instance.py`
- `src/ml_pipelines_kfp/iris_xgboost/models/prediction.py`
- `src/ml_pipelines_kfp/iris_xgboost/models/__init__.py` (if it only exists for the above)

### 6. Update/delete test files

- **Delete** `test/test_server_locally.py` — tests the old server directly, not salvageable
- **Update** `test/test_model_loading.py` — remove the `test_model_compatibility()` function (lines 17-48) which imports `Instance` and `Prediction` from the old models. The other test functions (`test_joblib_versions`, `test_fsspec_gcs`, `test_environment_variables`, `analyze_pipeline_model`) are server-agnostic and should be kept. Update `test_environment_variables()` to test `MODEL_GCS_PATH` instead of `AIP_STORAGE_URI`

### 7. Recompile pipeline.yaml

Run the pipeline compiler to regenerate `pipeline.yaml` and `src/ml_pipelines_kfp/iris_xgboost/pipelines/pipeline.yaml` so the compiled specs no longer reference `server:app`.

## File summary

| File | Action |
|---|---|
| `components/fastapi/fastapi_server.py` | Add `AIP_STORAGE_URI` fallback + `/health/live` alias |
| `components/register.py` | Remove `args` from container_spec |
| `pipelines/iris_pipeline_training.py` | Pass `FASTAPI_IMAGE_NAME` to `upload_model` |
| `Dockerfile` | Remove CMD line |
| `iris_xgboost/server.py` | **Delete** |
| `iris_xgboost/models/instance.py` | **Delete** |
| `iris_xgboost/models/prediction.py` | **Delete** |
| `iris_xgboost/models/__init__.py` | **Delete** (if empty after above) |
| `test/test_server_locally.py` | **Delete** |
| `test/test_model_loading.py` | Remove `test_model_compatibility`, update env var test |
| `pipeline.yaml` (root + pipelines/) | **Recompile** |

## Verification

1. `grep -rn "iris_xgboost.server\|server:app" src/ test/ Dockerfile *.yaml` — should return nothing
2. `grep -rn "models.instance\|models.prediction" src/ test/` — should return nothing
3. `python -m py_compile` on all modified files
4. `python -m pytest test/` — tests pass
5. Recompile pipeline: `python -m src.ml_pipelines_kfp.iris_xgboost.pipelines.iris_pipeline_training` (generates updated pipeline.yaml)
6. Verify `pipeline.yaml` no longer references `server:app`
