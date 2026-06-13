# Plan: Add Vertex AI Feature Store to ML Pipeline

## Context

The repo has an end-to-end ML pipeline (KFP on Vertex AI) for Iris classification. Currently, features are read directly from BigQuery with no centralized feature definitions — column names differ across paths (CamelCase in BQ/training, snake_case in Pub/Sub/streaming), there's a brittle conditional rename in batch inference (`inference.py:45-56`), and the FastAPI `Instance` model uses `Optional[float]` with no null validation. This creates training/serving skew risk.

Adding Vertex AI Feature Store (V2, BigQuery-backed) solves this by providing:
- A single source of truth for feature schemas and column names
- Consistent offline reads (training + batch inference via BQ feature table)
- Online reads (FastAPI serving via Feature Online Store)
- Elimination of ad-hoc column renaming scattered across components

The existing `google-cloud-aiplatform>=1.59.0` dependency already includes the Feature Store V2 API — no new packages required (just a version bump to `>=1.64.0`).

---

## Implementation Steps

### Step 1: Feature Definitions (foundation)

**Create `src/ml_pipelines_kfp/iris_xgboost/feature_store/__init__.py`** — empty init

**Create `src/ml_pipelines_kfp/iris_xgboost/feature_store/feature_definitions.py`** — single source of truth:
- Canonical feature column names: `sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, `petal_width_cm`
- Entity ID column name: `entity_id`
- Target column: `species`
- Mapping dicts: CamelCase to canonical, snake_case to canonical
- Feature Store resource name constants

**Modify `src/ml_pipelines_kfp/iris_xgboost/constants.py`** — add:
- `FEATURE_ONLINE_STORE_ID = "iris_online_store"`
- `FEATURE_VIEW_ID = "iris_features"`
- `BQ_FEATURE_TABLE = "iris_features"`

### Step 2: Feature Ingestion KFP Component

**Create `src/ml_pipelines_kfp/iris_xgboost/feature_store/ingest.py`** — a `@component` that:
1. Reads raw data from `ml_dataset.iris` (BQ)
2. Renames columns to canonical names using the mapping
3. Adds `entity_id` (row index) and `feature_timestamp` columns (required by Feature Store)
4. Writes to `ml_dataset.iris_features` (canonical BQ feature table)

Follows the same `@component(base_image="python:3.10", packages_to_install=[...])` pattern as existing components in `pipelines/components/data.py`. Column mappings passed as parameters (since KFP serializes functions and can't import project modules inside the component body).

### Step 3: Feature Store Infrastructure Setup Script

**Create `scripts/setup_feature_store.py`** — standalone, idempotent script (run once):
1. Creates a `FeatureOnlineStore` (Bigtable-backed) named `iris_online_store`
2. Creates a `FeatureView` named `iris_features` pointing to `bq://deeplearning-sahil.ml_dataset.iris_features`
3. Uses `google.cloud.aiplatform.FeatureOnlineStore.create_bigtable_store()` and `create_feature_view()`

### Step 4: Feature Store Sync KFP Component

**Create `src/ml_pipelines_kfp/iris_xgboost/pipelines/components/feature_store_sync.py`** — a `@component` that triggers a manual `FeatureView.sync()` after ingestion so the online store has fresh data.

### Step 5: Modify Training Pipeline

**Add new component in `src/ml_pipelines_kfp/iris_xgboost/pipelines/components/data.py`**: `load_data_from_feature_store`
- Reads from canonical BQ feature table (`iris_features`) instead of raw `iris` table
- Same train/test split logic as existing `load_data`
- No column renaming needed — canonical table already has standardized names
- Keep existing `load_data` as reference

**Modify `src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py`**:
- New pipeline flow: `ingest_features` to `sync_feature_store` to `load_data_from_feature_store` to models to evaluate to register to deploy
- Import and wire the new components

### Step 6: Update Instance Model + FastAPI Server

**Modify `src/ml_pipelines_kfp/iris_xgboost/models/instance.py`**:
- Switch to canonical field names (`sepal_length_cm`, etc.) as primary, non-optional `float`
- Add Pydantic `Field(alias="SepalLengthCm")` + `populate_by_name=True` for backward compat
- Add `EntityInstance(BaseModel)` with `entity_id: str` for online lookup requests

**Create `src/ml_pipelines_kfp/iris_xgboost/feature_store/online_serving.py`**:
- `fetch_online_features(entity_ids)` — wraps `FeatureView.read()`, returns DataFrame

**Modify `src/ml_pipelines_kfp/iris_xgboost/server.py`**:
- Add `/predict_by_id` endpoint: accepts entity IDs to fetches features from online store to runs prediction
- Existing `/predict` endpoint stays (raw feature input path still works)

**Standardize health endpoint to `/health/live`**:
- Modify `fastapi_server.py` (`components/fastapi/`) — rename `/health` to `/health/live`
- This aligns with the Vertex AI `health_route="/health/live"` already set in `register.py`
- Update the root `/` response to reference `/health/live` instead of `/health`
- Update `deploy.py` health check test to call `/health/live`

### Step 7: Fix Batch Inference

**Modify `src/ml_pipelines_kfp/iris_xgboost/pipelines/components/inference.py`**:
- Read from canonical feature table (`iris_features`) instead of raw tables
- Remove the conditional column renaming block (lines 45-57) — canonical table has consistent names
- Update column references to canonical names

**Modify `src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_inference.py`**:
- Pass `BQ_FEATURE_TABLE` as the data source instead of `BQ_TABLE`

### Step 8: Update Streaming Path

**Modify `src/ml_pipelines_kfp/dataflow/iris_streaming_pipeline.py`**:
- Update `CallFastAPIService` payload to use canonical column names (works because Instance model now accepts both via aliases)
- Add commented alternative showing the `/predict_by_id` pattern where Pub/Sub sends entity IDs and features are fetched from the online store

### Step 9: Dependency + Docs

**Modify `pyproject.toml`**: bump `google-cloud-aiplatform>=1.64.0`

**Modify `Readme.md`**: add Feature Store section covering setup, ingestion flow, online vs offline serving, and the new `/predict_by_id` endpoint

---

## File Summary

| Action | File | What changes |
|--------|------|-------------|
| Create | `feature_store/__init__.py` | Package init |
| Create | `feature_store/feature_definitions.py` | Canonical names, mappings, resource IDs |
| Create | `feature_store/ingest.py` | KFP component: raw BQ to canonical feature table |
| Create | `feature_store/online_serving.py` | Utility: online store reads |
| Create | `pipelines/components/feature_store_sync.py` | KFP component: trigger FeatureView sync |
| Create | `scripts/setup_feature_store.py` | One-time infra setup |
| Modify | `constants.py` | Add feature store constants |
| Modify | `pipelines/components/data.py` | Add `load_data_from_feature_store` component |
| Modify | `pipelines/iris_pipeline_training.py` | Wire feature ingestion + new data loader |
| Modify | `pipelines/components/inference.py` | Read from canonical table, remove conditional rename |
| Modify | `pipelines/iris_pipeline_inference.py` | Point to feature table |
| Modify | `models/instance.py` | Canonical names + aliases + EntityInstance |
| Modify | `server.py` | Add `/predict_by_id` endpoint |
| Modify | `pipelines/components/fastapi/fastapi_server.py` | Rename `/health` to `/health/live`, update root response |
| Modify | `pipelines/components/deploy.py` | Update health check test to `/health/live` |
| Modify | `dataflow/iris_streaming_pipeline.py` | Use canonical column names |
| Modify | `pyproject.toml` | Bump aiplatform version |
| Modify | `Readme.md` | Feature Store docs |

All paths above are relative to `src/ml_pipelines_kfp/iris_xgboost/` unless otherwise shown.

---

## Key Design Decisions

1. **Feature Store V2 (BigQuery-backed)**, not the deprecated Legacy Feature Store — data already lives in BQ, and this is the current Google-recommended approach.
2. **New `load_data_from_feature_store` component** alongside existing `load_data` (not replacing it) — keeps the original as reference since this is a learning repo.
3. **Pydantic aliases for backward compat** — the Instance model accepts both `SepalLengthCm` and `sepal_length_cm`, so existing consumers don't break.
4. **Column mappings as component parameters** — KFP components run in isolated containers and can't import project modules, so canonical names are passed in rather than imported.
5. **Model retraining required** — after this change, models will be trained on canonical column names. First deployment needs a fresh training run.

---

## Verification

1. **Unit**: Run `scripts/setup_feature_store.py` and verify resources appear in Vertex AI console
2. **Training**: Run the training pipeline — confirm `iris_features` BQ table is created with canonical columns, model trains successfully
3. **Batch inference**: Run inference pipeline — confirm it reads from `iris_features` without column rename hacks
4. **FastAPI**: Start server locally (`uvicorn`), test `/predict` with both CamelCase and canonical names, test `/predict_by_id` with an entity ID
5. **Streaming**: Deploy Dataflow job, send Pub/Sub messages, verify predictions flow through with canonical names
