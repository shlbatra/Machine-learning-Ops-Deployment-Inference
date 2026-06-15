# Plan: Add Vertex AI Feature Store to ML Pipeline

## Context

The repo has an end-to-end ML pipeline (KFP on Vertex AI) for Iris classification. Currently, features are read directly from BigQuery with no centralized feature definitions — column names differ across paths (CamelCase in BQ/training, snake_case in Pub/Sub/streaming), there's a brittle conditional rename in batch inference (`inference.py:36-48`), and the FastAPI `Instance` model uses `Optional[float]` with no null validation. This creates training/serving skew risk.

Adding Vertex AI Feature Store (V2, BigQuery-backed) solves this by providing:
- A single source of truth for feature schemas and column names
- Consistent offline reads (training + batch inference via BQ feature table)
- Online reads (FastAPI serving via Feature Online Store)
- Elimination of ad-hoc column renaming scattered across components

The existing `google-cloud-aiplatform>=1.59.0` dependency already includes the Feature Store V2 API — no new packages required (just a version bump to `>=1.64.0`).

---

## How the Feature Platform Serves Each Path

The feature store has two stores backed by the same data: an **offline store** (BQ feature table) for bulk reads and an **online store** (Bigtable) for low-latency lookups. The `feature_timestamp` column written by `ingest.py` is what enables the different query patterns.

### Training — point-in-time join (offline store)

Training needs features as they existed *at the time of each training event* to prevent data leakage (the model must not see future feature values). The query pattern is a point-in-time join:
- Input: a "spine" — a list of `(entity_id, event_timestamp)` pairs representing the training examples
- Output: for each pair, the feature values that were current *at or before* that `event_timestamp`
- Mechanism: query the BQ feature table with a timestamp filter, or use the Vertex AI `FeatureView` offline serving API which handles the point-in-time join automatically

For the iris dataset (static, single ingestion), all rows share the same `feature_timestamp`, so point-in-time is trivially correct. But the pattern is established for when the data evolves over time.

### Batch Inference — latest features (offline store)

Batch inference needs the *most recent* feature values to score all entities (or a filtered subset):
- Input: entity IDs to score (or all entities)
- Output: the latest feature row per entity
- Mechanism: query the BQ feature table for the most recent `feature_timestamp` per `entity_id`, or simply read the full table if `ingest.py` uses `WRITE_TRUNCATE` (each run is a full snapshot)

Same offline store as training, different query: training looks back in time, batch inference wants current state.

### Real-time Inference — online store lookup

Real-time serving needs features for a single entity (or small batch) with low latency:
- Input: one or more `entity_id` values (e.g. from a Pub/Sub message or API request)
- Output: the latest feature values, served from the online store (Bigtable)
- Mechanism: `FeatureView.read(key=[entity_id])` via the `/predict_by_id` endpoint or the streaming pipeline's online lookup path

The online store is synced from the offline store by `sync.py`. It always serves the latest snapshot — no timestamp query needed, just a key lookup.

### Summary

| Path | Store | Query pattern | Latency | Data freshness |
|------|-------|--------------|---------|----------------|
| Training | Offline (BQ) | Point-in-time join on `feature_timestamp` | Seconds–minutes | Historical |
| Batch inference | Offline (BQ) | Latest per entity | Seconds–minutes | Latest snapshot |
| Real-time inference | Online (Bigtable) | Key lookup by `entity_id` | Milliseconds | Last sync |

### How this applies to the Iris example

The original iris dataset is 150 labeled rows. To demonstrate the training vs inference split, `bq_dataloader.py` can also append random unlabeled rows simulating new data arriving for scoring. Here's the concrete flow:

**Setup (one-time):**
1. `bq_dataloader.py` loads 150 labeled rows to `ml_dataset.iris` (`WRITE_TRUNCATE`)
2. `ingest.py` reads `ml_dataset.iris` → canonical columns → `ml_dataset.iris_features` (150 rows, all with `feature_timestamp = T1`)
3. `setup.py` creates the online store + feature view
4. `sync.py` pushes to online store

**Training:**
- `load_data_from_feature_store` reads from `iris_features`, filters to rows with `species` (labeled data), drops `entity_id` + `feature_timestamp`, does 80/20 train/test split
- In a production dataset with ongoing ingestion, you'd also filter `feature_timestamp <= training_cutoff_date` for point-in-time correctness

**New data arrives → batch inference:**
1. `bq_dataloader.py --generate-random N` appends N random unlabeled rows to `ml_dataset.iris` (`WRITE_APPEND`)
2. `ingest.py` re-runs → `iris_features` now has 150 + N rows, new rows get `feature_timestamp = T2`
3. `sync.py` re-runs → online store updated
4. Batch inference reads the latest snapshot from `iris_features`, scores all rows (or filters to `species IS NULL` for only new/unlabeled rows), writes predictions to `iris_predictions`
5. The conditional column rename hack (lines 36-48) is gone — all data is already canonical

**Real-time inference:**
- A request with `entity_id: "155"` (a new row) hits `/predict_by_id` → online store returns features in milliseconds → model predicts → response
- The Dataflow streaming path can also use this: Pub/Sub message contains just an `entity_id` instead of all four feature values

---

## Implementation Steps

### Step 1: Feature Definitions (foundation)

**Create `src/feature_store/schema.py`** — shared `FeatureConfig` dataclass that defines the contract every ML project must follow: feature columns, entity/target/timestamp column names, column mappings (source → canonical), BQ table references, and Feature Store resource IDs. Frozen dataclass for immutability. Includes a `canonical_to_source` property for reverse lookups.

**Create `src/feature_store/iris/__init__.py`** — iris sub-package init

**Create `src/feature_store/iris/feature_definitions.py`** — iris-specific `IRIS_CONFIG` instance of `FeatureConfig`:
- Canonical feature column names: `sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, `petal_width_cm`
- Entity ID column name: `entity_id`
- Target column: `species`
- Column mappings: `camel` (BQ raw CamelCase → canonical), `snake` (Pub/Sub snake_case → canonical)
- Feature Store resource name constants (`iris_online_store`, `iris_features`, `iris_features` BQ table)

To add a new ML project, create `src/feature_store/<project>/feature_definitions.py` with its own `FeatureConfig` instance — same contract, different values.

### Step 2: Simulate New Data for Inference

**Modify `scripts/bq_dataloader.py`** — add a function to generate random iris-like rows and append them to `ml_dataset.iris`:
- Generate N random rows with realistic feature ranges (e.g. `sepal_length` 4.3–7.9, `sepal_width` 2.0–4.4, etc. based on min/max from the real dataset)
- Assign new `Id` values continuing from the existing max (150+)
- No `Species` label — these are unlabeled rows simulating new data arriving for scoring
- Write with `WRITE_APPEND` (adds to existing data, doesn't overwrite the training rows)

This gives batch inference new rows to score. The flow becomes:
1. `bq_dataloader.py` loads original 150 rows (existing, `WRITE_TRUNCATE`) + optionally appends random new rows (`WRITE_APPEND`)
2. `ingest.py` reads all rows from `ml_dataset.iris` → writes canonical version to `ml_dataset.iris_features`
3. Batch inference scores the latest snapshot from `iris_features` — including the new rows

### Step 3: Feature Ingestion

**Create `src/feature_store/ingest.py`** — standalone Python script (run directly, not a KFP component):
1. Reads raw data from `ml_dataset.iris` (BQ)
2. Renames columns to canonical names using mappings from `feature_definitions.py` (drops the raw `Id` column)
3. Adds `entity_id` (row index) and `feature_timestamp` columns (required by Feature Store)
4. Writes to `ml_dataset.iris_features` (canonical BQ feature table) using `write_disposition="WRITE_TRUNCATE"` (full refresh each run — always reflects the latest state of the raw table)

This is a data prep step that runs independently before the pipeline, similar to `setup.py` and the existing `scripts/bq_dataloader.py`. No KFP orchestration needed.

### Step 4: Feature Store Infrastructure Setup Script

**Create `src/feature_store/setup.py`** — standalone, idempotent script (run once):
1. Creates a `FeatureOnlineStore` (Bigtable-backed) named `iris_online_store`
2. Creates a `FeatureView` named `iris_features` pointing to `bq://deeplearning-sahil.ml_dataset.iris_features`
3. Uses `google.cloud.aiplatform.FeatureOnlineStore.create_bigtable_store()` and `create_feature_view()`
4. Imports resource name constants from `feature_definitions.py`

### Step 5: Feature Store Sync

**Create `src/feature_store/sync.py`** — standalone script that triggers a manual `FeatureView.sync()` after ingestion so the online store has fresh data. Run after `ingest.py` — both are pre-pipeline prep steps.

### Step 6: Modify Training Pipeline (offline features)

Training uses the **offline store** — reads historical features from the BQ-backed feature table. This is the standard feature platform pattern: train on point-in-time correct features from the offline store.

**Add new component in `src/ml_pipelines_kfp/iris_xgboost/pipelines/components/data.py`**: `load_data_from_feature_store`
- Reads from canonical BQ feature table (`iris_features`) — the offline store
- Drops `entity_id` and `feature_timestamp` columns before train/test split (Feature Store metadata, not model features)
- Same train/test split logic as existing `load_data`, but references lowercase `species` (canonical name)
- No column renaming needed — canonical table already has standardized names
- Keep existing `load_data` as reference

**Modify `src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py`**:
- New pipeline flow: `load_data_from_feature_store` → models → evaluate → register → deploy
- Assumes `ingest.py` and `sync.py` have already been run before pipeline submission
- Import and wire the new `load_data_from_feature_store` component

### Step 7: Update Instance Model + FastAPI Server

**Modify `src/ml_pipelines_kfp/iris_xgboost/models/instance.py`**:
- Switch to canonical field names (`sepal_length_cm`, etc.) as primary, non-optional `float`
- Add Pydantic `Field(alias="SepalLengthCm")` + `populate_by_name=True` for backward compat
- Add `EntityInstance(BaseModel)` with `entity_id: str` for online lookup requests

**Create `src/feature_store/online_serving.py`**:
- `fetch_online_features(entity_ids)` — wraps `FeatureView.read()`, returns DataFrame

**Modify `src/ml_pipelines_kfp/iris_xgboost/pipelines/components/fastapi/fastapi_server.py`**:
- Add `/predict_by_id` endpoint: accepts entity IDs, fetches features from online store, runs prediction
- Existing `/predict` endpoint stays (raw feature input path still works)
- Health endpoint (`/health/live`), root response, and deploy.py health check are already correct — no changes needed

### Step 8: Fix Batch Inference (offline features)

Batch inference also uses the **offline store** — scores features from the BQ feature table in bulk. Same data source as training, different purpose: training splits for model building, inference scores all rows (or a filtered subset) with the trained model.

**Modify `src/ml_pipelines_kfp/iris_xgboost/pipelines/components/inference.py`**:
- Read from canonical feature table (`iris_features`) — the offline store
- Remove the conditional column renaming block (lines 36-48) — canonical table has consistent names regardless of source
- Update column references to canonical names

**Modify `src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_inference.py`**:
- Pass `BQ_FEATURE_TABLE` as the data source instead of `BQ_TABLE`

### Step 9: Streaming Feature Pipeline (Pub/Sub → Feature Store)

Decouples feature ingestion from inference. This pipeline's only job is to persist incoming streaming data into the feature platform.

**Create `src/dataflow/iris_feature_pipeline.py`** — new Beam streaming pipeline:
- Reads from Pub/Sub (same `iris-inference-data` topic)
- `ParsePubSubMessage` — same parsing as existing pipeline
- `WriteToFeatureStore` DoFn:
  - Renames Pub/Sub fields (`sepal_length`, etc.) to canonical names (`sepal_length_cm`, etc.)
  - Assigns `entity_id` (from message `sample_id` or auto-generated)
  - Sets `feature_timestamp` to message timestamp
  - Writes to `ml_dataset.iris_features` BQ table (`WRITE_APPEND` — adds rows, unlike batch `ingest.py` which does `WRITE_TRUNCATE`)
- Triggers `FeatureView.sync()` periodically or per-batch so the online store stays fresh
- Writes to BQ only (no model calls, no prediction output)

### Step 10: Streaming Inference Pipeline (Feature Store → Predictions)

This pipeline reads features from the online store and runs inference — fully decoupled from feature ingestion.

**Modify `src/dataflow/iris_streaming_pipeline.py`** — refactor the existing pipeline:
- `ParsePubSubMessage` stays — but now the message only needs to carry an `entity_id` (or `sample_id`), not the full feature payload
- Replace `BatchCallFastAPIService` field mapping (lines 96-104) with an online store lookup: read features by `entity_id` from the online store, then call `/predict` (or `/predict_by_id` which does the lookup server-side)
- Keep the existing BQ prediction write + metadata steps unchanged
- The pipeline can also run independently of the feature pipeline — if features were written by batch `ingest.py` instead of streaming, inference still works via the online store

**Why two pipelines:**
- Feature ingestion and inference scale independently
- If the model endpoint is down, features still persist — no data loss
- Features are reusable: other models, analytics, or monitoring can read from the same feature store
- Each pipeline can be restarted, updated, or rolled back without affecting the other

### Step 11: Dependency + Docs

**Modify `pyproject.toml`**: bump `google-cloud-aiplatform>=1.64.0`

**Modify `Readme.md`**: add Feature Store section covering setup, ingestion flow, online vs offline serving, and the new `/predict_by_id` endpoint

---

## File Summary

| Action | File | What changes |
|--------|------|-------------|
| **`src/feature_store/` (new package)** | | |
| Create | `src/feature_store/schema.py` | Shared `FeatureConfig` dataclass — contract for all ML projects |
| Create | `src/feature_store/iris/__init__.py` | Iris sub-package init |
| Create | `src/feature_store/iris/feature_definitions.py` | `IRIS_CONFIG` instance with canonical names, mappings, resource IDs |
| Create | `src/feature_store/ingest.py` | Script: raw BQ → canonical feature table |
| Create | `src/feature_store/sync.py` | Script: trigger FeatureView sync after ingestion |
| Create | `src/feature_store/online_serving.py` | Utility: online store reads |
| Create | `src/feature_store/setup.py` | One-time infra setup script |
| **`src/ml_pipelines_kfp/` (existing)** | | |
| Modify | `iris_xgboost/pipelines/components/data.py` | Add `load_data_from_feature_store` component |
| Modify | `iris_xgboost/pipelines/iris_pipeline_training.py` | Wire feature ingestion + new data loader |
| Modify | `iris_xgboost/pipelines/components/inference.py` | Read from canonical table, remove conditional rename |
| Modify | `iris_xgboost/pipelines/iris_pipeline_inference.py` | Point to feature table |
| Modify | `iris_xgboost/models/instance.py` | Canonical names + aliases + EntityInstance |
| Modify | `iris_xgboost/pipelines/components/fastapi/fastapi_server.py` | Add `/predict_by_id` endpoint |
| **`src/dataflow/` (streaming)** | | |
| Create | `src/dataflow/iris_feature_pipeline.py` | Streaming feature pipeline: Pub/Sub → Feature Store |
| Modify | `src/dataflow/iris_streaming_pipeline.py` | Streaming inference pipeline: read from online store → predict |
| **Other** | | |
| Modify | `scripts/bq_dataloader.py` | Add random row generation for simulating new inference data |
| Modify | `pyproject.toml` | Bump aiplatform version |
| Modify | `Readme.md` | Feature Store docs |

---

## Key Design Decisions

1. **Feature Store V2 (BigQuery-backed)**, not the deprecated Legacy Feature Store — data already lives in BQ, and this is the current Google-recommended approach.
2. **New `load_data_from_feature_store` component** alongside existing `load_data` (not replacing it) — keeps the original as reference since this is a learning repo.
3. **Pydantic aliases for backward compat** — the Instance model accepts both `SepalLengthCm` and `sepal_length_cm`, so existing consumers don't break.
4. **`feature_store` as a separate package with per-project sub-packages** — lives in `src/feature_store/`, independent from `ml_pipelines_kfp`. A shared `FeatureConfig` dataclass (`schema.py`) defines the contract; each ML project gets its own sub-package (`iris/`, `fraud/`, etc.) with a `feature_definitions.py` that instantiates the config. New projects follow the same pattern without touching shared code.
5. **Two streaming pipelines (feature + inference)** — decoupled so feature ingestion and inference scale independently. Features persist even if the model is down, and the feature store becomes a shared asset for any downstream consumer.
6. **Model retraining required** — after this change, models will be trained on canonical column names. First deployment needs a fresh training run.

---

## Verification

1. **Infra setup**: Run `setup.py` and verify Feature Online Store + Feature View appear in Vertex AI console
2. **Batch ingestion**: Run `ingest.py` → verify `iris_features` BQ table has canonical columns, `entity_id`, `feature_timestamp`
3. **Sync**: Run `sync.py` → verify online store is populated
4. **Training**: Run the training pipeline — confirm model trains on canonical column names from `iris_features`
5. **New data**: Run `bq_dataloader.py --generate-random 20` → re-run `ingest.py` + `sync.py` → verify `iris_features` now has 170 rows
6. **Batch inference**: Run inference pipeline — confirm it reads from `iris_features` without column rename hacks, scores all rows
7. **FastAPI**: Start server locally (`uvicorn`), test `/predict` with both CamelCase and canonical names, test `/predict_by_id` with an entity ID
8. **Streaming feature pipeline**: Deploy `iris_feature_pipeline.py`, send Pub/Sub messages, verify new rows appear in `iris_features` and online store
9. **Streaming inference pipeline**: Deploy `iris_streaming_pipeline.py`, send Pub/Sub messages with entity IDs, verify predictions written to BQ
