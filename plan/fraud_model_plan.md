# Fraud Detection Model - Implementation Plan

## Overview

Build a fraud detection system modeled after the existing iris pipeline architecture, with three key enhancements:
1. **Complex feature engineering** across users and transfers tables
2. **Hyperparameter tuning** with Optuna integrated into the KFP training pipeline (LightGBM)
3. **Low-latency real-time inference** via a direct Cloud Run FastAPI service (replacing the Dataflow inference pipeline)

---

## Data Model (from `plan/code.py`)

### Source Tables

**`fraud_users`** — user profiles with fraud labels
| Column | Type | Notes |
|--------|------|-------|
| user_id | STRING | UUID, primary key |
| first_name | STRING | |
| last_name | STRING | |
| email | STRING | Unique |
| country_of_residence | STRING | GB, US, DE, AU, SG, BE, FR, CA |
| account_type | STRING | Personal (85%) / Business (15%) |
| kyc_status | STRING | Verified (90%) / Pending (7%) / Rejected (3%) |
| registration_date | TIMESTAMP | Up to 3 years ago |
| label | INT | 0 = legit (90%), 1 = fraud (10%) |

**`fraud_transfers`** — money transfer events
| Column | Type | Notes |
|--------|------|-------|
| transfer_id | STRING | UUID, primary key |
| sender_id | STRING | FK to fraud_users.user_id |
| recipient_id | STRING | FK to fraud_users.user_id |
| source_currency | STRING | GBP, EUR, USD, AUD, JPY, CAD |
| target_currency | STRING | Same set |
| source_amount | FLOAT | Log-normal distribution, median ~100 |
| target_amount | FLOAT | source_amount * exchange_rate |
| status | STRING | COMPLETED (92%) / PENDING / CANCELLED / FAILED |
| created_at | TIMESTAMP | Up to 2 years ago |

### Feature Tables: Two Granularities

Fraud features live in **two tables at different grain**, because training and real-time serving key on different entities:

| Table | Grain | Entity key | Written by | Purpose | Synced to Bigtable? |
|-------|-------|------------|------------|---------|---------------------|
| `fraud_features` | **transfer** (one row per transfer) | `entity_id = transfer_id + source` | batch ingest (point-in-time join) | training + batch inference | **No** — offline only |
| `fraud_user_features` | **user** (one row per user = latest snapshot) | `entity_id = user_id` (looked up by `sender_id`) | batch ingest (latest per user) + streaming pipeline | real-time serving | **Yes** — online store |

- The **transfer-grained** table is what the model trains on: every transfer is one labeled example with its features computed *as of that transfer's `created_at`* (point-in-time correct — see review finding #3).
- The **user-grained** table holds per-user attributes plus **velocity bucket ring buffers** (hourly ×24, daily ×30) — not rolled-up velocity scalars, which is what let the online store go stale (finding #6). The scalars the model consumes are derived from the buckets at request time by `velocity.merge_buckets()` (§3b). It is the online-store source: the FastAPI server fetches by `sender_id`, and the `__global__` cold-start entity (see Decisions) lives here.
- **Transaction-level features** (`source_amount`, `hour_of_day`, `day_of_week`, `is_cross_border`, `amount_ratio_to_avg`) are *not* stored in `fraud_user_features` — they are transaction-specific and computed at request time from the payload (and per-row in the training SQL).

The feature groups below map to tables as annotated in each subsection.

### Engineered Features

**User-level features** (slow-changing, precomputed in batch) — *→ both tables*:
| Feature | Description |
|---------|-------------|
| account_age_days | Days from registration_date to the **as-of date**: the transfer's `created_at` in `fraud_features` (point-in-time), the ingest date in `fraud_user_features` (serving). Never `CURRENT_DATE()` in the training query — see review finding #3. |
| is_business | 1 if account_type = Business |
| is_kyc_verified | 1 if kyc_status = Verified |
| country_risk_score | Mapped risk score by country (e.g., higher for less regulated jurisdictions) |

**Transaction velocity features** (computed over sender_id windows) — *→ both tables (latest snapshot in `fraud_user_features`)*:
| Feature | Description |
|---------|-------------|
| tx_count_1h | Transfers sent in the past 1 hour |
| tx_count_24h | Transfers sent in the past 24 hours |
| tx_count_7d | Transfers sent in the past 7 days |
| tx_count_30d | Transfers sent in the past 30 days |

**Amount aggregation features** — *→ both tables, except `amount_ratio_to_avg`*:
| Feature | Description | Table |
|---------|-------------|-------|
| avg_amount_7d | Average source_amount in past 7 days | both (latest in user table) |
| max_amount_7d | Max source_amount in past 7 days | both (latest in user table) |
| total_amount_24h | Sum of source_amount in past 24 hours | both (latest in user table) |
| amount_ratio_to_avg | current_amount / avg_amount_7d (spike detection) | transfer table + computed at request time (needs the payload amount) |

**Cross-border and diversity features**:
| Feature | Description | Table |
|---------|-------------|-------|
| is_cross_border | 1 if source_currency != target_currency | transfer table + computed at request time (transaction-specific) |
| cross_border_ratio_7d | Fraction of cross-border transfers in past 7 days | both (latest in user table) |
| unique_recipients_7d | Distinct recipient_ids in past 7 days | both (latest in user table) |
| unique_currencies_7d | Distinct target_currencies in past 7 days | both (latest in user table) |

**Transaction-level features** (computed at inference time from the payload; per-row in the training SQL) — *not stored in `fraud_user_features`*:
| Feature | Description |
|---------|-------------|
| source_amount | Raw amount from the transfer |
| hour_of_day | Hour extracted from created_at |
| day_of_week | Day of week (0-6) from created_at |

---

## Architecture Diagram

```
                                    TRAINING PATH
                                    ============

  fraud_dataloader.py ──▶ BQ: fraud_users + fraud_transfers
                                    │
                          fraud_ingest.py (feature engineering)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                                ▼
   BQ: fraud_features                   BQ: fraud_user_features
   (transfer-grained, offline           (user-grained, latest per user)
    only — training/batch)                          │
                                          feature_store/sync.py
                                                     │
                                                     ▼
                                          Bigtable (online store)
                                          — keyed by user_id, serves
                                            real-time lookups by sender_id

  KFP Training Pipeline:
  ┌─────────────────────────────────────────────────────────────────┐
  │  load_data ──▶ optuna_lightgbm_tuning ──▶ evaluate ──▶ register │
  │                                                       │        │
  │                                              deploy to Cloud   │
  │                                              Run (FastAPI)     │
  └─────────────────────────────────────────────────────────────────┘


                               INFERENCE PATHS
                               ===============

  BATCH (KFP Pipeline):
  ┌──────────────────────────────────────────────────────────────┐
  │  get_model ──▶ batch_inference (BQ fraud_features,          │
  │                transfer-grained) ──▶ BQ                      │
  └──────────────────────────────────────────────────────────────┘

  REAL-TIME (Direct Cloud Run — NO Dataflow):
  ┌───────────────────────────────────────────────────────────────────────┐
  │  Client ──POST──▶ Cloud Run FastAPI                                  │
  │                      │                                               │
  │                      ├─ Fetch user features from Bigtable            │
  │                      │    (fraud_user_features, by sender_id)        │
  │                      ├─ Compute real-time features from payload      │
  │                      ├─ Run LightGBM prediction                      │
  │                      ├─ Return {fraud_score, is_fraud} synchronously │
  │                      └─ Publish prediction log to PubSub (awaited)   │
  │                              └──▶ BQ: fraud_predictions_streaming    │
  └───────────────────────────────────────────────────────────────────────┘

  STREAMING FEATURE REFRESH (Dataflow) — updates the USER-grained view:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  PubSub (transfer events) ──▶ Beam ──▶ compute velocity features    │
  │                                    ──▶ dual-write:                   │
  │                                        BQ fraud_user_features        │
  │                                        + Bigtable (by user_id)       │
  └──────────────────────────────────────────────────────────────────────┘
  (Streaming writes the user-grained table only. The transfer-grained
   fraud_features is produced solely by the batch ingest.)
```

### Why Direct Cloud Run Instead of Dataflow for Real-Time Inference

The iris pipeline uses **PubSub → Dataflow → Online Store → FastAPI → BQ**, which adds latency from:
- Beam pipeline startup and batching/buffering delays
- Multiple network hops (PubSub → Dataflow workers → Bigtable → FastAPI → BQ)
- Dataflow autoscaling lag

For fraud detection, latency matters (you need to score a transfer before it completes). The direct approach:
- **Client → Cloud Run FastAPI** — single hop, synchronous response
- FastAPI fetches precomputed user features from Bigtable inline (~5-10ms)
- Computes real-time features from the transfer payload itself (no network call)
- Runs LightGBM prediction in-process (~1-5ms)
- Returns response immediately (**target: <50ms p95**)
- Publishes the prediction log to PubSub before returning (audit trail; awaited rather than fire-and-forget — see finding #7)

Dataflow is still used, but only for the **feature refresh path** — keeping velocity/aggregation features in Bigtable up-to-date as new transfers arrive.

---

## Component Inventory: Reuse vs. New

### Reusable Components (no or minimal changes)

| Component | Path | Change Needed |
|-----------|------|---------------|
| FeatureConfig schema | `src/feature_store/schema.py` | None — already generic |
| Feature Store setup | `src/feature_store/setup.py` | Add `"fraud"` and `"fraud_user"` entries to CONFIGS dict |
| Feature Store sync | `src/feature_store/sync.py` | None — already parameterized by config name (sync `fraud_user` only) |
| Online store writer | `src/dataflow/utils/online_store_writer.py` | None — generic |
| Online store reader | `src/dataflow/utils/online_store_reader.py` | None — generic |
| Dead letter handler | `src/dataflow/utils/dead_letter.py` | None — generic |
| Model registry (upload) | `iris_xgboost/pipelines/components/register.py` | Refactor to shared location (see below) |
| Cloud Run deploy | `iris_xgboost/pipelines/components/deploy.py` | Refactor to shared location **+ parameterize scaling/resources/smoke-test** (finding #8) — defaults preserve current iris behaviour |
| Get blessed model | `iris_xgboost/pipelines/components/get_model.py` | Refactor to shared location |
| Schema loader | `iris_xgboost/pipelines/components/schema.py` | Refactor to shared location |
| Logging utility | `src/ml_pipelines_kfp/log.py` | None |
| CI/CD pipeline | `.github/workflows/cicd.yaml` | Extend to build fraud images |
| Observability stack | `observability/` | Add fraud-specific Grafana dashboards |

### Refactoring: Shared KFP Components

Move model-agnostic KFP components out of `iris_xgboost/` into a shared location so both iris and fraud can use them:

```
src/ml_pipelines_kfp/
├── shared/
│   └── components/
│       ├── register.py      # upload_model (moved from iris_xgboost)
│       ├── deploy.py        # deploy_blessed_model_to_fastapi
│       ├── get_model.py     # get blessed model from registry
│       └── schema.py        # load schema artifacts
├── iris_xgboost/            # iris-specific (unchanged)
│   └── pipelines/
│       └── components/
│           ├── data.py
│           ├── models.py
│           └── evaluation.py
└── fraud_lgbm/           # NEW — fraud-specific
    └── ...
```

The iris pipeline imports would change from:
```python
from ml_pipelines_kfp.iris_xgboost.pipelines.components.register import upload_model
```
to:
```python
from ml_pipelines_kfp.shared.components.register import upload_model
```

#### Parameterizing `deploy.py` (finding #8)

The current component hardcodes every serving knob: `min_instance_count: 0`,
`max_instance_count: 10`, `memory 2Gi / cpu 2`, and a smoke test that POSTs an *iris* payload
to `/predict`. Scale-from-zero means a container boot plus a model download from GCS on the
first request — seconds, against a 50ms p95 target. The fraud service also has no `/predict`
endpoint, so the existing smoke test would fail it outright.

Every hardcoded value becomes a parameter **defaulted to today's behaviour**, so the iris
pipeline is unchanged by the refactor:

```python
def deploy_blessed_model_to_fastapi(
    project_id: str,
    location: str,
    model_name: str,
    service_name: str,
    fastapi_image_name: str,
    service_endpoint: Output[Artifact],
    # --- scaling (finding #8) ---
    min_instance_count: int = 0,        # fraud: >= 1
    max_instance_count: int = 10,
    # Default 80 concurrent requests per instance is throughput-oriented. LightGBM
    # predict is CPU-bound, so 80 in flight on 2 vCPU means queueing — which shows
    # up as p95 latency that no amount of min_instances fixes.
    max_instance_request_concurrency: int = 80,   # fraud: 8-16
    startup_cpu_boost: bool = False,              # fraud: True
    cpu: str = "2",
    memory: str = "2Gi",
    cpu_idle: bool = True,              # False == "CPU always allocated" (see #7)
    # --- smoke test (currently hardcoded to iris) ---
    smoke_test_path: str = "/predict",  # fraud: "/score"
    smoke_test_payload: dict = IRIS_SMOKE_PAYLOAD,
): ...
```

Fraud deploys with `min_instance_count=1` in staging and **2** in prod — one warm instance is
one instance, and a revision rollout or a zonal blip takes it out exactly when you are
serving live traffic.

**`min_instances` alone does not make cold starts go away**, and the plan should not pretend
otherwise. Two gaps remain:

1. **Scale-out beyond the floor still cold-starts.** The floor keeps instances warm at
   baseline; a traffic spike past it boots new containers, and those requests eat the full
   startup cost. This is a p99 problem rather than a p95 one. Mitigations: `startup_cpu_boost`
   to shorten boot, and keeping the image lean.
2. **A warm container is not a ready one.** If the model is downloaded from GCS lazily on
   first request, `min_instances` buys nothing — the instance is up and still slow. The fraud
   server must load the model in the FastAPI **lifespan startup hook**, and the deploy must
   set a Cloud Run **startup probe** so traffic isn't routed until the load completes. Without
   the probe, Cloud Run can send the first request into a container that is still unpacking
   the model.

Note `cpu_idle` is the same flag discussed under finding #7 — the plan keeps it `True`
(throttled) and publishes prediction logs inline instead. It is a parameter here so that the
documented fallback in §7 is a one-line change rather than a rewrite.

### New Components to Build

| # | Component | Path | Description |
|---|-----------|------|-------------|
| 1 | Fraud data generator | `scripts/fraud_dataloader.py` | Adapted from `plan/code.py`, writes users + transfers to BQ |
| 2 | Fraud feature config | `src/feature_store/fraud/feature_definitions.py` | FRAUD_CONFIG (transfer-grained, offline) + FRAUD_USER_CONFIG (user-grained, online-served) |
| 3 | Fraud feature ingestion | `src/feature_store/fraud_ingest.py` | Joins users + transfers, computes aggregation features, writes both `fraud_features` (transfer-grained) and `fraud_user_features` (per-user attributes + velocity buckets + `__global__`) |
| 4 | Fraud constants | `src/ml_pipelines_kfp/fraud_lgbm/constants.py` | BQ tables, model name, pipeline name, image names |
| 5 | Fraud instance model | `src/ml_pipelines_kfp/fraud_lgbm/models/instance.py` | Pydantic model for fraud features |
| 6 | Fraud prediction model | `src/ml_pipelines_kfp/fraud_lgbm/models/prediction.py` | fraud_score (float), is_fraud (bool) |
| 7 | Load fraud data (KFP) | `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/data.py` | Load from BQ fraud_features (source='training'), apply the out-of-time + user-disjoint train/test split (see §3c) |
| 8 | Optuna tuning (KFP) | `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/optuna_tuning.py` | LightGBM hyperparameter tuning with Optuna (fits on `split == 'fit'`) |
| 8b | Calibration (KFP) | `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/calibration.py` | Beta calibration on the user-disjoint `calib` split; emits the model+calibrator bundle (§4b, finding #10) |
| 9 | Fraud evaluation (KFP) | `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/evaluation.py` | AUC-PR, precision/recall at thresholds, confusion matrix, post-calibration Brier/ECE |
| 10 | Training pipeline | `src/ml_pipelines_kfp/fraud_lgbm/pipelines/fraud_pipeline_training.py` | Orchestrates load → tune → calibrate → evaluate → register → deploy |
| 11 | Batch inference (KFP) | `src/ml_pipelines_kfp/fraud_lgbm/pipelines/fraud_pipeline_inference.py` | get_model → score fraud_features → write predictions |
| 12 | Fraud FastAPI server | `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/fastapi/fraud_server.py` | Real-time scoring with inline feature fetch |
| 13 | Fraud Dockerfile | `Dockerfile.fraud-fastapi` | Fraud-specific FastAPI image |
| 14 | Fraud PubSub schema | `src/dataflow/models/fraud_schema.py` | Pydantic model for transfer events |
| 15 | Fraud feature pipeline | `src/dataflow/fraud_feature_pipeline.py` | Streaming: PubSub → compute velocity features → dual-write |
| 16 | Fraud training DAGs | `dags/fraud_training_staging_dag.py`, `dags/fraud_training_prod_dag.py` | Airflow DAGs for training |
| 17 | Fraud inference DAGs | `dags/fraud_batch_inference_staging_dag.py`, `dags/fraud_batch_inference_prod_dag.py` | Airflow DAGs for batch inference |
| 18 | Vertex schemas | `src/ml_pipelines_kfp/schemas/fraud_lgbm/vertex/` | instance.yaml, prediction.yaml |

---

## Detailed Design per Component

### 1. Data Generator — `scripts/fraud_dataloader.py`

Adapt `plan/code.py` to:
- Write `fraud_users` and `fraud_transfers` to BQ `ml_dataset`
- Add a `--generate-random` mode that appends unlabeled transfers for batch inference (mirroring `bq_dataloader.py` pattern)
- Add `source` column (`training` vs `batch_input`) and `load_timestamp`

```
python -m scripts.fraud_dataloader                    # Load labeled data
python -m scripts.fraud_dataloader --generate-random 500  # Unlabeled for inference
```

### 2. Feature Config — `src/feature_store/fraud/feature_definitions.py`

**Two configs**, one per feature table (see "Feature Tables: Two Granularities"). The transfer-grained config is the model's full feature list (offline, training/batch); the user-grained config is the subset served online, keyed by `user_id`.

```python
# --- Transfer-grained: offline only (training + batch inference), NOT synced ---
FRAUD_CONFIG = FeatureConfig(
    name="fraud",
    feature_columns=[
        # User-level
        "account_age_days", "is_business", "is_kyc_verified", "country_risk_score",
        # Velocity
        "tx_count_1h", "tx_count_24h", "tx_count_7d", "tx_count_30d",
        # Amount
        "avg_amount_7d", "max_amount_7d", "total_amount_24h", "amount_ratio_to_avg",
        # Cross-border
        "is_cross_border", "cross_border_ratio_7d",
        "unique_recipients_7d", "unique_currencies_7d",
        # Transaction-level
        "source_amount", "hour_of_day", "day_of_week",
    ],
    entity_id_column="entity_id",          # transfer_id + source, one row per transfer
    target_column="label",
    column_mappings={},  # No renaming needed — data generator uses canonical names
    bq_dataset="ml_dataset",
    bq_raw_table="fraud_transfers",
    bq_batch_input_table="fraud_transfers_batch_input",
    bq_feature_table="fraud_features",
    feature_view_id="fraud_features",
    online_serving=False,                  # offline store only — never synced to Bigtable
)

# --- User-grained: bucketed velocity state per user, synced to Bigtable for serving ---
FRAUD_USER_CONFIG = FeatureConfig(
    name="fraud_user",
    feature_columns=[
        # (a) User-level attributes — stored as scalars, used as-is.
        "account_age_days", "is_business", "is_kyc_verified", "country_risk_score",
        # (b) Velocity state as BUCKET COLUMNS, not rolled-up scalars (finding #6).
        #     Ring buffers: hourly slot = epoch_hour % 24, daily slot = epoch_day % 30.
        #     The *_bucket_hour / *_bucket_day arrays carry each slot's own timestamp,
        #     which is what makes a slot self-expiring at read time.
        "hourly_bucket_hour", "hourly_tx_count", "hourly_amount_sum",
        "hourly_amount_max", "hourly_cross_border_count", "hourly_recipients",
        "daily_bucket_day", "daily_tx_count", "daily_amount_sum",
        "daily_amount_max", "daily_cross_border_count", "daily_recipients",
        "daily_currencies",
        # (c) Cold-start defaults — populated ONLY on the __global__ row, which has
        #     empty bucket arrays. Real user rows leave these NULL.
        "global_tx_count_1h", "global_tx_count_24h", "global_tx_count_7d",
        "global_tx_count_30d", "global_avg_amount_7d", "global_max_amount_7d",
        "global_total_amount_24h", "global_cross_border_ratio_7d",
        "global_unique_recipients_7d", "global_unique_currencies_7d",
        # Transaction-level features (source_amount, hour_of_day, day_of_week,
        # is_cross_border, amount_ratio_to_avg) are computed at request time, NOT stored.
        # The model-facing scalars (tx_count_24h, ...) are DERIVED from (b) at read
        # time by velocity.merge_buckets() — they are not stored anywhere online.
    ],
    entity_id_column="user_id",            # one row per user; server looks up by sender_id
    target_column=None,                    # serving view has no label
    column_mappings={},
    bq_dataset="ml_dataset",
    bq_raw_table="fraud_transfers",
    bq_feature_table="fraud_user_features",
    feature_view_id="fraud_user_features",
    online_serving=True,                   # synced to Bigtable by feature_store/sync.py
)
```

> **Note:** `online_serving`/`target_column=None` may require a small `FeatureConfig` schema
> extension (`src/feature_store/schema.py`) if the current schema assumes a label column and
> always-online serving. Flag during Phase 1.

> **Phase 1 verification — array-typed features.** The bucket columns are BQ `ARRAY<INT64>` /
> `ARRAY<FLOAT64>` / `ARRAY<STRING>`. Confirm the Vertex AI Feature Store feature view sync
> and `FetchFeatureValues` round-trip repeated values end-to-end before building on this. If
> arrays don't survive the sync, the fallback is flattening each ring buffer into numbered
> scalar columns (`hourly_tx_count_00 … _23`) — same design, uglier schema, ~130 columns.
> Decide this before writing §8, since it changes the Beam output shape too.

Register **both** in `feature_store/setup.py`:
```python
CONFIGS = {
    "iris": "feature_store.iris.feature_definitions.IRIS_CONFIG",
    "fraud": "feature_store.fraud.feature_definitions.FRAUD_CONFIG",
    "fraud_user": "feature_store.fraud.feature_definitions.FRAUD_USER_CONFIG",
}
```

Only `fraud_user` is synced to the online store (`sync.py fraud_user`); `fraud` is offline only.

### 3. Feature Ingestion — `src/feature_store/fraud_ingest.py`

This is the most complex new component. It produces **both** feature tables from one window-function pass:

1. **`fraud_features` (transfer-grained)** — the CTE below emits one row per transfer with point-in-time features (as of each transfer's `created_at`, per review finding #3). `WRITE_TRUNCATE`.
2. **`fraud_user_features` (user-grained)** — a second write: one row per user holding user-level attributes as of ingest date plus the **velocity bucket ring buffers** (transaction-level columns dropped, velocity scalars deliberately *not* stored — see finding #6), plus the `__global__` cold-start row. `WRITE_TRUNCATE`.

```sql
-- Shared computation: joins users + transfers, computes aggregation features
-- via SQL window functions in BigQuery.

```sql
WITH user_features AS (
    SELECT
        user_id,
        -- registration_date is carried through, NOT turned into an age here:
        -- account_age_days must be computed per-transfer, as of created_at
        -- (review finding #3).
        registration_date,
        IF(account_type = 'Business', 1, 0) AS is_business,
        IF(kyc_status = 'Verified', 1, 0) AS is_kyc_verified,
        CASE country_of_residence
            WHEN 'GB' THEN 0.2 WHEN 'US' THEN 0.3 WHEN 'DE' THEN 0.2
            WHEN 'AU' THEN 0.3 WHEN 'SG' THEN 0.4 WHEN 'BE' THEN 0.2
            WHEN 'FR' THEN 0.2 WHEN 'CA' THEN 0.3
            ELSE 0.5
        END AS country_risk_score,
        label
    FROM `{project}.{dataset}.fraud_users`
),
transfer_windows AS (
    SELECT
        t.transfer_id,
        t.sender_id,
        t.source,
        t.source_amount,
        t.created_at,
        IF(t.source_currency != t.target_currency, 1, 0) AS is_cross_border,
        EXTRACT(HOUR FROM t.created_at) AS hour_of_day,
        EXTRACT(DAYOFWEEK FROM t.created_at) AS day_of_week,
        -- Velocity features (window functions over sender_id)
        COUNT(*) OVER (sender_1h) AS tx_count_1h,
        COUNT(*) OVER (sender_24h) AS tx_count_24h,
        COUNT(*) OVER (sender_7d) AS tx_count_7d,
        COUNT(*) OVER (sender_30d) AS tx_count_30d,
        -- Amount features
        AVG(t.source_amount) OVER (sender_7d) AS avg_amount_7d,
        MAX(t.source_amount) OVER (sender_7d) AS max_amount_7d,
        SUM(t.source_amount) OVER (sender_24h) AS total_amount_24h,
        -- Diversity: BigQuery has no COUNT(DISTINCT) analytic function, so
        -- collect the window's raw values and de-duplicate one level up.
        -- IGNORE NULLS is required — ARRAY_AGG errors on a NULL element.
        ARRAY_AGG(t.recipient_id IGNORE NULLS) OVER (sender_7d)
            AS recipients_7d,
        ARRAY_AGG(t.target_currency IGNORE NULLS) OVER (sender_7d)
            AS currencies_7d,
        -- Computed HERE, not in the outer SELECT: named windows are scoped to
        -- the query that defines them and don't escape the CTE. Uses the raw
        -- expression rather than the is_cross_border alias, which isn't
        -- visible to a window function in its own SELECT list.
        SAFE_DIVIDE(
            SUM(IF(t.source_currency != t.target_currency, 1, 0)) OVER (sender_7d),
            COUNT(*) OVER (sender_7d)
        ) AS cross_border_ratio_7d,
        ...
    FROM `{project}.{dataset}.fraud_transfers` t
    WINDOW
        -- BigQuery RANGE frames take a NUMERIC offset, not an INTERVAL, so the
        -- ORDER BY key is seconds-since-epoch and the bounds are second counts.
        sender_1h AS (PARTITION BY t.sender_id ORDER BY UNIX_SECONDS(t.created_at)
                      RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW),
        sender_24h AS (...86400 PRECEDING...),
        sender_7d AS (...604800 PRECEDING...),
        sender_30d AS (...2592000 PRECEDING...)
),
transfer_features AS (
    -- De-duplicate the collected arrays. Scalar subqueries over UNNEST are
    -- evaluated per row; splitting this out of transfer_windows keeps the
    -- analytic call and the subquery in separate scopes.
    SELECT
        * EXCEPT (recipients_7d, currencies_7d),
        (SELECT COUNT(DISTINCT r) FROM UNNEST(recipients_7d) AS r)
            AS unique_recipients_7d,
        (SELECT COUNT(DISTINCT c) FROM UNNEST(currencies_7d) AS c)
            AS unique_currencies_7d
    FROM transfer_windows
)
SELECT
    CONCAT(t.transfer_id, '_', t.source) AS entity_id,
    t.*,
    -- POINT-IN-TIME: account age as of this transfer, not as of query time.
    -- GREATEST(...,0) guards against a transfer that predates registration
    -- in the synthetic data; NULL registration_date stays NULL for LightGBM.
    GREATEST(DATE_DIFF(DATE(t.created_at), DATE(u.registration_date), DAY), 0)
        AS account_age_days,
    u.is_business, u.is_kyc_verified, u.country_risk_score,
    SAFE_DIVIDE(t.source_amount, t.avg_amount_7d) AS amount_ratio_to_avg,
    u.label,
    CURRENT_TIMESTAMP() AS feature_timestamp
FROM transfer_features t
JOIN user_features u ON t.sender_id = u.user_id
```

The ingestion script executes this query and writes the transfer-grained rows to `fraud_features` with WRITE_TRUNCATE.

**On the `ARRAY_AGG` diversity features.** `ARRAY_AGG(...) OVER (window)` materializes one
array per row containing every value in that row's 7-day window, so a sender with 10k
transfers in a week produces 10k arrays of ~10k elements. At demo scale (thousands of
users, ~2 years of transfers) this is comfortably fine and exact. If a real workload hits
`Resources exceeded` here, the escape hatches in order of preference are: cap the window
with a row bound as well (`ROWS BETWEEN 500 PRECEDING`), or switch to
`HLL_COUNT.MERGE`/sketch-based approximate distinct. Prefer the exact form until it
actually breaks — approximate counts on a low-cardinality field like `target_currency`
are pure downside.

**Point-in-time rule for user-join features.** Every feature on a training row must be
computable from information available at that transfer's `created_at`:

| Feature | As-of handling |
|---------|----------------|
| `account_age_days` | Derived per transfer: `DATE_DIFF(DATE(created_at), DATE(registration_date))`. Never `CURRENT_DATE()` in the training query. |
| `is_business`, `is_kyc_verified`, `country_risk_score` | `fraud_users` stores only the *current* value with no change history, so these are unavoidably as-of-now. Accepted with the caveat below. |
| Velocity / amount / diversity features | Already point-in-time — the `RANGE ... PRECEDING AND CURRENT ROW` windows only look backwards. |

> **Known limitation (accept, don't hide):** `kyc_status`, `account_type`, and
> `country_of_residence` are mutable in reality but are stored as current-state
> only. A user who was `Pending` at transfer time and is `Verified` today leaks a
> future fact into the training row — and it leaks in the *label-correlated*
> direction, since KYC review is often triggered by the fraud itself. Two options,
> to decide in Phase 1:
> 1. Add `kyc_verified_at` / `account_type_changed_at` / `country_changed_at` to
>    the generator in `code.py` and resolve each attribute as of `created_at`
>    (`IF(t.created_at >= u.kyc_verified_at, 1, 0)`). Preferred — it makes the
>    demo honest and costs a few lines in the generator.
> 2. Ship as-is and record it as a known skew source in the model card.
>
> This is exactly the class of bug that shows up as "great offline AUC-PR, mediocre
> production precision," so it is worth the generator change.

**Then it derives the user-grained serving table.** This table does **not** store rolled-up
velocity scalars — storing `tx_count_24h` as a number is what made the online store go stale
(finding #6). It stores the **hourly/daily buckets** those scalars are summed from, and the
serving path rolls them up as of request time.

```sql
-- fraud_user_features: one row per user holding
--   (a) user-level attributes, recomputed as of ingest date
--   (b) velocity BUCKET ring buffers (hourly x24, daily x30)
--   (c) the __global__ cold-start row (population averages, empty buckets)
CREATE OR REPLACE TABLE `{project}.{dataset}.fraud_user_features` AS
WITH hourly AS (
    -- Ring slot = epoch_hour % 24. Each slot carries its own epoch hour, which is
    -- what lets the reader tell a fresh bucket from one that is 24h stale.
    SELECT
        sender_id AS user_id,
        DIV(UNIX_SECONDS(TIMESTAMP_TRUNC(created_at, HOUR)), 3600) AS bucket_hour,
        COUNT(*) AS tx_count,
        SUM(source_amount) AS amount_sum,
        MAX(source_amount) AS amount_max,
        COUNTIF(source_currency != target_currency) AS cross_border_count,
        -- Distinct counts are NOT additive across buckets (see note below), so the
        -- member IDs travel with the bucket and are unioned at read time. Capped.
        ARRAY_AGG(DISTINCT recipient_id IGNORE NULLS LIMIT 64) AS recipients
    FROM `{project}.{dataset}.fraud_transfers`
    WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
    GROUP BY user_id, bucket_hour
),
daily AS (
    -- Ring slot = epoch_day % 30. Serves the 7d and 30d windows.
    SELECT
        sender_id AS user_id,
        DIV(UNIX_SECONDS(TIMESTAMP_TRUNC(created_at, DAY)), 86400) AS bucket_day,
        COUNT(*) AS tx_count,
        SUM(source_amount) AS amount_sum,
        MAX(source_amount) AS amount_max,
        COUNTIF(source_currency != target_currency) AS cross_border_count,
        ARRAY_AGG(DISTINCT recipient_id IGNORE NULLS LIMIT 64) AS recipients,
        ARRAY_AGG(DISTINCT target_currency IGNORE NULLS) AS currencies
    FROM `{project}.{dataset}.fraud_transfers`
    WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
    GROUP BY user_id, bucket_day
),
-- Pivot each user's buckets into fixed-length ring arrays indexed by slot, so that
-- position N always means "epoch hour H where H % 24 = N". Missing slots are 0 with
-- bucket_hour = -1 (never written). Implemented with a 0..23 / 0..29 slot spine
-- LEFT JOINed to the buckets above and ARRAY_AGG(... ORDER BY slot).
user_buckets AS ( ... ),
attrs AS (
    SELECT
        u.user_id,
        -- account_age_days is RECOMPUTED as of ingest time, not copied from the
        -- last transfer's row. The serving contract is "age as of scoring time";
        -- carrying a training-row value forward would freeze the user's age at
        -- their last transfer and drift further from truth every day they are
        -- idle. Same definition as training, different as-of date — that is what
        -- makes the two consistent, not identical SQL.
        GREATEST(DATE_DIFF(CURRENT_DATE(), DATE(u.registration_date), DAY), 0)
            AS account_age_days,
        IF(u.account_type = 'Business', 1, 0) AS is_business,
        IF(u.kyc_status = 'Verified', 1, 0) AS is_kyc_verified,
        ... AS country_risk_score
    FROM `{project}.{dataset}.fraud_users` u
)
SELECT a.*, b.* EXCEPT (user_id), CURRENT_TIMESTAMP() AS feature_timestamp
FROM attrs a LEFT JOIN user_buckets b USING (user_id)
UNION ALL
-- __global__ cold-start row: population averages of the DERIVED scalars, empty
-- bucket arrays. Only this row populates the global_* columns.
SELECT '__global__', AVG(account_age_days), ..., CURRENT_TIMESTAMP()
FROM `{project}.{dataset}.fraud_features` WHERE source = 'training';
```

> **Distinct counts are not additive.** `tx_count`, `amount_sum`, and `cross_border_count`
> roll up across buckets by summing, and `amount_max` by taking a max — but
> `unique_recipients_7d` cannot be recovered from per-bucket distinct *counts*, because a
> recipient appearing on three days would be counted three times. So the bucket carries the
> member IDs and the reader unions them. The `LIMIT 64` cap bounds row size; it undercounts
> only for users fanning out to >64 recipients in one bucket, who are already saturating
> `tx_count`. At real scale the principled version is per-bucket HLL++ sketches, which *are*
> mergeable — deferred here because BigQuery's sketches are zetasketch-format and have no
> good Python reader, and the read-time merge happens in the FastAPI process.

`sync.py fraud_user` then pushes `fraud_user_features` (including `__global__`) to Bigtable. `fraud_features` is never synced.

Note the join is on `fraud_users`, not on who transacted recently: **every** user gets a row
each ingest, so `account_age_days` is at most one day stale and idle users get their buckets
rewritten (i.e. emptied) rather than left behind holding old counts.

### 3b. Read-Time Bucket Merge — `src/feature_store/fraud/velocity.py`

Bucket columns only solve staleness if the rollup happens **at read time, as of the request**.
This module is the single implementation of that rollup, imported by both the FastAPI server
(§7) and the Beam pipeline (§8). Two implementations of this logic would be two chances to
disagree with the training SQL — which is training/serving skew in the same family as
finding #3, and just as invisible.

```python
# velocity.py — the ONLY place window rollup logic lives (besides the training SQL,
# which is pinned to it by the parity test below).

VELOCITY_SCALARS = (
    "tx_count_1h", "tx_count_24h", "tx_count_7d", "tx_count_30d",
    "avg_amount_7d", "max_amount_7d", "total_amount_24h",
    "cross_border_ratio_7d", "unique_recipients_7d", "unique_currencies_7d",
)

def merge_buckets(row: dict, as_of: datetime) -> dict:
    """Roll bucket ring buffers up into the model-facing velocity scalars.

    Decay is structural: a slot is included only if the epoch hour/day stamped on
    it falls inside the window measured back from `as_of`. A user who stopped
    transacting yesterday has no in-range slots, so their counts are 0 — with no
    writer having had to do anything. That is the whole point of the design.
    """
    now_hour = int(as_of.timestamp()) // 3600
    now_day = int(as_of.timestamp()) // 86400

    def hourly_slots(window_hours: int) -> list[int]:
        # Slot i is live iff its stamped hour is within the window AND not a stale
        # value left in the ring from a previous lap (bucket_hour <= now_hour).
        return [
            i for i, h in enumerate(row["hourly_bucket_hour"])
            if h >= 0 and now_hour - window_hours < h <= now_hour
        ]

    def daily_slots(window_days: int) -> list[int]: ...

    h1, h24 = hourly_slots(1), hourly_slots(24)
    d7, d30 = daily_slots(7), daily_slots(30)

    tx_7d = sum(row["daily_tx_count"][i] for i in d7)
    amt_7d = sum(row["daily_amount_sum"][i] for i in d7)
    cb_7d = sum(row["daily_cross_border_count"][i] for i in d7)

    return {
        "tx_count_1h": sum(row["hourly_tx_count"][i] for i in h1),
        "tx_count_24h": sum(row["hourly_tx_count"][i] for i in h24),
        "tx_count_7d": tx_7d,
        "tx_count_30d": sum(row["daily_tx_count"][i] for i in d30),
        "total_amount_24h": sum(row["hourly_amount_sum"][i] for i in h24),
        # avg is sum/count over the SAME slot set — never an average of averages
        "avg_amount_7d": (amt_7d / tx_7d) if tx_7d else None,
        "max_amount_7d": max((row["daily_amount_max"][i] for i in d7), default=None),
        "cross_border_ratio_7d": (cb_7d / tx_7d) if tx_7d else None,
        # Distinct counts UNION member IDs across buckets — they cannot be summed
        "unique_recipients_7d": len({r for i in d7 for r in row["daily_recipients"][i]}),
        "unique_currencies_7d": len({c for i in d7 for c in row["daily_currencies"][i]}),
    }
```

**Parity test (required, `tests/test_velocity_parity.py`).** Run the §3 training SQL and
`merge_buckets()` over the same fixture and assert every scalar matches. This is the test
that catches boundary disagreements — the training SQL's `RANGE ... AND CURRENT ROW` is
inclusive of the current row, and a `<` vs `<=` slip in `hourly_slots` reproduces exactly the
kind of skew that never shows up in offline metrics. Pin it in CI; it is cheap and it is the
only thing standing between three copies of this logic and silent divergence.

### 3c. Train/Test Split Strategy — `pipelines/components/data.py`

Fraud labels live on the **user**, and each user has many transfers. A random `train_test_split` (or a plain `StratifiedKFold`) puts the same `sender_id`'s transfers in both train and validation, letting the model memorize user identity via features like `account_age_days` / `country_risk_score` and producing inflated scores. Fraud is also inherently **temporal** — in production you always score *future* transfers with a model trained on the *past*.

So the split is **out-of-time AND user-disjoint** (decided over plain `StratifiedGroupKFold`, which would fix user leakage but not look-ahead leakage):

1. Pick a time cutoff `T` = 80th percentile of `created_at` over the training rows (last ~20% of the timeline is the test period). A fixed calendar date works too.
2. `test_users` = every `sender_id` with a transfer at/after `T`.
3. **Train** = transfers before `T` from users **not** in `test_users`.
   **Test** = transfers at/after `T`.
4. Inside train, hold out a **calibration** set (§4b) by **user, not by time**: randomly assign
   ~20% of train users to `calib`, and take *all* their pre-`T` rows. Result:
   **fit** and **calib** are user-disjoint and span the **same time range**; **test** is later
   in time and disjoint from both.

> **User-disjoint, time-overlapping — deliberately different from the train/test split.**
> "Random" here means random over *users*, never over rows: a row-level random split would put
> the same sender in both fit and calib, and the calibrator would be tuned on users the model
> already memorized, reporting confidence production won't reproduce.
>
> Unlike train/test, calib is **not** pushed into a later time window. Calibration corrects
> score *magnitude*, and holding the time span fixed keeps that job clean — a later slice
> would fold temporal drift into the calibration fit, so the calibrator would be silently
> correcting for two different things at once and you could not tell which. Keeping fit's full
> time span also avoids surrendering the most recent 20% of training history.
>
> What this gives up is any out-of-time check on the calibrator itself, so §5 measures Brier
> and ECE on the **test** split as well. A material gap between calib and test calibration
> quality is the signal that score distributions are drifting and recalibration needs to be
> scheduled, not a one-off.

```sql
-- data.py issues this against fraud_features (source = 'training') and labels each row.
-- test  : created_at >= T          (later in time, user-disjoint from everything)
-- calib : created_at <  T, user in calib_users   (same time span as fit)
-- fit   : created_at <  T, user not in calib_users
WITH cutoff AS (
    SELECT PERCENTILE_CONT(created_at, 0.8) OVER () AS t
    FROM `{project}.{dataset}.fraud_features`
    WHERE source = 'training' LIMIT 1
),
test_users AS (
    SELECT DISTINCT f.sender_id
    FROM `{project}.{dataset}.fraud_features` f, cutoff
    WHERE f.source = 'training' AND f.created_at >= cutoff.t
),
train_users AS (            -- users who survive the test-period exclusion
    SELECT DISTINCT f.sender_id
    FROM `{project}.{dataset}.fraud_features` f, cutoff
    WHERE f.source = 'training'
      AND f.created_at < cutoff.t
      AND f.sender_id NOT IN (SELECT sender_id FROM test_users)
),
calib_users AS (
    -- Random over USERS, deterministically seeded so the split is reproducible
    -- across pipeline runs. Never random over rows.
    SELECT sender_id FROM train_users
    WHERE MOD(ABS(FARM_FINGERPRINT(CONCAT(sender_id, '{split_seed}'))), 100) < 20
)
SELECT f.*,
       CASE
           WHEN f.created_at >= (SELECT t FROM cutoff) THEN 'test'
           WHEN f.sender_id IN (SELECT sender_id FROM calib_users) THEN 'calib'
           ELSE 'fit'
       END AS split
FROM `{project}.{dataset}.fraud_features` f
WHERE f.source = 'training'
  -- drop pre-T rows belonging to any test-period user → no user spans both sides
  AND NOT (f.created_at < (SELECT t FROM cutoff)
           AND f.sender_id IN (SELECT sender_id FROM test_users))
```

Guarantees:
- **No user overlap** — every `sender_id` in test is fully excluded from train (their pre-`T` rows are dropped), and fit/calib are disjoint by construction on `sender_id`.
- **No look-ahead into test** — fit and calib are strictly before `T`, test at/after; this also reinforces point-in-time correctness (finding #3).
- **Fit and calib share a time span on purpose** — see the note above; the temporal check on calibration lives in §5's test-split Brier/ECE instead.

> **Watch the positive count in `calib`.** Splitting 20% of users out of a small dataset can
> leave the calibration holdout with a handful of frauds, and a calibrator fit on ~10 positives
> is noise. Log the positive count per split and fail loudly below a floor (~100 positives).
> The same holdout also selects the decision threshold (§5), which makes a thin `calib` doubly
> expensive. A concrete reason to scale the generator up as part of finding #5.

Trade-off: users active across the boundary lose their pre-`T` history from train. With 2 years of data and a 0.8 cutoff that's acceptable; if train shrinks too much, assign **whole users** to train/test by user-recency (e.g. newest 20% of users by first `created_at` → test) — still user-disjoint, with slightly weaker temporal separation.

**Cross-validation inside Optuna must match the holdout** — use out-of-time folds (`TimeSeriesSplit` on the time-sorted training set; each fold validates on a *later* slice than it trains on). `StratifiedGroupKFold(groups=sender_id)` is the alternative when strict per-fold group purity matters more than temporal ordering — no single sklearn splitter does both at once.

Identifier/time columns (`sender_id`, `recipient_id`, `transfer_id`, `entity_id`, `created_at`, `feature_timestamp`, `source`, `split`) must be **dropped from the model matrix** in both tuning and evaluation, or the model can still memorize identity.

### 4. Optuna Hyperparameter Tuning — KFP Component

```python
@component(base_image=_constants.IMAGE_NAME)
def optuna_lightgbm_tuning(
    train_dataset: Input[Dataset],
    n_trials: int,
    metrics: Output[Metrics],
    output_model: Output[Model],
    best_params: Output[Artifact],
):
    """Run Optuna study to find best LightGBM hyperparameters for fraud detection."""

    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import average_precision_score
    import pandas as pd, joblib, json

    # Out-of-time CV needs rows in time order (see "Train/Test Split Strategy").
    # Rows where split == 'fit' only — 'calib' is reserved for §4b and must stay
    # unseen by both tuning and the final fit, or the calibrator is fit on data the
    # model already memorized.
    train = pd.read_csv(train_dataset.path).sort_values("created_at").reset_index(drop=True)
    y = train["label"]
    # Drop label + identifier/time columns so the model can't memorize user/transfer identity.
    ID_COLS = ["label", "sender_id", "recipient_id", "transfer_id", "entity_id",
               "created_at", "feature_timestamp", "source", "split"]
    X = train.drop(columns=ID_COLS, errors="ignore")

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 15),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 5),
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
        }

        # Out-of-time folds: each fold trains on earlier rows, validates on later ones,
        # mirroring the user-disjoint out-of-time holdout. (Swap for
        # StratifiedGroupKFold(groups=train["sender_id"]) if strict per-fold group
        # purity is preferred over temporal ordering.)
        cv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in cv.split(X):
            model = lgb.LGBMClassifier(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(average_precision_score(y.iloc[val_idx], preds))
        return sum(scores) / len(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Train final model with best params on full training data
    best = study.best_params
    best["objective"] = "binary"
    best["metric"] = "average_precision"
    best["verbosity"] = -1
    final_model = lgb.LGBMClassifier(**best)
    final_model.fit(X, y)

    # Log metrics
    metrics.log_metric("best_aucpr_cv", study.best_value)
    for k, v in study.best_params.items():
        metrics.log_metric(f"best_{k}", v)

    # Save model and params
    joblib.dump(final_model, output_model.path)
    with open(best_params.path, "w") as f:
        json.dump(study.best_params, f)
```

### 4b. Probability Calibration — KFP Component (Beta calibration)

Optuna searches `scale_pos_weight` in [1, 15], which reweights the positive class and pushes
predicted scores away from true posterior probabilities (finding #10). The scores still *rank*
correctly, so AUC-PR is unaffected — but their **magnitude** is meaningless, and every
downstream consumer of magnitude is therefore wrong: the 0.5 threshold in the config table,
any "risk score" shown to an analyst, and any expected-loss calculation.

**Beta calibration** (Kull et al., 2017) is a good fit here and better than the obvious
alternatives: Platt scaling assumes a sigmoid distortion and cannot map 0→0 and 1→1, which is
exactly wrong for scores already in [0,1] that have been squashed by class reweighting;
isotonic regression is non-parametric and overfits badly with few positives, which fraud
always has. Beta has three parameters and fits fine on a small holdout.

```python
@component(base_image=_constants.IMAGE_NAME)
def calibrate_fraud_model(
    calib_dataset: Input[Dataset],     # split == 'calib' — user-disjoint, out-of-time (§3c)
    tuned_model: Input[Model],
    calibrated_model: Output[Model],
    metrics: Output[Metrics],
):
    """Fit Beta calibration on a held-out slice and bundle it with the model."""

    import numpy as np, pandas as pd, joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score

    calib = pd.read_csv(calib_dataset.path)
    y = calib["label"].values
    X = calib.drop(columns=ID_COLS, errors="ignore")     # same ID_COLS as tuning
    s = joblib.load(tuned_model.path).predict_proba(X)[:, 1]
    s = np.clip(s, 1e-6, 1 - 1e-6)                       # log(0) guard

    # Beta calibration is logistic regression on [log(s), -log(1-s)]:
    #     p = 1 / (1 + 1 / (exp(c) * s**a / (1-s)**b))
    Z = np.column_stack([np.log(s), -np.log(1 - s)])
    lr = LogisticRegression().fit(Z, y)
    a, b = lr.coef_[0]
    c = lr.intercept_[0]

    # a, b MUST be >= 0 or the map is non-monotonic — it would reorder scores and
    # silently change AUC-PR, turning a calibration step into a ranking change.
    # The standard remedy is refitting a restricted 2-parameter variant.
    if a < 0:
        lr = LogisticRegression().fit(Z[:, [1]], y); a, b = 0.0, lr.coef_[0][0]
    elif b < 0:
        lr = LogisticRegression().fit(Z[:, [0]], y); a, b = lr.coef_[0][0], 0.0

    calibrator = BetaCalibrator(a=a, b=b, c=c)

    # Bundle model + calibrator as ONE artifact. Shipping them separately invites a
    # serving path that loads the model and forgets the calibrator — which fails
    # silently, because uncalibrated scores look perfectly reasonable.
    joblib.dump(CalibratedFraudModel(model, calibrator), calibrated_model.path)

    p = calibrator.transform(s)
    metrics.log_metric("brier_before", brier_score_loss(y, s))
    metrics.log_metric("brier_after", brier_score_loss(y, p))
    metrics.log_metric("ece_before", expected_calibration_error(y, s))
    metrics.log_metric("ece_after", expected_calibration_error(y, p))
    metrics.log_metric("calib_positives", int(y.sum()))
    # Guard: a monotonic calibrator cannot change ranking. If it did, a/b went
    # negative or the clip is wrong — fail rather than ship a reordered model.
    assert abs(roc_auc_score(y, p) - roc_auc_score(y, s)) < 1e-6
```

Two constraints that are easy to get wrong:

- **Do not resample or rebalance the calibration set.** The calibrator's whole job is to map
  reweighted scores back onto the true class prior, so the holdout has to carry production
  prevalence. Applying `scale_pos_weight`-style balancing here would re-break exactly what is
  being fixed.
- **Calibration is not threshold selection.** They are complementary, not the either/or
  finding #10 presents. Calibration makes 0.5 *meaningful* — "expected cost of a FP equals
  that of a FN" — but that is almost never the right operating point for fraud, where blocking
  a legitimate transfer and missing a fraudulent one have very different costs. The threshold
  is chosen separately from the PR curve at a target precision, in §5.

### 5. Fraud Evaluation — KFP Component

Unlike iris (accuracy-only), fraud needs imbalanced-class metrics:

```python
@component(base_image=_constants.IMAGE_NAME)
def evaluate_fraud_model(
    test_dataset: Input[Dataset],
    calib_dataset: Input[Dataset],    # threshold is SELECTED here, not on test
    calibrated_model: Input[Model],   # model + calibrator bundle from §4b
    target_precision: float,          # the operating point, e.g. 0.90
    min_predicted_positives: int,     # support floor for threshold selection
    metrics: Output[Metrics],
    best_model: Output[Model],
):
    """Evaluate the CALIBRATED model and pick the decision threshold from the PR curve."""

    import numpy as np
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve,
        precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, brier_score_loss,
    )

    # --- Threshold selection, on CALIB (not test) -----------------------------
    # Selecting the threshold on test and then reporting precision at it on the
    # same test rows is fitting a parameter to the holdout and grading yourself
    # on it. calib is already held out from the model fit, so one more parameter
    # estimated there is cheap; test stays clean as the verification set.
    p_cal = bundle.predict_proba(X_calib)[:, 1]
    prec, rec, thr = precision_recall_curve(y_calib, p_cal)
    prec, rec = prec[:-1], rec[:-1]          # align with thr

    # Maximize recall SUBJECT TO precision >= target. Precision is not monotonic
    # in threshold, so "first index clearing the target" would happily latch onto
    # a jagged spike in the sparse high-threshold tail.
    support = np.array([(p_cal >= t).sum() for t in thr])
    ok = (prec >= target_precision) & (support >= min_predicted_positives)
    if not ok.any():
        # Unachievable target must fail loudly. Silently falling back to 0.5 (or
        # to 1.0, blocking nothing) ships a model that does not do what the
        # config table claims it does.
        raise RuntimeError(
            f"No threshold reaches precision {target_precision} with >= "
            f"{min_predicted_positives} predicted positives on calib."
        )
    chosen_threshold = float(thr[ok][np.argmax(rec[ok])])

    # --- Evaluate on TEST at the chosen threshold ----------------------------
    # Load the out-of-time, user-disjoint TEST split (rows where split == 'test').
    # Drop the SAME identifier/time columns as tuning (ID_COLS) before predicting —
    # the held-out users never appeared in training, so this is a leakage-free estimate.
    # Log: auc_pr, auc_roc, precision_at_threshold, recall_at_threshold,
    #       f1_at_threshold, confusion_matrix counts, chosen_threshold
    # Log calibration on the TEST split too — §4b's numbers are in-sample for the
    # calibrator, so brier/ECE here are the ones that count: brier, ece,
    # reliability curve bins. A large calib-vs-test gap means score drift and is
    # the trigger for scheduled recalibration (§3c note).
    # Gate: fail pipeline if auc_pr < minimum threshold
    # Gate: fail if precision_on_test < target_precision - tolerance. The threshold
    #       was picked in-period; this is the check that it survives out-of-time.
    #       Without it, "precision >= 0.90" is a claim about calib, not production.
    # Save blessed model (the bundle — never the bare LightGBM model)
```

**Writing the threshold to `fraud_config`.** Only on a blessed model — a run that fails the
gate must not overwrite the live threshold. The row is keyed by `model_version`, not a single
mutable "current threshold":

```sql
INSERT INTO `{project}.ml_dataset.fraud_config`
  (model_version, threshold, target_precision, precision_on_test, recall_on_test,
   calib_positives, selected_at, is_active)
VALUES (@model_version, @chosen_threshold, @target_precision, ...)
```

Versioning it matters because a model rollback that leaves the previous model's threshold in
place silently changes the operating point — the threshold is a property *of a model*, not of
the service. The FastAPI server reads the active row on startup (and via the admin refresh
endpoint) and must load the threshold **belonging to the model version it has loaded**.

### 6. Fraud Training Pipeline — KFP Orchestration

```
load_data_from_feature_store
        │
        ▼
optuna_lightgbm_tuning (n_trials=50)        # fits on split == 'fit'
        │
        ▼
calibrate_fraud_model (Beta calibration)    # fits on split == 'calib' (§4b)
        │
        ▼
evaluate_fraud_model (target_precision=0.90)
    ├─ selects threshold from the PR curve on 'calib'
    ├─ evaluates on 'test' at that threshold
    └─ writes threshold to fraud_config keyed by model_version
        │
        ▼
upload_model (reused shared component)
        │
        ▼
deploy_blessed_model_to_fastapi (shared component, fraud serving params)
    min_instance_count=2 (prod) / 1 (staging)   # no scale-from-zero (#8)
    max_instance_request_concurrency=8          # CPU-bound predict, avoid queueing
    startup_cpu_boost=True
    smoke_test_path="/score"                    # fraud has no /predict endpoint
```

### 7. Real-Time Fraud Scoring — FastAPI Server

This is the key architectural difference from iris. Instead of PubSub → Dataflow → FastAPI, the fraud model uses a **direct synchronous API**:

```python
# fraud_server.py — runs on Cloud Run

@app.post("/score", response_model=FraudScoreResponse)
async def score_transfer(request: TransferScoreRequest):
    """Score a single transfer for fraud risk.

    1. Fetch the user's bucket row from Bigtable
    2. Roll buckets up as of NOW (velocity.merge_buckets) — this is where decay happens
    3. Compute real-time features from the transfer payload
    4. Run LightGBM prediction
    5. Return fraud score synchronously
    6. Publish prediction log to PubSub *before returning* (finding #7)
    """
    as_of = datetime.now(timezone.utc)

    # Step 1: Fetch the per-user row from the online store. The online store is the
    # fraud_user_features view (keyed by user_id); we look it up by sender_id.
    row = await fetch_from_online_store("fraud_user", request.sender_id)

    # Step 2: Resolve velocity features. THREE distinct states, not two — "idle"
    # is information about a user, not missing information about them (finding #6).
    if row is None:
        # (a) No history at all -> population averages for everything.
        row = await fetch_from_online_store("fraud_user", "__global__")
        user_features = {**global_attrs(row), **global_velocity(row)}
    elif is_stale(row["feature_timestamp"], as_of, MAX_FEATURE_AGE):
        # (b) Row exists but the streaming pipeline has evidently stopped writing.
        # Trust the user attributes, distrust the buckets. Belt-and-braces: with
        # read-time decay a stale row already merges to ~zeros, but this makes the
        # "pipeline is silently dead" failure mode explicit and alertable rather
        # than quietly scoring everyone as a dormant user.
        user_features = {**user_attrs(row), **ZERO_VELOCITY}
        STALE_FEATURE_COUNTER.inc()
    else:
        # (c) Normal path — roll the ring buffers up as of NOW. An idle user's
        # slots simply fall outside the window and merge to zeros; no writer had
        # to touch anything for that to be correct.
        user_features = {**user_attrs(row), **merge_buckets(row, as_of)}

    # Step 3: Compute real-time features from the payload
    realtime_features = compute_realtime_features(request)
    # hour_of_day, day_of_week, is_cross_border, source_amount,
    # amount_ratio_to_avg = source_amount / user_features["avg_amount_7d"]

    # Step 4: Merge and predict
    feature_vector = {**user_features, **realtime_features}
    df = pd.DataFrame([feature_vector])
    # CalibratedFraudModel bundle (§4b): predict_proba applies the Beta calibrator,
    # so fraud_prob is an actual probability and THRESHOLD means something. Loading
    # the bare LightGBM model here would still return plausible-looking scores —
    # which is why the two ship as one artifact.
    fraud_prob = model.predict_proba(df)[0][1]

    # Step 5: Build the response
    response = FraudScoreResponse(
        transfer_id=request.transfer_id,
        fraud_score=fraud_prob,
        is_fraud=fraud_prob >= THRESHOLD,
        model_version=MODEL_VERSION,
    )

    # Step 6: Publish the prediction log BEFORE returning (finding #7).
    # Not asyncio.create_task(): Cloud Run throttles CPU to ~0 once the response
    # is written, so a fire-and-forget task — and equally the Pub/Sub client's
    # background batch-commit thread — stalls silently. "Silently" is the
    # operative word: monitoring would show 100% success while the audit trail
    # quietly went missing.
    await publish_prediction_log(request, response, feature_vector, as_of)

    return response
```

**The publish path** — `publish()` alone is not enough; it only *enqueues* into a batch that a
background thread commits, which is the same throttled-CPU trap one layer down. The future has
to be resolved before the handler returns:

```python
# Batch settings are tuned for a latency budget, not for throughput. max_latency is
# the dominant term in what this adds to p95 — a batch that waits 100ms to fill would
# blow the entire request budget on logging.
_publisher = pubsub_v1.PublisherClient(
    batch_settings=pubsub_v1.types.BatchSettings(
        max_messages=100,
        max_bytes=1024 * 1024,
        max_latency=0.005,        # 5ms — under concurrent load batches still fill
    )
)

async def publish_prediction_log(request, response, feature_vector, as_of) -> None:
    payload = {
        "transfer_id": request.transfer_id,      # idempotency key for the BQ sink
        "sender_id": request.sender_id,
        "scored_at": as_of.isoformat(),
        "fraud_score": response.fraud_score,
        "is_fraud": response.is_fraud,
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        # The exact vector scored, including the merged velocity features. This is
        # what makes training/serving skew detectable later — without it you can
        # only compare distributions, never a specific decision.
        "features": feature_vector,
    }
    future = _publisher.publish(TOPIC, json.dumps(payload).encode())
    loop = asyncio.get_running_loop()
    try:
        # future.result() blocks; keep it off the event loop (same reason as the
        # blocking online-store fetch noted under Minor nits).
        await asyncio.wait_for(
            loop.run_in_executor(None, future.result), timeout=PUBLISH_TIMEOUT_S
        )
    except Exception:
        # Logging must never fail scoring. A Pub/Sub outage degrades the audit
        # trail; it does not take fraud decisioning offline. Fall back to a
        # structured stdout line (captured by Cloud Logging without needing
        # post-response CPU) so the record survives in a queryable form.
        PUBLISH_FAILURE_COUNTER.inc()
        logger.warning("prediction_log_publish_failed", extra={"payload": payload})
```

**Why not "CPU always allocated"** (the other option in finding #7): it works, but it makes
correctness depend on a deploy-time flag rather than on code. Someone edits the Cloud Run
service in Terraform months from now, the flag flips, and predictions stop being logged with
no error anywhere — the failure is invisible precisely because fire-and-forget swallows it.
It also bills CPU for idle instance time, and with `min_instances >= 1` (finding #8) those
instances are always up. Publishing before returning puts the guarantee in the request path
where a timeout and a counter can see it.

**Cost:** one Pub/Sub round trip on the critical path, ~5-15ms including the 5ms batch window.
That fits the 20-50ms p95 target but is not free — it is the single largest addition to the
budget from this change. If it proves too expensive, the next move is a bounded in-process
queue drained by a background task **with `min_instances >= 1` and CPU always allocated** —
i.e. buying the async back explicitly, rather than assuming it works by default.

**Request/Response models:**

```python
class TransferScoreRequest(BaseModel):
    transfer_id: str
    sender_id: str
    recipient_id: str
    source_currency: str
    target_currency: str
    source_amount: float
    target_amount: float
    created_at: str  # ISO 8601

class FraudScoreResponse(BaseModel):
    transfer_id: str
    fraud_score: float        # probability [0, 1]
    is_fraud: bool             # fraud_score >= threshold
    model_version: str
    processing_time_ms: float
```

**Why this is faster:**
- No PubSub → Dataflow hop (saves ~100-500ms batching latency)
- Feature fetch from Bigtable is inline (~5-10ms)
- LightGBM predict is in-process (~1-5ms)
- Prediction-log publish is awaited inline (~5-15ms) — the one deliberate cost, see finding #7
- Total expected latency: **20-50ms p95** vs **500ms-2s** with Dataflow

### 8. Streaming Feature Refresh — Dataflow Pipeline (Time-Bucketed)

Dataflow is still used, but only for **keeping the user-grained serving view up-to-date** — not for real-time scoring, and it writes **only `fraud_user_features` + Bigtable** (never the transfer-grained `fraud_features`, which is produced solely by the batch ingest). Features are aggregated using **hourly fixed windows** (time-bucketed), meaning Bigtable is updated once per user per hour rather than on every transfer. This trades up-to-1-hour staleness for ~10-50x fewer Bigtable writes at scale.

The pipeline writes **buckets, never rolled-up scalars** (finding #6). It emits the whole ring
buffer for a user each time that user's bucket closes; it never has to write anything for an
idle user, because expiry is the reader's job.

`src/dataflow/fraud_feature_pipeline.py`:
```
PubSub (transfer events)
    │
    ├─ Parse and validate (PubSubFraudMessage)
    │
    ├─ Fixed window (1 hour) keyed by sender_id
    │
    ├─ Aggregate within window — ONE bucket, not a rolling window:
    │     tx_count, amount sum/max, cross-border count,
    │     capped distinct recipient/currency ID sets
    │
    ├─ Stateful DoFn per sender_id: place the closed bucket into its ring slot
    │     hourly slot = epoch_hour % 24, daily slot = epoch_day % 30
    │     stamp the slot with its own epoch hour/day
    │     (no shifting, no read-modify-write of neighbours — a bucket only ever
    │      overwrites its own slot, and a slot from a previous lap is ignored by
    │      the reader because its stamp is out of range)
    │
    ├─ Write to BQ fraud_user_features (user-grained view) — WRITE_APPEND
    │
    └─ Write to Bigtable (online store, keyed by user_id) — one write per user per hour
```

**No timers are needed.** The obvious alternative fix for staleness was a timer-based flush in
the stateful DoFn that zeroes out idle users. That was rejected: it costs a Bigtable write per
idle user per interval — reintroducing exactly the write amplification time-bucketing was
adopted to avoid, and paying it for the *least* active users — and it still leaves a staleness
window between fires. Read-time decay has neither problem, and it stays correct when the
pipeline is down.

The FastAPI service reads the bucket row by `sender_id` and calls `velocity.merge_buckets(row,
as_of=now)` (§3b). Residual staleness is bounded by the **bucket granularity** (≤1 hour of
in-flight transfers not yet closed into a bucket) rather than by how long ago the user last
transacted — which was the actual bug. Beam and FastAPI import the same merge function.

**Batch reconciliation is still required.** The nightly `fraud_ingest.py` run rewrites every
user's buckets from BQ. This is not a staleness fix — read-time decay already handles that —
it is *reconciliation*: Beam drops and duplicates on worker restarts, and a bad deploy can
write garbage slots. The batch job re-establishes a known-good baseline from the source of
truth. The two mechanisms address different failure modes and the plan keeps both.

---

## File Structure (New + Modified Files)

```
# NEW FILES
scripts/fraud_dataloader.py
src/feature_store/fraud/__init__.py
src/feature_store/fraud/feature_definitions.py
src/feature_store/fraud/velocity.py             # bucket ring-buffer merge (§3b)
src/feature_store/fraud_ingest.py
tests/test_velocity_parity.py                   # SQL vs merge_buckets() parity (§3b)

src/ml_pipelines_kfp/shared/__init__.py
src/ml_pipelines_kfp/shared/components/__init__.py
src/ml_pipelines_kfp/shared/components/register.py
src/ml_pipelines_kfp/shared/components/deploy.py
src/ml_pipelines_kfp/shared/components/get_model.py
src/ml_pipelines_kfp/shared/components/schema.py

src/ml_pipelines_kfp/fraud_lgbm/__init__.py
src/ml_pipelines_kfp/fraud_lgbm/constants.py
src/ml_pipelines_kfp/fraud_lgbm/models/__init__.py
src/ml_pipelines_kfp/fraud_lgbm/models/instance.py
src/ml_pipelines_kfp/fraud_lgbm/models/prediction.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/__init__.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/fraud_pipeline_training.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/fraud_pipeline_inference.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/__init__.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/data.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/feature_engineering.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/optuna_tuning.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/calibration.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/evaluation.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/inference.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/fastapi/__init__.py
src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/fastapi/fraud_server.py

src/ml_pipelines_kfp/schemas/fraud_lgbm/vertex/instance.yaml
src/ml_pipelines_kfp/schemas/fraud_lgbm/vertex/prediction.yaml

src/dataflow/fraud_feature_pipeline.py
src/dataflow/models/fraud_schema.py

Dockerfile.fraud-fastapi

dags/fraud_training_staging_dag.py
dags/fraud_training_prod_dag.py
dags/fraud_batch_inference_staging_dag.py
dags/fraud_batch_inference_prod_dag.py

observability/grafana/dashboards/fraud-model-health.json

# MODIFIED FILES
src/feature_store/setup.py                    # Add "fraud" to CONFIGS
src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py  # Update imports to shared/
src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_inference.py # Update imports to shared/
.github/workflows/cicd.yaml                  # Add fraud image builds
pyproject.toml                                # Add optuna, lightgbm dependencies
requirements.fastapi.txt                      # Add fraud server dependencies (if separate)
```

---

## Implementation Phases

### Phase 1: Data Layer + Feature Store
**Estimated files: 7 | Dependencies: None**

0. **Verify array-typed features round-trip the Vertex FS sync** (§2 note). Do this first — a
   negative result changes the `fraud_user_features` schema and the Beam output shape, so it
   is cheap now and expensive after steps 2–3.
1. `scripts/fraud_dataloader.py` — synthetic data generator (adapt code.py)
2. `src/feature_store/fraud/__init__.py` + `feature_definitions.py` — FRAUD_CONFIG
3. `src/feature_store/fraud_ingest.py` — BQ SQL feature engineering
4. `src/feature_store/fraud/velocity.py` — bucket ring-buffer merge (§3b)
5. `tests/test_velocity_parity.py` — assert `merge_buckets()` matches the training SQL
6. Update `src/feature_store/setup.py` — add fraud to CONFIGS
7. Test: run dataloader → ingest → setup → sync end-to-end

### Phase 2: Shared Component Refactor
**Estimated files: 8 | Dependencies: None (can run in parallel with Phase 1)**

1. Create `src/ml_pipelines_kfp/shared/components/` with register, deploy, get_model, schema
2. Update iris pipeline imports to use shared components
3. Verify iris training pipeline still compiles and runs unchanged

### Phase 3: KFP Training Pipeline with Optuna
**Estimated files: 10 | Dependencies: Phase 1 + Phase 2**

1. `src/ml_pipelines_kfp/fraud_lgbm/constants.py`
2. `src/ml_pipelines_kfp/fraud_lgbm/models/instance.py` + `prediction.py`
3. `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/data.py`
4. `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/optuna_tuning.py`
5. `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/calibration.py` — Beta calibration (§4b)
6. `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/evaluation.py`
7. `src/ml_pipelines_kfp/fraud_lgbm/pipelines/fraud_pipeline_training.py`
8. Vertex schemas (instance.yaml, prediction.yaml)
9. Update `pyproject.toml` — add optuna, lightgbm (Beta calibration needs no new dep — it is
   `LogisticRegression` on two transformed columns)
9. Test: compile pipeline, submit to Vertex AI

### Phase 4: Real-Time Inference (FastAPI on Cloud Run)
**Estimated files: 4 | Dependencies: Phase 3 (needs a trained model)**

1. `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/fastapi/fraud_server.py`
   — model loaded in the **lifespan startup hook**, not lazily on first request (#8)
2. `Dockerfile.fraud-fastapi`
3. Test: local Docker run with mock model → score request → verify latency
4. Deploy to Cloud Run via the parameterized KFP deploy component (`min_instance_count>=1`,
   concurrency 8, startup CPU boost, startup probe, `smoke_test_path="/score"`)
5. Verify the *steady-state* p95 with the floor warm **and** measure the scale-out cold-start
   cost separately — the second number is the one that shows up as p99 in production

### Phase 5: Batch Inference Pipeline
**Estimated files: 3 | Dependencies: Phase 3**

1. `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/inference.py`
2. `src/ml_pipelines_kfp/fraud_lgbm/pipelines/fraud_pipeline_inference.py`
3. Test: compile and run on Vertex AI

### Phase 6: Streaming Feature Pipeline (Dataflow)
**Estimated files: 2 | Dependencies: Phase 1**

1. `src/dataflow/models/fraud_schema.py`
2. `src/dataflow/fraud_feature_pipeline.py`
3. Test: local DirectRunner with mock PubSub messages

### Phase 7: Orchestration + CI/CD
**Estimated files: 6 | Dependencies: All above**

1. Airflow DAGs (4 files: training/inference x staging/prod)
2. Update `.github/workflows/cicd.yaml` — add fraud image builds
3. Observability: fraud model health Grafana dashboard
4. End-to-end integration test

---

## Key Dependencies to Add

```toml
# pyproject.toml additions
optuna = ">=3.0"
lightgbm = ">=4.0"
```

```txt
# requirements.fastapi.txt additions (for fraud server)
google-cloud-aiplatform    # for online store reads
google-cloud-pubsub        # prediction logging (published inline, not fire-and-forget)
```

---

## Testing Strategy

| Layer | Test | Tool |
|-------|------|------|
| Feature engineering SQL | Validate output schema and feature ranges | BQ dry-run + unit tests with sample data |
| Optuna tuning component | Verify study completes with valid model | Local run with n_trials=3, small dataset |
| Evaluation component | Check metric logging and threshold gating | Unit test with known predictions |
| FastAPI fraud server | Score request latency, correct response format | pytest + httpx, local Docker |
| Dataflow feature pipeline | Parse → compute → write flow | DirectRunner with mock messages |
| KFP pipeline compilation | Pipeline YAML compiles without errors | `kfp.compiler.Compiler().compile()` |
| End-to-end | Full flow from data generation to prediction | Staging environment |

---

## BQ Table Summary

| Table | Purpose | Write Mode |
|-------|---------|------------|
| `ml_dataset.fraud_users` | Raw user data with labels | WRITE_TRUNCATE (full refresh) |
| `ml_dataset.fraud_transfers` | Raw transfer data | WRITE_TRUNCATE (training) / WRITE_APPEND (batch input) |
| `ml_dataset.fraud_features` | Transfer-grained feature table (offline only — training + batch inference; never synced to Bigtable) | WRITE_TRUNCATE (batch ingest) |
| `ml_dataset.fraud_user_features` | User-grained serving view (per-user attributes + velocity bucket ring buffers + `__global__` cold-start row; synced to Bigtable) | WRITE_TRUNCATE (batch ingest) / WRITE_APPEND (streaming) |
| `ml_dataset.fraud_predictions` | Batch inference results | WRITE_APPEND |
| `ml_dataset.fraud_predictions_streaming` | Real-time scoring audit log | WRITE_APPEND, partitioned by prediction_timestamp |

---

## Decisions Made

- **Model choice**: LightGBM (not XGBoost). Faster training, native categorical feature support, and lower memory footprint — good fit for the Optuna tuning loop.
- **Train/test split**: Out-of-time **and** user-disjoint (§3c) — time cutoff at the 80th percentile of `created_at`; any user appearing in the test period is fully excluded from train. Chosen over `StratifiedGroupKFold` because it prevents *both* user-identity leakage and temporal look-ahead (the latter matters most for fraud). Optuna CV uses `TimeSeriesSplit` to match. Identifier/time columns are dropped from the model matrix.
- **Point-in-time features**: Training features are computed as of each transfer's `created_at`, never as of query time. `account_age_days` is derived per transfer from `registration_date`; the serving table recomputes it as of ingest date rather than copying the training-row value. Mutable user attributes (`kyc_status`, `account_type`, `country_of_residence`) are current-state-only in the synthetic data and remain a documented skew source — adding `*_at` change timestamps to the generator is the preferred Phase 1 fix (§3).
- **Windowed distinct counts**: `ARRAY_AGG` over the window + `COUNT(DISTINCT)` on `UNNEST`, not `HLL_COUNT`. BigQuery has no `COUNT(DISTINCT)` analytic function; the array route stays exact, and sketch-based approximation buys nothing on fields with the cardinality of `recipient_id`/`target_currency` at this scale. `HLL_COUNT` remains the documented fallback if the arrays ever exceed resources (§3).
- **Threshold management**: **Derived, not configured.** The evaluation component selects the threshold from the PR curve at a target precision (default 0.90) on the `calib` split, then verifies it out-of-time on `test` and gates on the achieved precision. A fixed 0.5 is meaningless while Optuna tunes `scale_pos_weight`, and stays a poor default even after calibration — 0.5 encodes "a false positive costs exactly what a false negative costs", which is not true for fraud. The chosen value is written to `ml_dataset.fraud_config` **keyed by `model_version`** (a rollback must not inherit the previous model's operating point), read by the FastAPI server on startup and refreshable via an admin endpoint for ops overrides.
- **Cold start**: Three states, not two (§7). *No history* → `__global__` population averages, stored on a dedicated entity in `fraud_user_features`. *Has history but idle* → real user attributes with **zero** velocity, which read-time bucket decay produces automatically. *Active* → merged buckets. Treating idle as missing (and handing back population averages) would be wrong in both directions — idle is information about a user.
- **Feature freshness / velocity decay**: The online store holds **bucket ring buffers** (hourly ×24, daily ×30, each slot stamped with its own epoch hour/day), not rolled-up scalars. `velocity.merge_buckets(row, as_of=now)` (§3b) rolls them up at request time, so an idle user's counts decay to zero with no writer doing anything. Chosen over timer-based flushes in the Beam DoFn (pays a write per idle user per interval — the write amplification time-bucketing exists to avoid — and still leaves a staleness gap) and over daily batch re-sync alone (bounds staleness at 24h on a 24h-window feature, i.e. barely a fix). Residual staleness is one bucket of in-flight transfers, not "however long since the user last transacted". Nightly batch re-sync is retained, but as **reconciliation** against Beam drops/duplicates, not as the decay mechanism.
- **Rollup logic lives in one module**: `velocity.py` is imported by both Beam and FastAPI, and pinned to the training SQL by a required parity test. Three copies of window-boundary logic is a skew bug waiting to happen (same family as finding #3).
- **Prediction logging**: Published to PubSub **inline, awaited before the response returns** — not `asyncio.create_task`, and not relying on the client's background batch thread, both of which stall under Cloud Run's post-response CPU throttling. Costs ~5-15ms of the p95 budget (5ms batch window + round trip). Publish failure increments a counter and degrades to a structured stdout line; it never fails a scoring request. "CPU always allocated" was rejected as primary because it puts correctness in a deploy flag whose regression is silent; retained as the documented fallback.
- **Serving capacity**: No scale-from-zero for the fraud API — `min_instance_count` 2 in prod / 1 in staging, `max_instance_request_concurrency=8` (CPU-bound predict; the 80 default queues), startup CPU boost, model loaded in the lifespan hook behind a startup probe. The shared deploy component is parameterized with these knobs defaulted to the current iris values, so iris is unaffected (finding #8). The instance floor covers steady-state p95; scale-out cold starts remain a p99 cost and are measured separately rather than folded into the p95 claim.
- **Probability calibration**: **Beta calibration** fitted post-training on a dedicated `calib` split (§4b), chosen over Platt (cannot map 0→0/1→1 — wrong shape for reweighted scores) and isotonic (overfits with few positives). The holdout is **user-disjoint from fit but spans the same time range** — random over users, never over rows. Deliberately unlike the train/test split: a later time slice would fold temporal drift into the calibration fit, leaving the calibrator correcting two things at once. The out-of-time check moves to §5, which measures Brier/ECE on `test`; a large calib-vs-test gap is the signal to schedule recalibration. Model and calibrator persist as a single bundle so serving cannot apply one without the other.
- **Feature drift monitoring**: Deferred. Not included in the initial build. Can be added as a follow-up phase to track distribution drift between training and serving data — important for fraud since transaction patterns shift over time.

---

## Review Findings — To Address (added 2026-07-01, for review tomorrow)

Items 1–6 should be resolved in the plan before writing code — especially the transfer-grained vs. user-grained feature split (#1), since it changes the feature store setup, the streaming pipeline, and the FastAPI fetch path. **#1–#4, #6–#8, and #10 are now resolved in the plan (2026-08-04 → 08-07); #5 and #9 remain open.**

### Critical design issues

**1. Entity granularity mismatch between training and serving (biggest one). — ✅ RESOLVED (2026-08-04).**
The plan keyed `fraud_features` by `entity_id = transfer_id + source` — one row per transfer, right for training. But the FastAPI server fetches from the online store by `request.sender_id`. A feature view keyed by transfer_id can't serve lookups by sender_id. **Fixed** by splitting into two tables:
- `fraud_features` (transfer-grained) — offline only, for training and batch inference
- `fraud_user_features` (user-grained, keyed by `user_id`, looked up by `sender_id`) — holds per-user velocity state, synced to Bigtable for real-time serving (bucket ring buffers as of finding #6)

The streaming pipeline and the `__global__` cold-start entity now belong to the user-grained view. Applied throughout: "Feature Tables: Two Granularities" section, the architecture diagram, `FRAUD_USER_CONFIG`, the `fraud_ingest.py` second write, the FastAPI fetch path, the streaming §8 write targets, and the BQ Table Summary.

**2. Train/test leakage across users. — ✅ RESOLVED (2026-08-05).**
Labels live on the user, and each user has many transfers. A random `train_test_split` plus `StratifiedKFold` in the Optuna component would put the same sender's transfers in both train and validation folds — the model can memorize user identity via features like `account_age_days` and produce inflated CV scores. **Fixed** with an **out-of-time + user-disjoint** split (§3c) rather than plain `StratifiedGroupKFold`: a time cutoff at the 80th percentile of `created_at`, with every test-period user fully excluded from train — which prevents both user-identity leakage *and* temporal look-ahead. Optuna CV switched to `TimeSeriesSplit`; identifier/time columns are dropped from the model matrix in both tuning and evaluation. Applied in: new §3c, the §4 Optuna component, the §5 evaluation note, component #7, and Decisions Made.

**3. Point-in-time correctness in the feature SQL. — ✅ RESOLVED (2026-08-06).**
`account_age_days` was computed with `DATE_DIFF(CURRENT_DATE(), registration_date)` — that's account age now, not at the time of the transfer (transfers span 2 years back). Training features must be computed as of `created_at`, otherwise there's training/serving skew. The window functions are fine; the user-join features were not. **Fixed** by:
- The `user_features` CTE now carries `registration_date` through instead of pre-aggregating it into an age. The age is computed in the final SELECT as `DATE_DIFF(DATE(t.created_at), DATE(u.registration_date), DAY)` — per transfer, as of that transfer.
- `fraud_user_features` **recomputes** `account_age_days` from `fraud_users.registration_date` as of ingest time (it now joins `fraud_users`) rather than copying the value off the user's last transfer row, which would have frozen each user's age at their last activity. Same definition, different as-of date — that's what keeps training and serving consistent.
- A new as-of table in §3 documents the rule per feature, plus an explicit **known limitation**: `kyc_status` / `account_type` / `country_of_residence` are current-state-only in `fraud_users`, so they still leak a future fact — and in the label-correlated direction, since KYC review is often triggered by the fraud itself. Preferred fix (add `kyc_verified_at` etc. to the generator and resolve as of `created_at`) is written up for a Phase 1 decision.

Applied in: the §3 feature SQL, the `fraud_user_features` derivation, the §3 as-of table and limitation note, the Engineered Features table, and Decisions Made.

**4. The BQ SQL as written won't run. — ✅ RESOLVED (2026-08-06).**
`COUNT(DISTINCT recipient_id) OVER (...)` — BigQuery doesn't support `COUNT(DISTINCT)` as an analytic function. `unique_recipients_7d` and `unique_currencies_7d` need a workaround (e.g., `HLL_COUNT`, or a correlated aggregation via `ARRAY_AGG` in a subquery). Also, the outer `SELECT` references window names (`sender_7d`) defined inside a CTE — window definitions don't escape their query scope. **Fixed** by restructuring §3 into `transfer_windows` → `transfer_features` → final join:
- Distinct counts use the **`ARRAY_AGG` route** (chosen over `HLL_COUNT` — exact, and approximation buys nothing on a field as low-cardinality as `target_currency`): `ARRAY_AGG(recipient_id IGNORE NULLS) OVER (sender_7d)` in `transfer_windows`, then `(SELECT COUNT(DISTINCT r) FROM UNNEST(recipients_7d) r)` in `transfer_features`. `IGNORE NULLS` is not optional — `ARRAY_AGG` errors on a NULL element. The arrays are dropped via `* EXCEPT (...)` so they never reach the feature table. Cost/fan-out characteristics and the fallbacks if it ever exceeds resources are documented under the query.
- `cross_border_ratio_7d` moved **into** `transfer_windows`, where `sender_7d` is actually in scope, and is written with the raw `IF(source_currency != target_currency, ...)` expression rather than the `is_cross_border` alias (a window function can't see an alias defined in its own SELECT list).

Two further errors in the same block, found while rewriting it and fixed:
- **The window frames wouldn't parse.** `RANGE BETWEEN INTERVAL 1 HOUR PRECEDING` is Postgres/Snowflake syntax; BigQuery `RANGE` frames require a *numeric* offset over a numeric `ORDER BY` key. Rewritten as `ORDER BY UNIX_SECONDS(created_at) RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW` (86400 / 604800 / 2592000 for the 24h / 7d / 30d windows).
- **`t.source` was never selected** in the CTE but the outer `SELECT` builds `entity_id` from it — added to the projection.

Applied in: the §3 feature SQL and the new `ARRAY_AGG` cost note.

**5. Synthetic labels have no signal — the pipeline will "fail" by design.**
In `code.py` the label is `np.random.choice([0,1], p=[0.9, 0.1])`, completely independent of behavior. A model trained on this gets AUC-PR ≈ 0.1 (random baseline), so the evaluation gate ("fail pipeline if auc_pr < minimum") will always trip. The data generator needs to inject actual fraud patterns — e.g., fraudulent users get newer accounts, higher transfer velocity, more cross-border transfers, unverified KYC — so the end-to-end demo actually works.

**6. Velocity features never decay. — ✅ RESOLVED (2026-08-06).**
With time-bucketed aggregation, Bigtable only updates when a user sends a transfer. A user who sent 50 transfers yesterday and then stopped keeps `tx_count_24h = 50` in the online store indefinitely — a stale high-risk signal. Need either timer-based flushes in the Beam stateful DoFn, or (simpler) let the daily batch ingest + `sync.py` re-sync recompute and overwrite everyone. Worth stating which one in the plan.

**Fixed by rejecting both options and changing the representation instead.** `tx_count_24h` is not a property of a user, it is a property of *(user, time)*; storing it as a scalar written once and read back later is the defect, and both proposed fixes keep that representation and just rewrite the scalar more often. Timer flushes cost a Bigtable write per idle user per interval — the write amplification time-bucketing exists to avoid, paid on the least active users — and still leave a gap between fires. Daily batch re-sync bounds staleness at 24h on a feature whose window *is* 24h.

The online store now holds **bucket columns**: hourly ×24 and daily ×30 ring buffers (slot = `epoch_hour % 24` / `epoch_day % 30`), each slot stamped with its own epoch hour/day. `velocity.merge_buckets(row, as_of)` sums only the slots inside the window measured back from request time, so an idle user's slots fall out of range and merge to zero **with no writer having to act** — decay that stays correct even when the streaming pipeline is down. Bucket columns were chosen over a composite `user_id#reverse_timestamp` row key with a range scan: the keyed design is cleaner for raw Bigtable, but Vertex AI Feature Store serves point lookups by entity ID and does not expose range scans, and keeping the Vertex Feature Store path end-to-end is a goal of this repo.

Applied in: `FRAUD_USER_CONFIG` (bucket + `global_*` columns), the `fraud_user_features` derivation SQL, new §3b (`velocity.py` + required parity test), §7 three-state fetch with a staleness guard, §8 (buckets not scalars, no timers, batch retained as reconciliation), the file structure, and Decisions Made.

Three consequences worth tracking:
- **Distinct counts are not additive** across buckets, so member IDs travel with each bucket and are unioned at read time (capped at 64/bucket). Per-bucket HLL++ sketches are the scale-up path, deferred because BigQuery's zetasketch format has no good Python reader and the merge runs in FastAPI.
- **Array-typed features must round-trip the Vertex FS sync** — a Phase 1 verification, with flattened numbered scalar columns as the fallback. Flagged in §2.
- **Cold start is now three states** (no history / idle / active), not two.

### Serving-layer concerns

**7. `asyncio.create_task` for prediction logging won't work reliably on Cloud Run. — ✅ RESOLVED (2026-08-07).**
By default Cloud Run throttles CPU to ~0 after the response is sent, so fire-and-forget tasks silently stall. Either publish to PubSub before returning (a batched publish is ~1-2ms), or deploy with "CPU always allocated." **Fixed** by publishing to PubSub before returning, and specifically by **awaiting the publish future** — calling `publish()` only enqueues into a batch that a background thread commits, which is the same throttled-CPU trap one layer down. Details in §7:
- `BatchSettings(max_messages=100, max_bytes=1MB, max_latency=0.005)` — `max_latency` is the term that lands directly in p95, so it is tuned as a latency budget, not for throughput.
- `future.result()` is blocking, so it runs via `run_in_executor` under `asyncio.wait_for` — same treatment the blocking online-store fetch needs (Minor nits).
- A publish failure **must not fail scoring**: it increments a counter and falls back to a structured stdout line (captured by Cloud Logging without post-response CPU). A Pub/Sub outage degrades the audit trail; it does not take fraud decisioning offline.
- The payload carries the full scored `feature_vector`, not just the score — that is what makes training/serving skew detectable per-decision rather than only in aggregate.

"CPU always allocated" was rejected as the primary fix: it works, but it makes correctness depend on a deploy-time flag. A future Terraform edit flips it and prediction logging stops with no error anywhere — invisible precisely because fire-and-forget swallows the failure. It also bills idle CPU, which with `min_instances >= 1` (finding #8) is always-on. It remains the documented fallback if the ~5-15ms inline cost proves too expensive, paired with an explicit in-process queue.

Applied in: the §7 handler, the new publish-path section, the architecture diagram, the latency breakdown, dependencies, and Decisions Made.

**8. Cold starts break the latency target. — ✅ RESOLVED (2026-08-07).**
The reused deploy component sets `min_instance_count: 0`. A scale-from-zero Cloud Run start (container boot + model download from GCS) is multiple seconds — incompatible with a <50ms p95 fraud API. Fraud deploy needs `min_instances >= 1`, which likely means parameterizing the shared deploy component. **Fixed** by parameterizing every hardcoded serving knob in `shared/components/deploy.py`, each defaulted to its current value so the iris pipeline is untouched by the refactor. Fraud deploys with `min_instance_count=2` in prod (1 in staging) — one warm instance is one instance, and a revision rollout or zonal blip removes it exactly when live traffic needs it.

Three things the finding didn't cover, now in the plan:
- **Concurrency matters as much as the instance floor.** Cloud Run defaults to 80 concurrent requests per instance, which is a throughput default. LightGBM predict is CPU-bound, so 80 in flight on 2 vCPU queues, and queueing is p95 latency that no instance floor fixes. Fraud sets `max_instance_request_concurrency=8`.
- **A warm container is not a ready one.** If the model is downloaded from GCS lazily on first request, `min_instances` buys nothing. The model loads in the FastAPI lifespan startup hook and the service gets a **startup probe** so traffic isn't routed into a container still unpacking the model.
- **The floor doesn't cover scale-out.** Traffic past the floor still boots cold containers; that's a p99 problem the floor cannot solve, mitigated with `startup_cpu_boost` and a lean image, and called out as a separate measurement in Phase 4 rather than hidden inside the p95 number.

Also fixed while in there: the component's post-deploy smoke test POSTs a hardcoded *iris* payload to `/predict`. The fraud service has no `/predict` endpoint, so the existing test would have failed every fraud deploy — `smoke_test_path` and `smoke_test_payload` are now parameters.

Applied in: the new "Parameterizing `deploy.py`" subsection, the reusable-components table, §6 pipeline orchestration, and Phase 4. `cpu_idle` is parameterized in the same signature so finding #7's documented fallback is a one-line change.

> **Note:** the `allUsers` IAM binding (finding #9) is in this same component and this same
> refactor. It is deliberately **not** fixed here — #9 remains open — but it lands on the seam
> this change creates, so do them together if #9 is picked up next.

**9. The deploy component grants `allUsers` invoker.**
Fine for the iris demo; a public unauthenticated fraud-scoring endpoint is not something to copy. The fraud service should require IAM auth (and this is another reason to parameterize the shared deploy component rather than reuse as-is).

**10. Threshold 0.5 is meaningless if you tune `scale_pos_weight`. — ✅ RESOLVED (2026-08-07).**
Optuna searching `scale_pos_weight` in [1, 15] distorts predicted probabilities, so a fixed 0.5 threshold in the config table won't correspond to any intended precision/recall point. Either calibrate the model after training, or have the evaluation component pick the threshold from the PR curve (e.g., precision ≥ X) and write that into the config table. **Both halves are now done** — they are complementary, not the either/or the finding presents. Calibration fixes score *magnitude*; threshold selection picks the *operating point*. Calibration alone would make 0.5 meaningful but not correct, since 0.5 encodes "a false positive costs exactly what a false negative costs", which is false for fraud.

**Calibration (§4b).** **Beta calibration** (Kull et al., 2017) — chosen over Platt, which assumes a sigmoid distortion and cannot map 0→0 or 1→1, exactly wrong for [0,1] scores squashed by class reweighting; and over isotonic, which overfits on the few positives fraud data has. It is `LogisticRegression` on `[log(s), -log(1-s)]`, so it adds no dependency.
- **`a, b ≥ 0` is enforced.** A negative coefficient makes the Beta map non-monotonic, which reorders scores and silently changes AUC-PR — a "calibration" step that is actually a ranking change. Restricted two-parameter refit on violation, plus an assert that AUC-ROC is unchanged.
- **The calibration set is not rebalanced** — its job is mapping reweighted scores back onto the true prior, so it must carry production prevalence.
- **Model and calibrator ship as one artifact** (`CalibratedFraudModel`). Separate artifacts invite a serving path that loads the model and forgets the calibrator — silent, because uncalibrated scores look perfectly reasonable.

**Threshold selection (§5).** Maximize recall subject to `precision >= target_precision` (default 0.90) on the PR curve, with a `min_predicted_positives` support floor — precision is not monotonic in threshold, so "first index clearing the target" would latch onto a jagged spike in the sparse high-threshold tail. An unachievable target raises rather than falling back to a default; silently shipping 0.5 would mean the config table claims an operating point the model does not deliver.
- **Selected on `calib`, verified on `test`.** Picking the threshold on test and then reporting precision at it on the same rows is fitting a parameter to the holdout and grading yourself on it. A second gate fails the run if test precision falls short of the target by more than a tolerance — the threshold is chosen in-period, and this is the check that it survives out-of-time.
- **Written to `fraud_config` keyed by `model_version`**, and only for a blessed model. A threshold is a property of a model, not of the service: a rollback that inherits the previous model's threshold silently moves the operating point.

**Split shape (§3c).** `calib` is **user-disjoint from `fit` but spans the same time range** — random over *users*, deterministically seeded; never random over rows, which would put the same sender on both sides. Deliberately unlike train/test: pushing `calib` into a later window would fold temporal drift into the calibration fit, so the calibrator would correct two things at once with no way to tell them apart, and it would also cost `fit` the most recent 20% of history. The out-of-time check moves to §5's test-split Brier/ECE, where a large calib-vs-test gap is the signal to schedule recalibration rather than treat it as one-off.

**Positive count per split is logged and gated** (~100 floor). `calib` now carries two jobs — calibration *and* threshold selection — so a thin calibration holdout is doubly expensive. Another concrete reason to scale the generator up under finding #5.

Applied in: §3c (three-way split, user-disjoint calib), §4 (tunes on `fit` only), §4b, §5 (threshold selection + calibrated evaluation + `fraud_config` write), §6 orchestration, §7 serving, the components table, file structure, Phase 3, and Decisions Made.

### Minor nits

- BQ `EXTRACT(DAYOFWEEK)` is 1–7 (Sunday=1) while Python's `weekday()` is 0–6 (Monday=0) — the FastAPI real-time feature computation must match the SQL convention or there's silent skew.
- The `FeatureOnlineStoreServiceClient` fetch is a blocking call; calling it inside `async def score_transfer` blocks the event loop. Use the async client or `run_in_executor`.
- In `code.py`, `exchange_rate` is sampled once for all transfers — fine to fix while adapting the generator.
- Optuna with 50 trials × 5 folds × up to 1000 estimators = 250 fits; add a pruner (e.g., `MedianPruner`) and set CPU/memory on the KFP component so it doesn't crawl on the default machine.
- The fraud server response includes `model_version` — the deploy component should pass the blessed model version as an env var so the server can report it.
