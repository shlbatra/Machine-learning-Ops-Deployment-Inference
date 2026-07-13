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

### Engineered Features (written to `fraud_features`)

**User-level features** (slow-changing, precomputed in batch):
| Feature | Description |
|---------|-------------|
| account_age_days | Days since registration_date |
| is_business | 1 if account_type = Business |
| is_kyc_verified | 1 if kyc_status = Verified |
| country_risk_score | Mapped risk score by country (e.g., higher for less regulated jurisdictions) |

**Transaction velocity features** (computed over sender_id windows):
| Feature | Description |
|---------|-------------|
| tx_count_1h | Transfers sent in the past 1 hour |
| tx_count_24h | Transfers sent in the past 24 hours |
| tx_count_7d | Transfers sent in the past 7 days |
| tx_count_30d | Transfers sent in the past 30 days |

**Amount aggregation features**:
| Feature | Description |
|---------|-------------|
| avg_amount_7d | Average source_amount in past 7 days |
| max_amount_7d | Max source_amount in past 7 days |
| total_amount_24h | Sum of source_amount in past 24 hours |
| amount_ratio_to_avg | current_amount / avg_amount_7d (spike detection) |

**Cross-border and diversity features**:
| Feature | Description |
|---------|-------------|
| is_cross_border | 1 if source_currency != target_currency |
| cross_border_ratio_7d | Fraction of cross-border transfers in past 7 days |
| unique_recipients_7d | Distinct recipient_ids in past 7 days |
| unique_currencies_7d | Distinct target_currencies in past 7 days |

**Transaction-level features** (computed at inference time from the payload):
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
                                    ▼
                          BQ: fraud_features (offline store)
                                    │
                          feature_store/sync.py
                                    │
                                    ▼
                          Bigtable (online store)

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
  │  get_model ──▶ batch_inference (BQ fraud_features) ──▶ BQ   │
  └──────────────────────────────────────────────────────────────┘

  REAL-TIME (Direct Cloud Run — NO Dataflow):
  ┌───────────────────────────────────────────────────────────────────────┐
  │  Client ──POST──▶ Cloud Run FastAPI                                  │
  │                      │                                               │
  │                      ├─ Fetch user features from Bigtable            │
  │                      ├─ Compute real-time features from payload      │
  │                      ├─ Run LightGBM prediction                      │
  │                      ├─ Return {fraud_score, is_fraud} synchronously │
  │                      └─ Async: publish prediction log to PubSub      │
  │                              └──▶ BQ: fraud_predictions_streaming    │
  └───────────────────────────────────────────────────────────────────────┘

  STREAMING FEATURE REFRESH (Dataflow):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  PubSub (transfer events) ──▶ Beam ──▶ compute velocity features    │
  │                                    ──▶ dual-write BQ + Bigtable     │
  └──────────────────────────────────────────────────────────────────────┘
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
- Async publishes prediction log to PubSub (non-blocking, for audit trail)

Dataflow is still used, but only for the **feature refresh path** — keeping velocity/aggregation features in Bigtable up-to-date as new transfers arrive.

---

## Component Inventory: Reuse vs. New

### Reusable Components (no or minimal changes)

| Component | Path | Change Needed |
|-----------|------|---------------|
| FeatureConfig schema | `src/feature_store/schema.py` | None — already generic |
| Feature Store setup | `src/feature_store/setup.py` | Add `"fraud"` entry to CONFIGS dict |
| Feature Store sync | `src/feature_store/sync.py` | None — already parameterized by config name |
| Online store writer | `src/dataflow/utils/online_store_writer.py` | None — generic |
| Online store reader | `src/dataflow/utils/online_store_reader.py` | None — generic |
| Dead letter handler | `src/dataflow/utils/dead_letter.py` | None — generic |
| Model registry (upload) | `iris_xgboost/pipelines/components/register.py` | Refactor to shared location (see below) |
| Cloud Run deploy | `iris_xgboost/pipelines/components/deploy.py` | Refactor to shared location |
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

### New Components to Build

| # | Component | Path | Description |
|---|-----------|------|-------------|
| 1 | Fraud data generator | `scripts/fraud_dataloader.py` | Adapted from `plan/code.py`, writes users + transfers to BQ |
| 2 | Fraud feature config | `src/feature_store/fraud/feature_definitions.py` | FRAUD_CONFIG with all engineered feature columns |
| 3 | Fraud feature ingestion | `src/feature_store/fraud_ingest.py` | Joins users + transfers, computes aggregation features, writes to `fraud_features` |
| 4 | Fraud constants | `src/ml_pipelines_kfp/fraud_lgbm/constants.py` | BQ tables, model name, pipeline name, image names |
| 5 | Fraud instance model | `src/ml_pipelines_kfp/fraud_lgbm/models/instance.py` | Pydantic model for fraud features |
| 6 | Fraud prediction model | `src/ml_pipelines_kfp/fraud_lgbm/models/prediction.py` | fraud_score (float), is_fraud (bool) |
| 7 | Load fraud data (KFP) | `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/data.py` | Load from BQ fraud_features, filter source='training' |
| 8 | Optuna tuning (KFP) | `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/optuna_tuning.py` | LightGBM hyperparameter tuning with Optuna |
| 9 | Fraud evaluation (KFP) | `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/evaluation.py` | AUC-PR, precision/recall at thresholds, confusion matrix |
| 10 | Training pipeline | `src/ml_pipelines_kfp/fraud_lgbm/pipelines/fraud_pipeline_training.py` | Orchestrates load → tune → evaluate → register → deploy |
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

```python
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
    entity_id_column="entity_id",
    target_column="label",
    column_mappings={},  # No renaming needed — data generator uses canonical names
    bq_dataset="ml_dataset",
    bq_raw_table="fraud_transfers",
    bq_batch_input_table="fraud_transfers_batch_input",
    bq_feature_table="fraud_features",
    feature_view_id="fraud_features",
)
```

Register in `feature_store/setup.py`:
```python
CONFIGS = {
    "iris": "feature_store.iris.feature_definitions.IRIS_CONFIG",
    "fraud": "feature_store.fraud.feature_definitions.FRAUD_CONFIG",
}
```

### 3. Feature Ingestion — `src/feature_store/fraud_ingest.py`

This is the most complex new component. It joins users + transfers and computes all aggregation features using SQL window functions in BigQuery:

```sql
WITH user_features AS (
    SELECT
        user_id,
        DATE_DIFF(CURRENT_DATE(), DATE(registration_date), DAY) AS account_age_days,
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
transfer_features AS (
    SELECT
        t.transfer_id,
        t.sender_id,
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
        -- Diversity features
        COUNT(DISTINCT t.recipient_id) OVER (sender_7d) AS unique_recipients_7d,
        COUNT(DISTINCT t.target_currency) OVER (sender_7d) AS unique_currencies_7d,
        ...
    FROM `{project}.{dataset}.fraud_transfers` t
    WINDOW
        sender_1h AS (PARTITION BY sender_id ORDER BY created_at
                      RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW),
        sender_24h AS (...INTERVAL 24 HOUR...),
        sender_7d AS (...INTERVAL 7 DAY...),
        sender_30d AS (...INTERVAL 30 DAY...)
)
SELECT
    CONCAT(t.transfer_id, '_', t.source) AS entity_id,
    t.*, u.account_age_days, u.is_business, u.is_kyc_verified, u.country_risk_score,
    SAFE_DIVIDE(t.source_amount, t.avg_amount_7d) AS amount_ratio_to_avg,
    SAFE_DIVIDE(
        SUM(t.is_cross_border) OVER (sender_7d),
        COUNT(*) OVER (sender_7d)
    ) AS cross_border_ratio_7d,
    u.label,
    CURRENT_TIMESTAMP() AS feature_timestamp
FROM transfer_features t
JOIN user_features u ON t.sender_id = u.user_id
```

The ingestion script executes this query and writes results to `fraud_features` with WRITE_TRUNCATE.

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
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score
    import pandas as pd, joblib, json

    train = pd.read_csv(train_dataset.path)
    X = train.drop("label", axis=1)
    y = train["label"]

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

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
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

### 5. Fraud Evaluation — KFP Component

Unlike iris (accuracy-only), fraud needs imbalanced-class metrics:

```python
@component(base_image=_constants.IMAGE_NAME)
def evaluate_fraud_model(
    test_dataset: Input[Dataset],
    tuned_model: Input[Model],
    threshold: float,
    metrics: Output[Metrics],
    best_model: Output[Model],
):
    """Evaluate fraud model with AUC-PR, precision, recall at threshold."""

    from sklearn.metrics import (
        average_precision_score, precision_recall_curve,
        precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )

    # Load and predict
    # Log: auc_pr, auc_roc, precision_at_threshold, recall_at_threshold,
    #       f1_at_threshold, confusion_matrix counts
    # Gate: fail pipeline if auc_pr < minimum threshold
    # Save blessed model
```

### 6. Fraud Training Pipeline — KFP Orchestration

```
load_data_from_feature_store
        │
        ▼
optuna_lightgbm_tuning (n_trials=50)
        │
        ▼
evaluate_fraud_model (threshold=0.5)
        │
        ▼
upload_model (reused shared component)
        │
        ▼
deploy_blessed_model_to_fastapi (reused shared component)
```

### 7. Real-Time Fraud Scoring — FastAPI Server

This is the key architectural difference from iris. Instead of PubSub → Dataflow → FastAPI, the fraud model uses a **direct synchronous API**:

```python
# fraud_server.py — runs on Cloud Run

@app.post("/score", response_model=FraudScoreResponse)
async def score_transfer(request: TransferScoreRequest):
    """Score a single transfer for fraud risk.

    1. Fetch precomputed user + velocity features from Bigtable
    2. Compute real-time features from the transfer payload
    3. Run LightGBM prediction
    4. Return fraud score synchronously
    5. Async log prediction to PubSub
    """

    # Step 1: Fetch precomputed features from online store
    user_features = await fetch_from_online_store(request.sender_id)

    # Step 2: Compute real-time features from the payload
    realtime_features = compute_realtime_features(request)
    # hour_of_day, day_of_week, is_cross_border, source_amount

    # Step 3: Merge and predict
    feature_vector = {**user_features, **realtime_features}
    df = pd.DataFrame([feature_vector])
    fraud_prob = model.predict_proba(df)[0][1]

    # Step 4: Return immediately
    response = FraudScoreResponse(
        transfer_id=request.transfer_id,
        fraud_score=fraud_prob,
        is_fraud=fraud_prob >= THRESHOLD,
        model_version=MODEL_VERSION,
    )

    # Step 5: Async log (non-blocking)
    asyncio.create_task(publish_prediction_log(request, response))

    return response
```

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
- Total expected latency: **20-50ms p95** vs **500ms-2s** with Dataflow

### 8. Streaming Feature Refresh — Dataflow Pipeline (Time-Bucketed)

Dataflow is still used, but only for **keeping the feature store up-to-date** — not for real-time scoring. Features are aggregated using **hourly fixed windows** (time-bucketed), meaning Bigtable is updated once per user per hour rather than on every transfer. This trades up-to-1-hour staleness for ~10-50x fewer Bigtable writes at scale.

`src/dataflow/fraud_feature_pipeline.py`:
```
PubSub (transfer events)
    │
    ├─ Parse and validate (PubSubFraudMessage)
    │
    ├─ Fixed window (1 hour) keyed by sender_id
    │
    ├─ Aggregate within window:
    │     tx_count, sum/avg/max amounts, unique recipients/currencies,
    │     cross-border count
    │
    ├─ Merge hourly buckets into rolling features:
    │     tx_count_1h  = current bucket count
    │     tx_count_24h = sum of last 24 hourly buckets
    │     tx_count_7d  = sum of last 168 hourly buckets
    │     avg_amount_7d, max_amount_7d, etc.
    │
    ├─ Write to BQ fraud_features (offline store) — WRITE_APPEND
    │
    └─ Write to Bigtable (online store) — one write per user per hour
```

The real-time scoring FastAPI service reads the latest bucketed features from Bigtable. Features may be stale by up to 1 hour within the current bucket window.

---

## File Structure (New + Modified Files)

```
# NEW FILES
scripts/fraud_dataloader.py
src/feature_store/fraud/__init__.py
src/feature_store/fraud/feature_definitions.py
src/feature_store/fraud_ingest.py

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
**Estimated files: 5 | Dependencies: None**

1. `scripts/fraud_dataloader.py` — synthetic data generator (adapt code.py)
2. `src/feature_store/fraud/__init__.py` + `feature_definitions.py` — FRAUD_CONFIG
3. `src/feature_store/fraud_ingest.py` — BQ SQL feature engineering
4. Update `src/feature_store/setup.py` — add fraud to CONFIGS
5. Test: run dataloader → ingest → setup → sync end-to-end

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
5. `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/evaluation.py`
6. `src/ml_pipelines_kfp/fraud_lgbm/pipelines/fraud_pipeline_training.py`
7. Vertex schemas (instance.yaml, prediction.yaml)
8. Update `pyproject.toml` — add optuna, lightgbm
9. Test: compile pipeline, submit to Vertex AI

### Phase 4: Real-Time Inference (FastAPI on Cloud Run)
**Estimated files: 4 | Dependencies: Phase 3 (needs a trained model)**

1. `src/ml_pipelines_kfp/fraud_lgbm/pipelines/components/fastapi/fraud_server.py`
2. `Dockerfile.fraud-fastapi`
3. Test: local Docker run with mock model → score request → verify latency
4. Deploy to Cloud Run via KFP deploy component

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
google-cloud-pubsub        # for async prediction logging
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
| `ml_dataset.fraud_features` | Canonical feature table (offline store) | WRITE_TRUNCATE (batch ingest) / WRITE_APPEND (streaming) |
| `ml_dataset.fraud_predictions` | Batch inference results | WRITE_APPEND |
| `ml_dataset.fraud_predictions_streaming` | Real-time scoring audit log | WRITE_APPEND, partitioned by prediction_timestamp |

---

## Decisions Made

- **Model choice**: LightGBM (not XGBoost). Faster training, native categorical feature support, and lower memory footprint — good fit for the Optuna tuning loop.
- **Threshold management**: Stored in a BQ config table (`ml_dataset.fraud_config`), read by the FastAPI server on startup and refreshable via an admin endpoint. Allows ops to adjust without redeployment.
- **Cold start**: Use global averages as defaults for velocity/aggregation features when a user has no transfer history. The feature ingestion pipeline computes global stats and stores them as a special `__global__` entity in the online store.
- **Feature freshness**: Time-bucketed aggregation with 1-hour fixed windows. Bigtable is updated once per user per hour rather than on every transfer. Trades up-to-1-hour staleness for ~10-50x fewer writes at scale.
- **Feature drift monitoring**: Deferred. Not included in the initial build. Can be added as a follow-up phase to track distribution drift between training and serving data — important for fraud since transaction patterns shift over time.

---

## Review Findings — To Address (added 2026-07-01, for review tomorrow)

Items 1–6 should be resolved in the plan before writing code — especially the transfer-grained vs. user-grained feature split (#1), since it changes the feature store setup, the streaming pipeline, and the FastAPI fetch path.

### Critical design issues

**1. Entity granularity mismatch between training and serving (biggest one).**
The plan keys `fraud_features` by `entity_id = transfer_id + source` — one row per transfer, which is right for training. But the FastAPI server fetches from the online store by `request.sender_id`. A feature view keyed by transfer_id can't serve lookups by sender_id. We actually need two tables/views:
- `fraud_features` (transfer-grained) — offline only, for training and batch inference
- `fraud_user_features` (user-grained, keyed by sender_id) — holds the latest velocity/aggregate features per user, synced to Bigtable for real-time serving

The streaming pipeline and the `__global__` cold-start entity belong to the user-grained view. The plan currently conflates the two.

**2. Train/test leakage across users.**
Labels live on the user, and each user has many transfers. A random `train_test_split` plus `StratifiedKFold` in the Optuna component will put the same sender's transfers in both train and validation folds — the model can memorize user identity via features like `account_age_days` and produce inflated CV scores. Both the data split and the CV should be grouped by `sender_id` (`StratifiedGroupKFold`).

**3. Point-in-time correctness in the feature SQL.**
`account_age_days` is computed with `DATE_DIFF(CURRENT_DATE(), registration_date)` — that's account age now, not at the time of the transfer (transfers span 2 years back). Training features must be computed as of `created_at`, otherwise there's training/serving skew. The window functions are fine; the user-join features are not.

**4. The BQ SQL as written won't run.**
`COUNT(DISTINCT recipient_id) OVER (...)` — BigQuery doesn't support `COUNT(DISTINCT)` as an analytic function. `unique_recipients_7d` and `unique_currencies_7d` need a workaround (e.g., `HLL_COUNT`, or a correlated aggregation via `ARRAY_AGG` in a subquery). Also, the outer `SELECT` references window names (`sender_7d`) defined inside a CTE — window definitions don't escape their query scope.

**5. Synthetic labels have no signal — the pipeline will "fail" by design.**
In `code.py` the label is `np.random.choice([0,1], p=[0.9, 0.1])`, completely independent of behavior. A model trained on this gets AUC-PR ≈ 0.1 (random baseline), so the evaluation gate ("fail pipeline if auc_pr < minimum") will always trip. The data generator needs to inject actual fraud patterns — e.g., fraudulent users get newer accounts, higher transfer velocity, more cross-border transfers, unverified KYC — so the end-to-end demo actually works.

**6. Velocity features never decay.**
With time-bucketed aggregation, Bigtable only updates when a user sends a transfer. A user who sent 50 transfers yesterday and then stopped keeps `tx_count_24h = 50` in the online store indefinitely — a stale high-risk signal. Need either timer-based flushes in the Beam stateful DoFn, or (simpler) let the daily batch ingest + `sync.py` re-sync recompute and overwrite everyone. Worth stating which one in the plan.

### Serving-layer concerns

**7. `asyncio.create_task` for prediction logging won't work reliably on Cloud Run.**
By default Cloud Run throttles CPU to ~0 after the response is sent, so fire-and-forget tasks silently stall. Either publish to PubSub before returning (a batched publish is ~1-2ms), or deploy with "CPU always allocated."

**8. Cold starts break the latency target.**
The reused deploy component sets `min_instance_count: 0`. A scale-from-zero Cloud Run start (container boot + model download from GCS) is multiple seconds — incompatible with a <50ms p95 fraud API. Fraud deploy needs `min_instances >= 1`, which likely means parameterizing the shared deploy component.

**9. The deploy component grants `allUsers` invoker.**
Fine for the iris demo; a public unauthenticated fraud-scoring endpoint is not something to copy. The fraud service should require IAM auth (and this is another reason to parameterize the shared deploy component rather than reuse as-is).

**10. Threshold 0.5 is meaningless if you tune `scale_pos_weight`.**
Optuna searching `scale_pos_weight` in [1, 15] distorts predicted probabilities, so a fixed 0.5 threshold in the config table won't correspond to any intended precision/recall point. Either calibrate the model after training, or have the evaluation component pick the threshold from the PR curve (e.g., precision ≥ X) and write that into the config table.

### Minor nits

- BQ `EXTRACT(DAYOFWEEK)` is 1–7 (Sunday=1) while Python's `weekday()` is 0–6 (Monday=0) — the FastAPI real-time feature computation must match the SQL convention or there's silent skew.
- The `FeatureOnlineStoreServiceClient` fetch is a blocking call; calling it inside `async def score_transfer` blocks the event loop. Use the async client or `run_in_executor`.
- In `code.py`, `exchange_rate` is sampled once for all transfers — fine to fix while adapting the generator.
- Optuna with 50 trials × 5 folds × up to 1000 estimators = 250 fits; add a pruner (e.g., `MedianPruner`) and set CPU/memory on the KFP component so it doesn't crawl on the default machine.
- The fraud server response includes `model_version` — the deploy component should pass the blessed model version as an env var so the server can report it.
