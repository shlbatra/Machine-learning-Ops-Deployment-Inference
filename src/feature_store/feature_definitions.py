"""Single source of truth for feature names, mappings, and Feature Store resources."""

# --- Canonical feature column names ---
FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]

ENTITY_ID_COLUMN = "entity_id"
TARGET_COLUMN = "species"
FEATURE_TIMESTAMP_COLUMN = "feature_timestamp"

# --- Column mapping: CamelCase (BQ raw / training) → canonical ---
CAMEL_TO_CANONICAL = {
    "SepalLengthCm": "sepal_length_cm",
    "SepalWidthCm": "sepal_width_cm",
    "PetalLengthCm": "petal_length_cm",
    "PetalWidthCm": "petal_width_cm",
    "Species": "species",
}

# --- Column mapping: snake_case (Pub/Sub / streaming) → canonical ---
SNAKE_TO_CANONICAL = {
    "sepal_length": "sepal_length_cm",
    "sepal_width": "sepal_width_cm",
    "petal_length": "petal_length_cm",
    "petal_width": "petal_width_cm",
}

# --- Reverse mapping: canonical → CamelCase (for backward compat) ---
CANONICAL_TO_CAMEL = {v: k for k, v in CAMEL_TO_CANONICAL.items()}

# --- BigQuery ---
BQ_DATASET = "ml_dataset"
BQ_RAW_TABLE = "iris"
BQ_FEATURE_TABLE = "iris_features"

# --- Feature Store resource IDs ---
FEATURE_ONLINE_STORE_ID = "iris_online_store"
FEATURE_VIEW_ID = "iris_features"
