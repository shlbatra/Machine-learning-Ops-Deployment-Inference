"""Iris project feature definitions."""

from feature_store.schema import FeatureConfig

IRIS_CONFIG = FeatureConfig(
    name="iris",
    feature_columns=[
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ],
    entity_id_column="entity_id",
    target_column="species",
    column_mappings={
        "camel": {
            "SepalLengthCm": "sepal_length_cm",
            "SepalWidthCm": "sepal_width_cm",
            "PetalLengthCm": "petal_length_cm",
            "PetalWidthCm": "petal_width_cm",
            "Species": "species",
        },
        "snake": {
            "sepal_length": "sepal_length_cm",
            "sepal_width": "sepal_width_cm",
            "petal_length": "petal_length_cm",
            "petal_width": "petal_width_cm",
        },
    },
    bq_dataset="ml_dataset",
    bq_raw_table="iris",
    bq_batch_input_table="iris_batch_input",
    bq_feature_table="iris_features",
    online_store_id="iris_online_store",
    feature_view_id="iris_features",
)
