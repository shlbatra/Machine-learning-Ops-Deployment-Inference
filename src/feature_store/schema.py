"""Shared schema for feature store configurations across ML projects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureConfig:
    """Defines the feature contract for a single ML project."""

    # Project identifier, e.g. "iris", "fraud"
    name: str
    # Model input features in canonical form, e.g. ["sepal_length_cm", ...]
    feature_columns: list[str]
    # Row-level unique key used for online store lookups
    entity_id_column: str
    # Label column for supervised training; dropped during inference
    target_column: str
    # Timestamp column required by Feature Store for point-in-time joins
    feature_timestamp_column: str = "feature_timestamp"

    # Source-to-canonical name mappings keyed by source label,
    # e.g. {"camel": {"SepalLengthCm": "sepal_length_cm"}, "snake": {"sepal_length": "sepal_length_cm"}}
    column_mappings: dict[str, dict[str, str]] = field(default_factory=dict)

    # BigQuery dataset shared across raw and feature tables
    bq_dataset: str = ""
    # Raw ingestion table written by bq_dataloader, e.g. "iris"
    bq_raw_table: str = ""
    # Separate table for unlabeled batch inference input data
    bq_batch_input_table: str = ""
    # Canonical feature table used as the offline store for training and batch inference
    bq_feature_table: str = ""

    # Bigtable-backed online store shared across ML projects
    online_store_id: str = "ml_online_store"
    # View that maps the BQ feature table into the online store
    feature_view_id: str = ""

    @property
    def canonical_to_source(self) -> dict[str, dict[str, str]]:
        """Reverse each mapping: canonical name → source name, keyed by source label."""
        return {
            label: {v: k for k, v in mapping.items()}
            for label, mapping in self.column_mappings.items()
        }
