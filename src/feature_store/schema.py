"""Shared schema for feature store configurations across ML projects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureConfig:
    """Defines the feature contract for a single ML project."""

    name: str
    feature_columns: list[str]
    entity_id_column: str
    target_column: str
    feature_timestamp_column: str = "feature_timestamp"

    column_mappings: dict[str, dict[str, str]] = field(default_factory=dict)

    bq_dataset: str = ""
    bq_raw_table: str = ""
    bq_feature_table: str = ""

    online_store_id: str = ""
    feature_view_id: str = ""

    @property
    def canonical_to_source(self) -> dict[str, dict[str, str]]:
        """Reverse each mapping: canonical name → source name, keyed by source label."""
        return {
            label: {v: k for k, v in mapping.items()}
            for label, mapping in self.column_mappings.items()
        }
