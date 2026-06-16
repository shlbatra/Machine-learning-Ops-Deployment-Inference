"""One-time setup for Feature Store infrastructure.

Creates a shared FeatureOnlineStore (Bigtable-backed) and a
project-specific FeatureView pointing at the canonical BQ feature table.
Idempotent — skips resources that already exist.

Run once per ML project before first sync:
    python -m feature_store.setup                  # defaults to iris
    python -m feature_store.setup --config fraud   # new project
"""

from __future__ import annotations

import argparse
import importlib

from google.api_core import exceptions as api_exceptions
from google.cloud import aiplatform

from feature_store.schema import FeatureConfig

PROJECT = "deeplearning-sahil"
REGION = "us-central1"

CONFIGS: dict[str, str] = {
    "iris": "feature_store.iris.feature_definitions.IRIS_CONFIG",
}


def _load_config(name: str) -> FeatureConfig:
    """Import a FeatureConfig by registered name."""
    if name not in CONFIGS:
        available = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    module_path, attr = CONFIGS[name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _get_or_create_online_store(
    store_id: str, project: str, region: str
) -> aiplatform.FeatureOnlineStore:
    """Return existing online store or create a new Bigtable-backed one."""
    try:
        store = aiplatform.FeatureOnlineStore(
            feature_online_store_name=store_id,
            project=project,
            location=region,
        )
        print(f"Online store '{store_id}' already exists — skipping creation.")
        return store
    except api_exceptions.NotFound:
        pass

    print(f"Creating online store '{store_id}'...")
    store = aiplatform.FeatureOnlineStore.create_bigtable_store(
        name=store_id,
        project=project,
        location=region,
    )
    print(f"Created online store '{store_id}'.")
    return store


def _get_or_create_feature_view(
    store: aiplatform.FeatureOnlineStore,
    view_id: str,
    bq_uri: str,
    entity_id_column: str,
) -> aiplatform.FeatureView:
    """Return existing feature view or create one backed by a BQ table."""
    try:
        view = store.get_feature_view(feature_view_id=view_id)
        print(f"Feature view '{view_id}' already exists — skipping creation.")
        return view
    except api_exceptions.NotFound:
        pass

    print(f"Creating feature view '{view_id}' → {bq_uri}...")
    view = store.create_feature_view(
        name=view_id,
        source=aiplatform.FeatureView.BigQuerySource(
            uri=bq_uri,
            entity_id_columns=[entity_id_column],
        ),
    )
    print(f"Created feature view '{view_id}'.")
    return view


def setup(
    config_name: str = "iris",
    project: str = PROJECT,
    region: str = REGION,
) -> None:
    aiplatform.init(project=project, location=region)

    cfg = _load_config(config_name)
    bq_uri = f"bq://{project}.{cfg.bq_dataset}.{cfg.bq_feature_table}"

    store = _get_or_create_online_store(cfg.online_store_id, project, region)
    _get_or_create_feature_view(store, cfg.feature_view_id, bq_uri, cfg.entity_id_column)

    print(f"Feature Store setup complete for '{config_name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up Feature Store online store and feature view",
    )
    parser.add_argument(
        "--config",
        default="iris",
        help="Feature config name (default: %(default)s). Available: "
        + ", ".join(sorted(CONFIGS)),
    )
    parser.add_argument(
        "--project", default=PROJECT, help="GCP project ID (default: %(default)s)"
    )
    parser.add_argument(
        "--region", default=REGION, help="GCP region (default: %(default)s)"
    )
    args = parser.parse_args()
    setup(config_name=args.config, project=args.project, region=args.region)


if __name__ == "__main__":
    main()
