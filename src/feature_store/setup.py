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
from google.cloud.aiplatform_v1 import (
    FeatureOnlineStore,
    FeatureOnlineStoreAdminServiceClient,
    FeatureView,
)

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


def _parent(project: str, region: str) -> str:
    return f"projects/{project}/locations/{region}"


def _store_name(project: str, region: str, store_id: str) -> str:
    return f"{_parent(project, region)}/featureOnlineStores/{store_id}"


def _get_or_create_online_store(
    client: FeatureOnlineStoreAdminServiceClient,
    store_id: str,
    project: str,
    region: str,
) -> str:
    """Return existing online store name or create a new Bigtable-backed one."""
    name = _store_name(project, region, store_id)
    try:
        client.get_feature_online_store(name=name)
        print(f"Online store '{store_id}' already exists — skipping creation.")
        return name
    except api_exceptions.NotFound:
        pass

    print(f"Creating online store '{store_id}'...")
    op = client.create_feature_online_store(
        parent=_parent(project, region),
        feature_online_store_id=store_id,
        feature_online_store=FeatureOnlineStore(
            bigtable=FeatureOnlineStore.Bigtable(
                auto_scaling=FeatureOnlineStore.Bigtable.AutoScaling(
                    min_node_count=1,
                    max_node_count=1,
                ),
            ),
        ),
    )
    op.result()
    print(f"Created online store '{store_id}'.")
    return name


def _get_or_create_feature_view(
    client: FeatureOnlineStoreAdminServiceClient,
    store_name: str,
    view_id: str,
    bq_uri: str,
    entity_id_column: str,
) -> None:
    """Create a feature view backed by a BQ table if it doesn't exist."""
    view_name = f"{store_name}/featureViews/{view_id}"
    try:
        client.get_feature_view(name=view_name)
        print(f"Feature view '{view_id}' already exists — skipping creation.")
        return
    except api_exceptions.NotFound:
        pass

    print(f"Creating feature view '{view_id}' → {bq_uri}...")
    op = client.create_feature_view(
        parent=store_name,
        feature_view_id=view_id,
        feature_view=FeatureView(
            big_query_source=FeatureView.BigQuerySource(
                uri=bq_uri,
                entity_id_columns=[entity_id_column],
            ),
        ),
    )
    op.result()
    print(f"Created feature view '{view_id}'.")


def setup(
    config_name: str = "iris",
    project: str = PROJECT,
    region: str = REGION,
) -> None:
    endpoint = f"{region}-aiplatform.googleapis.com"
    client = FeatureOnlineStoreAdminServiceClient(
        client_options={"api_endpoint": endpoint},
    )

    cfg = _load_config(config_name)
    bq_uri = f"bq://{project}.{cfg.bq_dataset}.{cfg.bq_feature_table}"

    store_name = _get_or_create_online_store(
        client, cfg.online_store_id, project, region
    )
    _get_or_create_feature_view(
        client, store_name, cfg.feature_view_id, bq_uri, cfg.entity_id_column
    )

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
