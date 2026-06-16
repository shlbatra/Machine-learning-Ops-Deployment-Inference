"""Sync Feature Store online store with the latest BQ feature data.

Triggers a manual FeatureView sync so the online store (Bigtable) reflects
the latest rows in the canonical BQ feature table. Run after ingest.py.

Usage:
    python -m feature_store.sync                  # defaults to iris
    python -m feature_store.sync --config fraud   # different project
"""

from __future__ import annotations

import argparse

from google.cloud.aiplatform_v1 import FeatureOnlineStoreAdminServiceClient

from feature_store.setup import CONFIGS, PROJECT, REGION, _load_config


def sync(
    config_name: str = "iris",
    project: str = PROJECT,
    region: str = REGION,
) -> None:
    endpoint = f"{region}-aiplatform.googleapis.com"
    client = FeatureOnlineStoreAdminServiceClient(
        client_options={"api_endpoint": endpoint},
    )

    cfg = _load_config(config_name)
    feature_view_name = (
        f"projects/{project}/locations/{region}"
        f"/featureOnlineStores/{cfg.online_store_id}"
        f"/featureViews/{cfg.feature_view_id}"
    )

    print(f"Syncing feature view '{cfg.feature_view_id}'...")
    response = client.sync_feature_view(feature_view=feature_view_name)
    print(f"Sync started: {response.feature_view_sync}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync Feature Store online store from BQ feature table",
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
    sync(config_name=args.config, project=args.project, region=args.region)


if __name__ == "__main__":
    main()
