"""Ingest raw BQ tables into the canonical feature table.

Reads from ml_dataset.iris (labeled training data) and
ml_dataset.iris_batch_input (unlabeled inference data), renames columns
to canonical names, adds entity_id and feature_timestamp, and writes to
ml_dataset.iris_features with WRITE_TRUNCATE (full refresh each run).

Run independently before the KFP pipeline:
    python -m feature_store.ingest
"""

from __future__ import annotations

import argparse

import pandas as pd
from google.cloud import bigquery

from feature_store.iris.feature_definitions import IRIS_CONFIG

PROJECT = "deeplearning-sahil"


def _read_table(client: bigquery.Client, table_ref: str) -> pd.DataFrame | None:
    """Read a BQ table, returning None if the table doesn't exist."""
    try:
        return client.query(f"SELECT * FROM `{table_ref}`").result().to_dataframe()
    except Exception:
        return None


def _rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns present in *mapping*; pass through unmapped columns."""
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def ingest(project: str = PROJECT) -> None:
    client = bigquery.Client(project=project)
    cfg = IRIS_CONFIG
    camel_map = cfg.column_mappings["camel"]

    raw_ref = f"{project}.{cfg.bq_dataset}.{cfg.bq_raw_table}"
    batch_ref = f"{project}.{cfg.bq_dataset}.{cfg.bq_batch_input_table}"
    feature_ref = f"{project}.{cfg.bq_dataset}.{cfg.bq_feature_table}"

    frames: list[pd.DataFrame] = []

    raw_df = _read_table(client, raw_ref)
    if raw_df is not None and not raw_df.empty:
        raw_df = _rename_columns(raw_df, camel_map)
        frames.append(raw_df)

    batch_df = _read_table(client, batch_ref)
    if batch_df is not None and not batch_df.empty:
        batch_df = _rename_columns(batch_df, camel_map)
        frames.append(batch_df)

    if not frames:
        print("No source data found — nothing to ingest.")
        return

    combined = pd.concat(frames, ignore_index=True)

    if "load_timestamp" in combined.columns:
        combined[cfg.feature_timestamp_column] = pd.to_datetime(
            combined["load_timestamp"], utc=True
        )
    else:
        combined[cfg.feature_timestamp_column] = pd.Timestamp.now(tz="UTC")

    combined[cfg.entity_id_column] = (
        combined["Id"].astype(int).astype(str) + "_" + combined["source"]
    )

    keep_cols = list(cfg.feature_columns)
    if cfg.target_column in combined.columns:
        keep_cols.append(cfg.target_column)
    keep_cols.extend(["source", cfg.entity_id_column, cfg.feature_timestamp_column])

    combined = combined[[c for c in keep_cols if c in combined.columns]]

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    load_job = client.load_table_from_dataframe(
        combined, feature_ref, job_config=job_config
    )
    load_job.result()

    print(f"Wrote {len(combined)} rows to {feature_ref}")
    print(f"Columns: {list(combined.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest raw BQ data into canonical feature table",
    )
    parser.add_argument(
        "--project",
        default=PROJECT,
        help="GCP project ID (default: %(default)s)",
    )
    args = parser.parse_args()
    ingest(project=args.project)


if __name__ == "__main__":
    main()
