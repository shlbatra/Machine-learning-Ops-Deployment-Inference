import argparse
import os
import random
from datetime import datetime, timezone

import pandas as pd
from google.cloud import bigquery
from ml_pipelines_kfp.log import get_logger

logger = get_logger(__name__)

PROJECT = "deeplearning-sahil"
DATASET = "ml_dataset"
TABLE = "iris"
BATCH_INPUT_TABLE = "iris_batch_input"

FEATURE_RANGES = {
    "SepalLengthCm": (4.3, 7.9),
    "SepalWidthCm": (2.0, 4.4),
    "PetalLengthCm": (1.0, 6.9),
    "PetalWidthCm": (0.1, 2.5),
}


def _table_ref() -> str:
    return f"{PROJECT}.{DATASET}.{TABLE}"


def load_iris_to_bigquery():
    """Load the original 150 labeled iris rows (WRITE_TRUNCATE)."""
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "src",
        "ml_pipelines_kfp",
        "iris_xgboost",
        "data",
        "iris.csv",
    )
    df = pd.read_csv(csv_path)

    df["Species"] = df["Species"].replace(
        {
            "Setosa": "Iris-setosa",
            "Versicolor": "Iris-versicolor",
            "Virginica": "Iris-virginica",
        }
    )
    df.insert(0, "Id", range(1, len(df) + 1))
    df["load_timestamp"] = datetime.now(timezone.utc)

    client = bigquery.Client(project=PROJECT)
    dataset = bigquery.Dataset(f"{PROJECT}.{DATASET}")
    dataset.location = "US"

    try:
        client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset {DATASET} ready")
    except Exception as e:
        logger.error(f"Error with dataset: {str(e)}")
        raise

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    try:
        load_job = client.load_table_from_dataframe(
            df, _table_ref(), job_config=job_config
        )
        load_job.result()
        logger.info(f"Loaded {len(df)} labeled rows to {_table_ref()}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def _batch_input_table_ref() -> str:
    return f"{PROJECT}.{DATASET}.{BATCH_INPUT_TABLE}"


def generate_random_iris_data(n: int):
    """Generate N random unlabeled iris rows to a separate batch input table.

    Rows are appended to ml_dataset.iris_batch_input with WRITE_APPEND so
    data accumulates across daily runs. Id values continue from the current
    max in the table.
    """
    client = bigquery.Client(project=PROJECT)

    table_ref = _batch_input_table_ref()
    query = f"SELECT COALESCE(MAX(Id), 0) AS max_id FROM `{table_ref}`"
    try:
        max_id = next(iter(client.query(query).result())).max_id
    except Exception:
        max_id = 0
    logger.info(f"Current max Id in {BATCH_INPUT_TABLE}: {max_id}")

    rows = []
    for i in range(n):
        row = {"Id": max_id + i + 1}
        for col, (lo, hi) in FEATURE_RANGES.items():
            row[col] = round(random.uniform(lo, hi), 1)
        row["Species"] = None
        row["load_timestamp"] = datetime.now(timezone.utc)
        rows.append(row)

    df = pd.DataFrame(rows)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    try:
        load_job = client.load_table_from_dataframe(
            df, _batch_input_table_ref(), job_config=job_config
        )
        load_job.result()
        logger.info(f"Appended {n} random unlabeled rows (Id {max_id + 1}–{max_id + n}) to {_batch_input_table_ref()}")
    except Exception as e:
        logger.error(f"Error writing batch input data: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Load iris data to BigQuery")
    parser.add_argument(
        "--generate-random",
        type=int,
        metavar="N",
        help="Append N random unlabeled rows for inference (skips base load)",
    )
    args = parser.parse_args()

    if args.generate_random:
        generate_random_iris_data(args.generate_random)
    else:
        load_iris_to_bigquery()


if __name__ == "__main__":
    main()
