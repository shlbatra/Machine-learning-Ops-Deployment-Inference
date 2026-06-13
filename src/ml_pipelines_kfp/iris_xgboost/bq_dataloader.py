import os
from pathlib import Path
import pandas as pd
from google.cloud import bigquery
from ml_pipelines_kfp.log import get_logger

logger = get_logger(__name__)


def load_iris_to_bigquery():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data", "iris.csv")
    df = pd.read_csv(csv_path)

    client = bigquery.Client(project="deeplearning-sahil")

    dataset_id = "ml_dataset"
    table_id = "iris"
    table_ref = f"{client.project}.{dataset_id}.{table_id}"

    dataset = bigquery.Dataset(f"{client.project}.{dataset_id}")
    dataset.location = "US"

    try:
        dataset = client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset {dataset_id} ready")
    except Exception as e:
        logger.error(f"Error with dataset: {str(e)}")
        raise

    df.columns = df.columns.str.replace(".", "_")
    df["Species"] = df["Species"].replace(
        {
            "Setosa": "Iris-setosa",
            "Versicolor": "Iris-versicolor",
            "Virginica": "Iris-virginica",
        }
    )

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    try:
        load_job = client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )
        load_job.result()
        logger.info(f"Successfully loaded {len(df)} rows to BigQuery table {table_ref}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


if __name__ == "__main__":
    load_iris_to_bigquery()
