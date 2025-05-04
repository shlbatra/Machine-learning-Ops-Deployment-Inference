import os
from pathlib import Path
import pandas as pd
from google.cloud import bigquery

def load_iris_to_bigquery():
    # Read CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'data', 'iris.csv')
    df = pd.read_csv(csv_path)
    
    # Initialize BigQuery client
    client = bigquery.Client(project="deeplearning-sahil")
    
    # Configure the job
    dataset_id = 'ml_dataset'  # Using the dataset name from pipeline configuration
    table_id = 'iris'
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    
    # Create dataset if it doesn't exist
    dataset = bigquery.Dataset(f"{client.project}.{dataset_id}")
    dataset.location = "US"
    
    try:
        dataset = client.create_dataset(dataset, exists_ok=True)
        print(f"Dataset {dataset_id} ready")
    except Exception as e:
        print(f"Error with dataset: {str(e)}")
        raise
    
    # Format column names to match BigQuery standards
    df.columns = df.columns.str.replace('.', '_')  # Replace dots with underscores
    df['variety'] = df['variety'].replace({
        "Setosa": "Iris-setosa",
        "Versicolor": "Iris-versicolor",
        "Virginica": "Iris-virginica"
    })
    
    # Load data to BigQuery
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    
    try:
        load_job = client.load_table_from_dataframe(
            df,
            table_ref,
            job_config=job_config
        )
        # Wait for the load job to complete
        load_job.result()
        print(f"Successfully loaded {len(df)} rows to BigQuery table {table_ref}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

if __name__ == '__main__':
    load_iris_to_bigquery()