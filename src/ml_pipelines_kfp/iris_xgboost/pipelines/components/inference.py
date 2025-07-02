
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

@component(base_image="python:3.10", 
    packages_to_install=[
        "pandas==2.0.0",
        "scikit-learn==1.5.1",
        "numpy==1.23.0",
        "joblib==1.4.2",
        "google-cloud-bigquery==2.34.3"
    ],
)
def inference_model(
    project_id: str,
    location: str,
    bq_dataset: str,
    bq_table: str,
    bq_table_predictions: str,
    model: Input[Model],
):
    import joblib
    import pandas as pd
    import numpy as np
    from google.cloud import bigquery
    from datetime import datetime

    client = bigquery.Client()

    dataset_ref = bigquery.DatasetReference(project_id, bq_dataset)
    table_ref = dataset_ref.table(bq_table)
    table = bigquery.Table(table_ref)
    iterable_table = client.list_rows(table).to_dataframe_iterable()

    dfs = []
    for row in iterable_table:
        dfs.append(row)

    df = pd.concat(dfs, ignore_index=True)
    print("Model Path: f{model.path}")
    inf_model = joblib.load(model.path+'/model.joblib')
    inf_pred = inf_model.predict(df)
    print(len(inf_pred))
    print(inf_pred[:5])
    
    # Create predictions dataframe
    predictions_df = df.copy()
    predictions_df['prediction'] = inf_pred
    predictions_df['prediction_timestamp'] = datetime.now()
    predictions_df['model_path'] = model.path
    
    # Write predictions to BigQuery
    predictions_table_id = f"{project_id}.{bq_dataset}.{bq_table_predictions}"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    
    job = client.load_table_from_dataframe(
        predictions_df, predictions_table_id, job_config=job_config
    )
    job.result()  # Wait for the job to complete
    
    print(f"Loaded {len(predictions_df)} rows to {predictions_table_id}")