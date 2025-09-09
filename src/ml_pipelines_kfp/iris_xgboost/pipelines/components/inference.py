
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

@component(base_image="python:3.10", 
    packages_to_install=[
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.5.1",
        "joblib==1.4.2",
        "google-cloud-bigquery==3.11.4",
        "pyarrow==12.0.1",
        "db-dtypes==1.1.1"
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

    client = bigquery.Client(project=project_id)

    dataset_ref = bigquery.DatasetReference(project_id, bq_dataset)
    table_ref = dataset_ref.table(bq_table)
    table = bigquery.Table(table_ref)
    iterable_table = client.list_rows(table).to_dataframe_iterable()

    dfs = []
    for row in iterable_table:
        dfs.append(row)

    df = pd.concat(dfs, ignore_index=True)
    print(df.head())
    print(df.columns)

    if bq_table == 'iris_pubsub_data':
        df_cols = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].rename(columns={'sepal_length': 'SepalLengthCm', 'sepal_width': 'SepalWidthCm', 'petal_length': 'PetalLengthCm', 'petal_width': 'PetalWidthCm'})
    else:
        df_cols = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

    print(f"Model Path: {model.path}")
    inf_model = joblib.load(model.path+'/model.joblib')
    inf_pred = inf_model.predict(df_cols)
    print(len(inf_pred))
    print(inf_pred[:5])
    
    # Create predictions dataframe
    predictions_df = df.copy()
    predictions_df['prediction'] = inf_pred
    predictions_df['prediction_timestamp'] = datetime.now()
    predictions_df['model_path'] = model.path
    print(len(predictions_df))
    # Write predictions to BigQuery
    predictions_table_id = f"{project_id}.{bq_dataset}.{bq_table_predictions}"
    print(predictions_table_id)

    # try:
    #     client.get_table(table_ref)
    #     # Table exists, use WRITE_APPEND
    #     write_disposition = "WRITE_APPEND"
    # except:
    #     # Table doesn't exist, use WRITE_TRUNCATE to create it
    #     write_disposition = "WRITE_TRUNCATE"
    #print(write_disposition)

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE"
        #schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    
    job = client.load_table_from_dataframe(
        predictions_df, predictions_table_id, job_config=job_config
    )
    job.result()  # Wait for the job to complete
    
    print(f"Loaded {len(predictions_df)} rows to {predictions_table_id}")