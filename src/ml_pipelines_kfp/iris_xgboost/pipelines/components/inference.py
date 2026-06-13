from kfp.dsl import Dataset, Input, Metrics, Model, Output, component
import ml_pipelines_kfp.iris_xgboost.constants as _constants


@component(base_image=_constants.IMAGE_NAME)
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
    from ml_pipelines_kfp.log import get_logger

    logger = get_logger(__name__)

    client = bigquery.Client(project=project_id)

    dataset_ref = bigquery.DatasetReference(project_id, bq_dataset)
    table_ref = dataset_ref.table(bq_table)
    table = bigquery.Table(table_ref)
    iterable_table = client.list_rows(table).to_dataframe_iterable()

    dfs = []
    for row in iterable_table:
        dfs.append(row)

    df = pd.concat(dfs, ignore_index=True)

    if bq_table == "iris_pubsub_data":
        df_cols = df[
            ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        ].rename(
            columns={
                "sepal_length": "SepalLengthCm",
                "sepal_width": "SepalWidthCm",
                "petal_length": "PetalLengthCm",
                "petal_width": "PetalWidthCm",
            }
        )
    else:
        df_cols = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

    inf_model = joblib.load(model.path + "/model.joblib")
    inf_pred = inf_model.predict(df_cols)

    predictions_df = df.copy()
    predictions_df["prediction"] = inf_pred
    predictions_df["prediction_timestamp"] = datetime.now()
    predictions_df["model_path"] = model.path

    predictions_table_id = f"{project_id}.{bq_dataset}.{bq_table_predictions}"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE"
    )

    job = client.load_table_from_dataframe(
        predictions_df, predictions_table_id, job_config=job_config
    )
    job.result()

    logger.info(f"Loaded {len(predictions_df)} rows to {predictions_table_id}")
