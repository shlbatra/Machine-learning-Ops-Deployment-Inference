from kfp.dsl import Dataset, Input, Metrics, Model, Output, component
import ml_pipelines_kfp.iris_xgboost.constants as _constants


@component(base_image=_constants.IMAGE_NAME)
def inference_model(
    project_id: str,
    location: str,
    bq_dataset: str,
    bq_feature_table: str,
    bq_table_predictions: str,
    model: Input[Model],
):
    import joblib
    from google.cloud import bigquery
    from datetime import datetime
    from ml_pipelines_kfp.log import get_logger

    logger = get_logger(__name__)

    client = bigquery.Client(project=project_id)

    query = f"""
        SELECT * FROM `{project_id}.{bq_dataset}.{bq_feature_table}`
        WHERE source = 'batch_input'
    """
    df = client.query(query).result().to_dataframe()

    logger.info(f"Loaded {len(df)} batch inference rows from feature store")

    feature_cols = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    ]
    df_cols = df[feature_cols]

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
