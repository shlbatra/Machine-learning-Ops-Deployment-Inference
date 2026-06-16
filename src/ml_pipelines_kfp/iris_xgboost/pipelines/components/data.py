from kfp.dsl import Dataset, Output, component
import ml_pipelines_kfp.iris_xgboost.constants as _constants


@component(base_image=_constants.IMAGE_NAME)
def load_data(
    project_id: str,
    bq_dataset: str,
    bq_table: str,
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    import pandas as pd
    from google.cloud import bigquery
    from sklearn.model_selection import train_test_split
    from ml_pipelines_kfp.log import get_logger

    logger = get_logger(__name__)

    logger.info(f"Setup loading data from {bq_dataset}.{bq_table}")

    client = bigquery.Client()

    dataset_ref = bigquery.DatasetReference(project_id, bq_dataset)
    table_ref = dataset_ref.table(bq_table)
    table = bigquery.Table(table_ref)
    iterable_table = client.list_rows(table).to_dataframe_iterable()

    dfs = []
    for row in iterable_table:
        dfs.append(row)

    df = pd.concat(dfs, ignore_index=True)
    del dfs

    df["Species"].replace(
        {
            "Iris-versicolor": 0,
            "Iris-virginica": 1,
            "Iris-setosa": 2,
        },
        inplace=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["Species"], axis=1),
        df["Species"],
        test_size=0.2,
        random_state=42,
    )

    X_train["Species"] = y_train
    X_test["Species"] = y_test

    X_train.to_csv(f"{train_dataset.path}", index=False)
    X_test.to_csv(f"{test_dataset.path}", index=False)


@component(base_image=_constants.IMAGE_NAME)
def load_data_from_feature_store(
    project_id: str,
    bq_dataset: str,
    bq_feature_table: str,
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    import pandas as pd
    from google.cloud import bigquery
    from sklearn.model_selection import train_test_split
    from ml_pipelines_kfp.log import get_logger

    logger = get_logger(__name__)

    table_ref = f"{project_id}.{bq_dataset}.{bq_feature_table}"
    logger.info(f"Loading training data from feature table {table_ref}")

    client = bigquery.Client()

    dataset_ref = bigquery.DatasetReference(project_id, bq_dataset)
    table = bigquery.Table(dataset_ref.table(bq_feature_table))
    df = client.list_rows(table).to_dataframe()

    df = df[df["species"].notna()]
    logger.info(f"Loaded {len(df)} labeled rows from feature store")

    df = df.drop(
        columns=["entity_id", "feature_timestamp", "source"],
        errors="ignore",
    )

    df["species"].replace(
        {
            "Iris-versicolor": 0,
            "Iris-virginica": 1,
            "Iris-setosa": 2,
        },
        inplace=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["species"], axis=1),
        df["species"],
        test_size=0.2,
        random_state=42,
    )

    X_train["species"] = y_train
    X_test["species"] = y_test

    X_train.to_csv(f"{train_dataset.path}", index=False)
    X_test.to_csv(f"{test_dataset.path}", index=False)
