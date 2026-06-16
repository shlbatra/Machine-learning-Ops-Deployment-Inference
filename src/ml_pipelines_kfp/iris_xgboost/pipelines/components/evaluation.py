from kfp.dsl import Dataset, Input, Metrics, Model, Output, component
import ml_pipelines_kfp.iris_xgboost.constants as _constants


@component(base_image=_constants.IMAGE_NAME)
def choose_best_model(
    test_dataset: Input[Dataset],
    decision_tree_model: Input[Model],
    random_forest_model: Input[Model],
    metrics: Output[Metrics],
    best_model: Output[Model],
):
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import os, pickle, fsspec, gcsfs
    from ml_pipelines_kfp.log import get_logger

    logger = get_logger(__name__)

    test_data = pd.read_csv(test_dataset.path)

    dt = joblib.load(decision_tree_model.path)
    rf = joblib.load(random_forest_model.path)

    dt_pred = dt.predict(test_data.drop("species", axis=1))
    rf_pred = rf.predict(test_data.drop("species", axis=1))

    dt_accuracy = accuracy_score(test_data["species"], dt_pred)
    rf_accuracy = accuracy_score(test_data["species"], rf_pred)

    metrics.log_metric("Decision Tree (Accuracy)", (dt_accuracy))
    metrics.log_metric("Random Forest (Accuracy)", (rf_accuracy))

    filepath = best_model.path.replace("/gcs/", "gs://")
    filename = "model.joblib"
    fs, _ = fsspec.core.url_to_fs(filepath)
    fs.makedirs(filepath, exist_ok=True)
    model_uri = os.path.join(filepath, filename)
    with fs.open(model_uri, "wb") as f:
        if rf_accuracy >= dt_accuracy:
            joblib.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Selected Random Forest model with accuracy: {rf_accuracy}")
        else:
            joblib.dump(dt, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Selected Decision Tree model with accuracy: {dt_accuracy}")
