from kfp.dsl import Dataset, Input, Metrics, Model, Output, component, Artifact
import ml_pipelines_kfp.iris_xgboost.constants as _constants


@component(
    base_image=_constants.IMAGE_NAME,
)
def load_schema(
    repo_root: str,
    gcs_schema: Output[Artifact],
):
    import os
    import fsspec

    schema_path = "src/ml_pipelines_kfp/schemas/iris_xgboost"

    fs, _ = fsspec.core.url_to_fs(gcs_schema.path)
    fs.makedirs(gcs_schema.path, exist_ok=True)

    with fs.open(os.path.join(gcs_schema.path, "instance.yaml"), "w") as f:
        with fsspec.open(
            "src/ml_pipelines_kfp/schemas/iris_xgboost/vertex/instance.yaml", "r"
        ) as f2:
            f.write(f2.read())

    with fs.open(os.path.join(gcs_schema.path, "prediction.yaml"), "w") as f:
        with fsspec.open(
            "src/ml_pipelines_kfp/schemas/iris_xgboost/vertex/prediction.yaml", "r"
        ) as f2:
            f.write(f2.read())
