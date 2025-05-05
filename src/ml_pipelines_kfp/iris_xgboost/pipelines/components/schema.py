from kfp.dsl import Dataset, Input, Metrics, Model, Output, component, Artifact
from ml_pipelines_kfp.iris_xgboost.constants import IMAGE_NAME

@component(base_image=IMAGE_NAME,
           packages_to_install=["fsspec==2024.6.1","gcsfs==2024.6.1"])

def load_schema(
    repo_root: str,
    gcs_schema: Output[Artifact],
):
    import os
    import fsspec

    schema_path = "/schemas/iris_xgboost"

    fs, _ = fsspec.core.url_to_fs(gcs_schema.path)
    fs.makedirs(gcs_schema.path, exist_ok=True)

    # Write serving schema into serving model directory.
    with fs.open(os.path.join(gcs_schema.path, "instance.yaml"), "w") as f:
        with fsspec.open("schemas/iris_xgboost/vertex/instance.yaml", "r") as f2: #fsspec.open(os.path.join(repo_root, "schemas/iris_xgboost/vertex/instance.yaml"), "r") as f2:
            f.write(f2.read())

    with fs.open(os.path.join(gcs_schema.path, "prediction.yaml"), "w") as f:
        with fsspec.open("schemas/iris_xgboost/vertex/prediction.yaml", "r") as f2: #fsspec.open(os.path.join(repo_root, "schemas/iris_xgboost/vertex/prediction.yaml"), "r") as f2:
            f.write(f2.read())