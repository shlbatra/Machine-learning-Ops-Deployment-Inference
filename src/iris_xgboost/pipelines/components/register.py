from kfp.dsl import Input, Model, component

@component(
    packages_to_install=["google-cloud-aiplatform"],
)
def upload_model(
    project_id: str,
    location: str,
    model: Input[Model],
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    aiplatform.Model.upload_scikit_learn_model_file(
        model_file_path=model.path,
        display_name="Iris-Classifier",
        project=project_id,
    )