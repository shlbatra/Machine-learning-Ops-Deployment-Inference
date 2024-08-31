from kfp.dsl import Input, Model, component

@component(base_image="python:3.9", 
    packages_to_install=["google-cloud-aiplatform"],
)
def upload_model(
    project_id: str,
    location: str,
    model: Input[Model],
    model_name: str
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    aiplatform.Model.upload_scikit_learn_model_file(
        model_file_path=model.path,
        display_name=model_name,
        project=project_id,
    )