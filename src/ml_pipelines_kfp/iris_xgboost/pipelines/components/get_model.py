from kfp.dsl import Input, Model, component, Output

@component(base_image="python:3.10", 
    packages_to_install=["google-cloud-aiplatform==1.64.0","fsspec==2024.6.1","gcsfs==2024.6.1","joblib==1.4.2",],
)
def get_model(
    project_id: str,
    location: str,
    model_name: str,
    latest_model: Output[Model],
):
    from google.cloud import aiplatform, aiplatform_v1
    import fsspec
    import gcsfs
    import joblib

    aiplatform.init(project=project_id, location=location)

    
    client = aiplatform_v1.ModelServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )

    request = {
        "parent": f"projects/{project_id}/locations/{location}",
        "filter": f"display_name={model_name}"
    }
    parent_models = list(client.list_models(request=request))
    parent_model = parent_models[0] if parent_models else None

    if not parent_model:
        print(f"Could not find model with f{model_name}")
        return
    
    print(f"Parent Model - {parent_model}")
    print(f"class - {type(parent_model)}")
    print(f"name - {parent_model.name}")
    print(f"model path- {parent_model.artifact_uri}")
    print(f"output model path - {latest_model.path}")
    latest_model_path = latest_model.path.replace("/gcs/", "gs://")
    print(f"output model path cleaned - {latest_model_path}")
    fs, _ = fsspec.core.url_to_fs(parent_model.artifact_uri)
    print(f"file system: {fs}")
    fs.copy(parent_model.artifact_uri+"/", latest_model.path.replace("/gcs/", "gs://"), recursive=True)