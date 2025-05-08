from kfp.dsl import Input, Model, component, Artifact
from ml_pipelines_kfp.iris_xgboost.constants import IMAGE_NAME

@component(
    base_image=IMAGE_NAME, 
    packages_to_install=[
        "google-cloud-aiplatform",
        "pandas==2.0.0",
        "scikit-learn==1.5.1",
        "numpy==1.23.0",
        "joblib==1.4.2"
    ]
)
def deploy_model(
    project_id: str,
    location: str,
    model: Input[Model],
    vertex_model: Input[Artifact],
    endpoint_name: str,
    model_name: str
):
    from google.cloud import aiplatform, aiplatform_v1
    import pandas
    import numpy
    import joblib

    print(f"Pandas version: {pandas.__version__}")
    print(f"NumPy version: {numpy.__version__}")
    print(f"Joblib version: {joblib.__version__}")

    aiplatform.init(project=project_id, location=location)

    client = aiplatform_v1.ModelServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    request = {
        "parent": f"projects/{project_id}/locations/{location}",
        "filter": f"display_name={model_name}"
    }

    parent_models = list(client.list_models(request=request))
    print(f"Parent models: {parent_models}")

    parent_model = parent_models[0] if parent_models else None
    if not parent_model:
        raise ValueError("No parent model found with the specified name.")

    model_name = parent_model.name.split('/')[-1]
    model = aiplatform.Model(model_name=model_name)

    print(f"Model type: {type(model)}")
    print(f"Model details: {model}")

    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    print(f"Endpoint created: {endpoint}")

    endpoint.deploy(
        model=model,
        machine_type="n1-standard-2",
        traffic_percentage=100
    )

    print(f"Model deployed to endpoint {endpoint.display_name}")

    print("Cleaning up legacy deployments with no traffic assigned...")
    traffic_split = endpoint.traffic_split
    for deployed_model in endpoint.list_models():
        print(f"Checking deployed model: {deployed_model.id}@{deployed_model.model_version_id}")
        if deployed_model.id not in traffic_split or traffic_split[deployed_model.id] == 0:
            try:
                endpoint.undeploy(deployed_model.id)
                print(f"Successfully undeployed model {deployed_model.id}@{deployed_model.model_version_id}")
            except Exception as e:
                print(f"Failed to undeploy model {deployed_model.id}@{deployed_model.model_version_id}: {e}")