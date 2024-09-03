from kfp.dsl import Input, Model, component

@component(base_image="python:3.10", 
    packages_to_install=["google-cloud-aiplatform==1.64.0"],
)
def upload_model(
    project_id: str,
    location: str,
    model: Input[Model],
    model_name: str,
    image_name: str
):
    from google.cloud import aiplatform, aiplatform_v1

    aiplatform.init(project=project_id, location=location)

    explanation_parameters =  aiplatform_v1.types.explanation.ExplanationParameters({"sampled_shapley_attribution": {"path_count": 10}})
    explanation_metadata = aiplatform_v1.types.explanation_metadata.ExplanationMetadata(
                            inputs = {
                                "SepalLengthCm": {},
                                "SepalWidthCm": {},
                                "PetalLengthCm": {},
                                "PetalWidthCm": {}
                            },
                            outputs = {
                                "Species": {}
                            }
                        )
    
    client = aiplatform_v1.ModelServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )

    request = {
        "parent": f"projects/{project_id}/locations/{location}",
        "filter": f"display_name={model_name}"
    }
    results = list(client.list_models(request=request))
    print(results)
    if results:
        parent_model = results[0]
    else:
        parent_model = None

    predict_schemata=None

    container_spec = aiplatform_v1.types.model.ModelContainerSpec(
        image_uri=image_name,
        #env=[{"name": key, "value": value] for key, value in env_vars.items()],
        #command=command,
        args=["uvicorn", "src.iris_xgboost.server:app", "--host", "0.0.0.0", "--port", "8080"],
        ports=[{"container_port": 8080}],
        predict_route="/predict",
        health_route="/health/live"
    )

    new_model = aiplatform_v1.Model(
                display_name=model_name,
                container_spec=container_spec,
                artifact_uri=model.path.replace('/gcs/','gs://'),
                # predict_schemata = predict_schemata
    )


    result = client.upload_model(
                    request = dict(
                        parent= f"projects/{project_id}/locations/{location}",
                        parent_model= parent_model.name if parent_model else None,
                        model=new_model,
                    )
            ).result()

    # aiplatform.Model.upload(
    #     artifact_uri=model.path.replace('/gcs/','gs://'),
    #     serving_container_image_uri=image_name,
    #     parent_model=parent_model.name if parent_model else None,
    #     display_name=model_name,
    #     project=project_id,
    #     explanation_parameters=explanation_parameters,
    #     explanation_metadata=explanation_metadata,
    # )

    # aiplatform.Model.upload_scikit_learn_model_file(
    #     model_file_path=model.path,
    #     parent_model=parent_model.name if parent_model else None,
    #     display_name=model_name,
    #     project=project_id,
    #     explanation_parameters=explanation_parameters,
    #     explanation_metadata=explanation_metadata,
    # )