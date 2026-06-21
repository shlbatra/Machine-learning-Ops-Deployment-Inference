from datetime import datetime, timedelta
from airflow import DAG
from airflow.models.param import Param
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

PROJECT_ID = "deeplearning-sahil"
REGION = "us-central1"
BUCKET = "gs://sb-vertex"
SERVICE_ACCOUNT = "kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"
REGISTRY = f"us-docker.pkg.dev/{PROJECT_ID}/sahil-experiment-docker-images"

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["shlbatra123bot@gmail.com"],
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="iris_training_staging",
    default_args=default_args,
    description="Train Iris classifier (staging)",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "training", "iris-staging"],
    params={
        "project_id": Param(PROJECT_ID, type="string", description="GCP project ID"),
        "region": Param(REGION, type="string", description="GCP region"),
        "bq_dataset": Param("ml_dataset", type="string"),
        "bq_table": Param("iris", type="string"),
        "bq_feature_table": Param("iris_features", type="string"),
        "service_account": Param(SERVICE_ACCOUNT, type="string"),
    },
) as dag:

    run_training_pipeline = KubernetesPodOperator(
        task_id="run_training_pipeline",
        name="iris-training-pipeline-staging",
        namespace="composer-user-workloads",
        image=f"{REGISTRY}/ml-pipelines-kfp-image:staging",
        cmds=["python", "-m", "ml_pipelines_kfp.iris_xgboost.pipelines.iris_pipeline_training"],
        arguments=[
            "--project-id", "{{ params.project_id }}",
            "--region", "{{ params.region }}",
            "--pipeline-name", "pipeline-iris-staging",
            "--pipeline-root", f"{BUCKET}/staging/pipeline_root",
            "--model-name", "Iris-Classifier-XGBoost-staging",
            "--image-name", f"{REGISTRY}/ml-pipelines-kfp-image:staging",
            "--fastapi-image-name", f"{REGISTRY}/fastapi-ml-generic:staging",
            "--bq-dataset", "{{ params.bq_dataset }}",
            "--bq-table", "{{ params.bq_table }}",
            "--bq-feature-table", "{{ params.bq_feature_table }}",
            "--service-account", "{{ params.service_account }}",
        ],
        startup_timeout_seconds=300,
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "1Gi"},
            limits={"cpu": "1", "memory": "2Gi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )
