from datetime import datetime, timedelta
from airflow import DAG
from airflow.models.param import Param
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s

PROJECT_ID = "deeplearning-sahil"
REGION = "us-central1"
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
    dag_id="iris_batch_inference_staging",
    default_args=default_args,
    description="Batch inference using blessed Iris model (staging)",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "inference", "batch", "iris-staging"],
    params={
        "project_id": Param(PROJECT_ID, type="string", description="GCP project ID"),
        "region": Param(REGION, type="string", description="GCP region"),
        "bq_dataset": Param("ml_dataset", type="string"),
        "bq_feature_table": Param("iris_features", type="string"),
        "bq_table_predictions": Param("iris_predictions_staging", type="string",
                                      description="BQ table for predictions"),
    },
) as dag:

    run_inference_pipeline = KubernetesPodOperator(
        task_id="run_inference_pipeline",
        name="iris-inference-pipeline-staging",
        namespace="composer-user-workloads",
        image=f"{REGISTRY}/ml-pipelines-kfp-image:staging",
        cmds=["python", "-m", "ml_pipelines_kfp.iris_xgboost.pipelines.iris_pipeline_inference"],
        arguments=[
            "--project-id", "{{ params.project_id }}",
            "--region", "{{ params.region }}",
            "--bq-dataset", "{{ params.bq_dataset }}",
            "--bq-feature-table", "{{ params.bq_feature_table }}",
            "--bq-table-predictions", "{{ params.bq_table_predictions }}",
        ],
        env_vars=[
            k8s.V1EnvVar(name="ENVIRONMENT", value="staging"),
        ],
        startup_timeout_seconds=300,
        resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "1Gi"},
            limits={"cpu": "1", "memory": "2Gi"},
        ),
        is_delete_operator_pod=True,
        get_logs=True,
    )
