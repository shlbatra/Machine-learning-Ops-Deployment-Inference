#!/bin/bash

# Setup script for Cloud Composer 2 (managed Airflow on GKE)
# Creates the environment, grants required IAM roles, and configures Workload Identity

set -e

# Configuration
PROJECT_ID="deeplearning-sahil"
REGION="us-central1"
ENVIRONMENT_NAME="ml-pipelines-composer"
SERVICE_ACCOUNT="kfp-mlops@deeplearning-sahil.iam.gserviceaccount.com"
COMPOSER_IMAGE_VERSION="composer-2.17.3-airflow-2.10.5"
ENVIRONMENT_SIZE="small"
KSA_NAMESPACE="composer-user-workloads"
KSA_NAME="default"
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
COMPOSER_AGENT="service-${PROJECT_NUMBER}@cloudcomposer-accounts.iam.gserviceaccount.com"

echo "=== Cloud Composer 2 Setup ==="
echo ""

# --- 1. Enable required APIs ---
echo "Enabling required APIs..."
gcloud services enable \
    composer.googleapis.com \
    container.googleapis.com \
    cloudbuild.googleapis.com \
    --project=$PROJECT_ID

# --- 1b. Grant Composer Service Agent the V2 extension role ---
# Google auto-creates this agent when the Composer API is enabled.
# Composer 2 requires this additional role to manage IAM bindings during environment creation.
echo ""
echo "Granting Composer Service Agent V2 extension role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$COMPOSER_AGENT" \
    --role="roles/composer.ServiceAgentV2Ext" \
    --condition=None \
    --quiet

# --- 2. Create Composer 2 environment ---
echo ""
echo "Checking if Composer environment already exists..."
ENV_STATE=$(gcloud composer environments describe $ENVIRONMENT_NAME \
    --location=$REGION \
    --project=$PROJECT_ID \
    --format="value(state)" 2>/dev/null) || true

if [ "$ENV_STATE" = "RUNNING" ]; then
    echo "Environment $ENVIRONMENT_NAME already exists and is RUNNING — skipping creation."
elif [ -n "$ENV_STATE" ]; then
    echo "Environment $ENVIRONMENT_NAME exists in state: $ENV_STATE"
    echo "Wait for it to reach RUNNING before continuing."
    exit 1
else
    echo "Creating Composer 2 environment: $ENVIRONMENT_NAME"
    gcloud composer environments create $ENVIRONMENT_NAME \
        --location=$REGION \
        --environment-size=$ENVIRONMENT_SIZE \
        --image-version=$COMPOSER_IMAGE_VERSION \
        --service-account=$SERVICE_ACCOUNT \
        --project=$PROJECT_ID
fi

# --- 3. Grant IAM roles to service account ---
# The kfp-mlops@ SA already has most roles from pipeline setup.
# These ensure Composer-specific roles are present.
ROLES=(
    "roles/composer.worker"
    "roles/container.developer"
    "roles/artifactregistry.reader"
    "roles/bigquery.dataEditor"
    "roles/storage.objectAdmin"
    "roles/aiplatform.user"
    "roles/run.admin"
)

echo ""
echo "Granting IAM roles to $SERVICE_ACCOUNT..."
for ROLE in "${ROLES[@]}"; do
    echo "  Granting $ROLE"
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="$ROLE" \
        --condition=None \
        --quiet
done

# --- 4. Configure Workload Identity ---
# Allows KPO pods in the composer-user-workloads namespace to authenticate
# as the GCP service account via the GKE metadata server (no key files needed).
#
# The Workload Identity pool (PROJECT_ID.svc.id.goog) only exists once it is
# enabled on the GKE cluster. We must fetch the cluster, enable WI if needed,
# then create the IAM binding.
echo ""
echo "Configuring Workload Identity..."

echo "  Fetching GKE cluster from Composer environment..."
GKE_CLUSTER=$(gcloud composer environments describe $ENVIRONMENT_NAME \
    --location=$REGION \
    --project=$PROJECT_ID \
    --format="value(config.gkeCluster)" 2>/dev/null)

if [ -n "$GKE_CLUSTER" ]; then
    CLUSTER_NAME=$(basename "$GKE_CLUSTER")
    CLUSTER_LOCATION=$(echo "$GKE_CLUSTER" | sed -n 's|.*/locations/\([^/]*\)/.*|\1|p')

    echo "  Cluster: $CLUSTER_NAME ($CLUSTER_LOCATION)"

    echo "  Enabling Workload Identity on cluster..."
    gcloud container clusters update "$CLUSTER_NAME" \
        --region="$CLUSTER_LOCATION" \
        --project=$PROJECT_ID \
        --workload-pool="${PROJECT_ID}.svc.id.goog" \
        --quiet \
        || echo "  Workload Identity already enabled"

    echo "  Binding K8s SA to GCP SA..."
    gcloud iam service-accounts add-iam-policy-binding $SERVICE_ACCOUNT \
        --role=roles/iam.workloadIdentityUser \
        --member="serviceAccount:${PROJECT_ID}.svc.id.goog[${KSA_NAMESPACE}/${KSA_NAME}]" \
        --project=$PROJECT_ID \
        --quiet

    echo "  Fetching cluster credentials..."
    gcloud container clusters get-credentials "$CLUSTER_NAME" \
        --region="$CLUSTER_LOCATION" \
        --project=$PROJECT_ID

    echo "  Annotating K8s service account..."
    kubectl annotate serviceaccount "$KSA_NAME" \
        --namespace="$KSA_NAMESPACE" \
        iam.gke.io/gcp-service-account=$SERVICE_ACCOUNT \
        --overwrite

    # --- 5. Grant Airflow SA RBAC to manage pods in composer-user-workloads ---
    # The Airflow scheduler runs in its own namespace and needs permission to
    # create/list/delete pods in composer-user-workloads for KubernetesPodOperator.
    echo ""
    echo "Configuring RBAC for KubernetesPodOperator..."

    AIRFLOW_NAMESPACE=$(kubectl get namespaces -o name \
        | grep -oP '(?<=namespace/)composer-.*' \
        | grep -v composer-user-workloads \
        | head -1)

    if [ -n "$AIRFLOW_NAMESPACE" ]; then
        echo "  Airflow namespace: $AIRFLOW_NAMESPACE"

        kubectl create role pod-manager \
            --namespace="$KSA_NAMESPACE" \
            --verb=create,get,list,watch,delete,patch \
            --resource=pods,pods/log,pods/status \
            --dry-run=client -o yaml | kubectl apply -f -

        kubectl create rolebinding airflow-pod-manager \
            --namespace="$KSA_NAMESPACE" \
            --role=pod-manager \
            --serviceaccount="${AIRFLOW_NAMESPACE}:default" \
            --dry-run=client -o yaml | kubectl apply -f -

        echo "  RBAC configured: ${AIRFLOW_NAMESPACE}:default → pod-manager in $KSA_NAMESPACE"
    else
        echo "  WARNING: Could not detect Airflow namespace."
        echo "  Create RBAC manually after identifying the namespace:"
        echo ""
        echo "    kubectl get namespaces | grep composer"
    fi
else
    echo "  WARNING: Could not fetch GKE cluster info — Composer environment may still be creating."
    echo "  Re-run this script after the environment is ready, or run these commands manually:"
    echo ""
    echo "    # 1. Enable Workload Identity on the Composer GKE cluster"
    echo "    gcloud container clusters update <CLUSTER_NAME> \\"
    echo "      --region=$REGION --project=$PROJECT_ID \\"
    echo "      --workload-pool=${PROJECT_ID}.svc.id.goog"
    echo ""
    echo "    # 2. Bind K8s SA to GCP SA"
    echo "    gcloud iam service-accounts add-iam-policy-binding $SERVICE_ACCOUNT \\"
    echo "      --role=roles/iam.workloadIdentityUser \\"
    echo "      --member='serviceAccount:${PROJECT_ID}.svc.id.goog[${KSA_NAMESPACE}/${KSA_NAME}]'"
    echo ""
    echo "    # 3. Annotate K8s service account"
    echo "    kubectl annotate serviceaccount $KSA_NAME \\"
    echo "      --namespace=$KSA_NAMESPACE \\"
    echo "      iam.gke.io/gcp-service-account=$SERVICE_ACCOUNT \\"
    echo "      --overwrite"
fi

# --- 5. Get the DAGs bucket for later use ---
echo ""
echo "Fetching Composer DAGs bucket..."
DAGS_BUCKET=$(gcloud composer environments describe $ENVIRONMENT_NAME \
    --location=$REGION \
    --project=$PROJECT_ID \
    --format="value(config.dagGcsPrefix)" 2>/dev/null) || true

echo ""
echo "=== Composer Setup Complete ==="
echo ""
echo "Environment: $ENVIRONMENT_NAME"
echo "Region:      $REGION"
echo "Image:       $COMPOSER_IMAGE_VERSION"
echo "Service Acct: $SERVICE_ACCOUNT"
if [ -n "$DAGS_BUCKET" ]; then
    echo "DAGs Bucket: $DAGS_BUCKET"
    echo ""
    echo "To upload DAGs:"
    echo "  gsutil cp dags/*.py $DAGS_BUCKET/"
fi
echo ""
echo "To open the Airflow UI:"
echo "  gcloud composer environments describe $ENVIRONMENT_NAME \\"
echo "    --location=$REGION --format='value(config.airflowUri)'"
