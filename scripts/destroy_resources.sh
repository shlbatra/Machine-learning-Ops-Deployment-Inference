#!/bin/bash

# Teardown script for the three "always-on" (cost-incurring) POC resources:
#   1. Cloud Composer 2 environment  (GKE + Airflow, runs 24/7 — most expensive)
#   2. Vertex AI Feature Store        (Bigtable-backed online store, billed per node-hour)
#   3. Artifact Registry repository   (Docker images — cheap, but deleting frees storage)
#
# Mirrors the create scripts:
#   setup_composer.sh, setup_feature_store.sh, setup_artifact_registry.sh
#
# All three can be recreated later from those scripts (see Readme / notes at bottom).
# BigQuery datasets, GCS buckets, Pub/Sub topics and the kfp-mlops@ service account
# are intentionally NOT touched — the setup scripts depend on them.
#
# Usage:
#   ./scripts/destroy_resources.sh                     # destroy all three (asks to confirm)
#   ./scripts/destroy_resources.sh --yes               # skip the confirmation prompt
#   ./scripts/destroy_resources.sh --skip-composer     # leave Composer alone
#   ./scripts/destroy_resources.sh --skip-feature-store
#   ./scripts/destroy_resources.sh --skip-artifact-registry
#   ./scripts/destroy_resources.sh --keep-bucket       # keep the leftover Composer DAGs bucket
#   ./scripts/destroy_resources.sh --dry-run           # print the delete commands, run nothing

set -euo pipefail

# --- Configuration (must match the setup scripts) ---
PROJECT_ID="deeplearning-sahil"

# Composer (setup_composer.sh)
COMPOSER_REGION="us-central1"
COMPOSER_ENV="ml-pipelines-composer"

# Feature Store (feature_store/setup.py + iris/feature_definitions.py)
FS_REGION="us-central1"
ONLINE_STORE_ID="ml_online_store"
FEATURE_VIEW_ID="iris_features"

# Artifact Registry (setup_artifact_registry.sh)
AR_LOCATION="us"
AR_REPOSITORY="sahil-experiment-docker-images"

# --- Flags ---
ASSUME_YES=false
SKIP_COMPOSER=false
SKIP_FEATURE_STORE=false
SKIP_ARTIFACT_REGISTRY=false
KEEP_BUCKET=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --yes|-y)                ASSUME_YES=true ;;
        --skip-composer)         SKIP_COMPOSER=true ;;
        --skip-feature-store)    SKIP_FEATURE_STORE=true ;;
        --skip-artifact-registry) SKIP_ARTIFACT_REGISTRY=true ;;
        --keep-bucket)           KEEP_BUCKET=true ;;
        --dry-run)               DRY_RUN=true ;;
        -h|--help)
            sed -n '3,31p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# run <cmd...> — execute, or just print when --dry-run is set.
run() {
    if [ "$DRY_RUN" = "true" ]; then
        echo "  [dry-run] $*"
    else
        "$@"
    fi
}

echo "=== DESTROY POC resources in project: $PROJECT_ID ==="
echo ""
echo "The following will be PERMANENTLY deleted:"
$SKIP_COMPOSER          || echo "  - Composer env      : $COMPOSER_ENV ($COMPOSER_REGION) + its GKE cluster"
$SKIP_COMPOSER          || { $KEEP_BUCKET || echo "  - Composer DAGs bucket (leftover after env delete)"; }
$SKIP_FEATURE_STORE     || echo "  - Feature view      : $FEATURE_VIEW_ID"
$SKIP_FEATURE_STORE     || echo "  - Online store      : $ONLINE_STORE_ID ($FS_REGION, Bigtable)"
$SKIP_ARTIFACT_REGISTRY || echo "  - Artifact repo     : $AR_LOCATION/$AR_REPOSITORY (ALL images)"
echo ""
echo "NOT touched: BigQuery, GCS buckets (sb-vertex), Pub/Sub, service accounts, IAM."
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "(dry-run: no resources will be deleted)"
elif [ "$ASSUME_YES" != "true" ]; then
    read -r -p "Type 'destroy' to proceed: " CONFIRM
    if [ "$CONFIRM" != "destroy" ]; then
        echo "Aborted."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 1. Feature Store  (delete view BEFORE the store — the store can't be
#    deleted while a view still references it)
# ---------------------------------------------------------------------------
if [ "$SKIP_FEATURE_STORE" != "true" ]; then
    echo ""
    echo "--- Feature Store ---"

    if gcloud ai feature-online-stores feature-views describe "$FEATURE_VIEW_ID" \
        --feature-online-store="$ONLINE_STORE_ID" \
        --region="$FS_REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
        echo "Deleting feature view '$FEATURE_VIEW_ID'..."
        run gcloud ai feature-online-stores feature-views delete "$FEATURE_VIEW_ID" \
            --feature-online-store="$ONLINE_STORE_ID" \
            --region="$FS_REGION" --project="$PROJECT_ID" --quiet
    else
        echo "Feature view '$FEATURE_VIEW_ID' not found — skipping."
    fi

    if gcloud ai feature-online-stores describe "$ONLINE_STORE_ID" \
        --region="$FS_REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
        echo "Deleting online store '$ONLINE_STORE_ID' (stops Bigtable node billing)..."
        run gcloud ai feature-online-stores delete "$ONLINE_STORE_ID" \
            --region="$FS_REGION" --project="$PROJECT_ID" --quiet
    else
        echo "Online store '$ONLINE_STORE_ID' not found — skipping."
    fi
    echo "Feature Store torn down."
fi

# ---------------------------------------------------------------------------
# 2. Artifact Registry  (deletes the repo and every image inside it)
# ---------------------------------------------------------------------------
if [ "$SKIP_ARTIFACT_REGISTRY" != "true" ]; then
    echo ""
    echo "--- Artifact Registry ---"

    if gcloud artifacts repositories describe "$AR_REPOSITORY" \
        --location="$AR_LOCATION" --project="$PROJECT_ID" >/dev/null 2>&1; then
        echo "Deleting repository '$AR_REPOSITORY' and all images..."
        run gcloud artifacts repositories delete "$AR_REPOSITORY" \
            --location="$AR_LOCATION" --project="$PROJECT_ID" --quiet
    else
        echo "Repository '$AR_REPOSITORY' not found — skipping."
    fi
    echo "Artifact Registry torn down."
fi

# ---------------------------------------------------------------------------
# 3. Composer  (slowest — ~20 min; deletes the underlying GKE cluster too.
#    The DAGs GCS bucket is left behind by Composer, so remove it separately.)
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPOSER" != "true" ]; then
    echo ""
    echo "--- Cloud Composer ---"

    if gcloud composer environments describe "$COMPOSER_ENV" \
        --location="$COMPOSER_REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then

        # Capture the DAGs bucket before deleting the env (afterwards it's unreachable).
        DAGS_PREFIX=$(gcloud composer environments describe "$COMPOSER_ENV" \
            --location="$COMPOSER_REGION" --project="$PROJECT_ID" \
            --format="value(config.dagGcsPrefix)" 2>/dev/null) || true

        echo "Deleting Composer environment '$COMPOSER_ENV' (this can take ~20 min)..."
        run gcloud composer environments delete "$COMPOSER_ENV" \
            --location="$COMPOSER_REGION" --project="$PROJECT_ID" --quiet

        # dagGcsPrefix looks like gs://<bucket>/dags — strip to the bucket root.
        if [ "$KEEP_BUCKET" != "true" ] && [ -n "${DAGS_PREFIX:-}" ]; then
            BUCKET=$(echo "$DAGS_PREFIX" | sed -E 's|^(gs://[^/]+).*|\1|')
            if [ -n "$BUCKET" ] && { [ "$DRY_RUN" = "true" ] || gsutil ls "$BUCKET" >/dev/null 2>&1; }; then
                echo "Deleting leftover DAGs bucket: $BUCKET"
                run gsutil -m rm -r "$BUCKET" || echo "  Could not remove $BUCKET — delete it manually if needed."
            fi
        elif [ "$KEEP_BUCKET" = "true" ]; then
            echo "Keeping DAGs bucket (${DAGS_PREFIX:-unknown}) as requested."
        fi
    else
        echo "Composer environment '$COMPOSER_ENV' not found — skipping."
    fi
    echo "Composer torn down."
fi

echo ""
echo "=== Teardown complete ==="
echo ""
echo "To recreate later:"
echo "  ./scripts/setup_artifact_registry.sh     # repo + IAM"
echo "  # then push to main (or re-run the CI/CD workflow) to rebuild & push images"
echo "  ./scripts/setup_feature_store.sh          # online store + view (needs BQ table present)"
echo "  ./scripts/setup_composer.sh               # ~25 min; then ./scripts/sync_dags.sh"
