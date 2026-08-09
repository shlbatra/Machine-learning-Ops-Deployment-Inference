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

# --- Vertex AI Feature Store REST helpers ---
# `gcloud ai feature-online-stores` is beta-only and absent from a stock gcloud
# install, so it can't be used here: it exits 2 ("Invalid choice") and a
# `if gcloud ... 2>/dev/null` guard silently misreads that as "doesn't exist",
# leaving a Bigtable node billing forever. Creation already goes straight to the
# REST API via the Python SDK (feature_store/setup.py), so deletion does too.
FS_API="https://${FS_REGION}-aiplatform.googleapis.com/v1"
FS_PARENT="projects/${PROJECT_ID}/locations/${FS_REGION}"

fs_curl() {  # fs_curl <method> <url> -> body on stdout, HTTP status as exit-code proxy
    curl -sS -X "$1" \
        -H "Authorization: Bearer $(gcloud auth print-access-token --project="$PROJECT_ID")" \
        -H "Content-Type: application/json" \
        -w '\n%{http_code}' "$2"
}

# fs_get <url> — succeeds only on HTTP 200; prints the body.
fs_get() {
    local out status
    out=$(fs_curl GET "$1") || return 1
    status=$(printf '%s' "$out" | tail -n1)
    [ "$status" = "200" ] || return 1
    printf '%s' "$out" | sed '$d'
}

# fs_delete <url> <label> — DELETE, then poll the returned LRO to completion.
fs_delete() {
    local url="$1" label="$2" out status body op done_at
    out=$(fs_curl DELETE "$url") || { echo "  Delete request for $label failed."; return 1; }
    status=$(printf '%s' "$out" | tail -n1)
    body=$(printf '%s' "$out" | sed '$d')
    if [ "$status" != "200" ]; then
        echo "  Delete of $label returned HTTP $status:"
        echo "$body" | head -20
        return 1
    fi

    op=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("name",""))')
    [ -n "$op" ] || { echo "  $label: delete accepted (no operation returned)."; return 0; }

    # Poll the long-running operation — store deletes take a minute or two.
    for _ in $(seq 1 60); do
        done_at=$(fs_get "${FS_API}/${op}" 2>/dev/null \
            | python3 -c 'import json,sys; d=json.load(sys.stdin); print("done" if d.get("done") else "")' 2>/dev/null) || true
        if [ "$done_at" = "done" ]; then
            echo "  $label deleted."
            return 0
        fi
        sleep 5
    done
    echo "  Timed out waiting for $label deletion — check the console."
    return 1
}

echo "=== DESTROY POC resources in project: $PROJECT_ID ==="
echo ""
echo "The following will be PERMANENTLY deleted:"
$SKIP_COMPOSER          || echo "  - Composer env      : $COMPOSER_ENV ($COMPOSER_REGION) + its GKE cluster"
$SKIP_COMPOSER          || { $KEEP_BUCKET || echo "  - Composer DAGs bucket (leftover after env delete)"; }
$SKIP_FEATURE_STORE     || echo "  - Feature views     : ALL views in $ONLINE_STORE_ID (e.g. $FEATURE_VIEW_ID)"
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

    STORE_URL="${FS_API}/${FS_PARENT}/featureOnlineStores/${ONLINE_STORE_ID}"

    if ! fs_get "$STORE_URL" >/dev/null 2>&1; then
        echo "Online store '$ONLINE_STORE_ID' not found — skipping."
    else
        # Delete every view in the store, not just $FEATURE_VIEW_ID — any
        # leftover view blocks the store delete with FAILED_PRECONDITION.
        VIEWS=$(fs_get "${STORE_URL}/featureViews" 2>/dev/null \
            | python3 -c 'import json,sys; print("\n".join(v["name"] for v in json.load(sys.stdin).get("featureViews",[])))' \
            || true)

        if [ -z "${VIEWS:-}" ]; then
            echo "No feature views in '$ONLINE_STORE_ID'."
        else
            while IFS= read -r view; do
                [ -n "$view" ] || continue
                echo "Deleting feature view '${view##*/}'..."
                run fs_delete "${FS_API}/${view}" "feature view '${view##*/}'"
            done <<< "$VIEWS"
        fi

        echo "Deleting online store '$ONLINE_STORE_ID' (stops Bigtable node billing)..."
        run fs_delete "$STORE_URL" "online store '$ONLINE_STORE_ID'"
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
