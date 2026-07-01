#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-deeplearning-sahil}"
DATASET="ml_dataset"
TABLE="dead_letters"

echo "Creating dead letter table: ${PROJECT_ID}:${DATASET}.${TABLE}"

bq mk --table \
  --project_id="${PROJECT_ID}" \
  --time_partitioning_field=timestamp \
  --time_partitioning_type=DAY \
  --description="Dead letter table for pipeline failures — parse errors, fetch failures, prediction errors" \
  "${DATASET}.${TABLE}" \
  entity_id:STRING,pipeline:STRING,stage:STRING,error_type:STRING,error_message:STRING,original_message:STRING,timestamp:TIMESTAMP,retry_count:INTEGER

echo "Done. Table created: ${PROJECT_ID}:${DATASET}.${TABLE}"
