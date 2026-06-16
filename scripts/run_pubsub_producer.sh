#!/bin/bash

# Generate random Iris data and publish to Pub/Sub for streaming inference testing
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ID="deeplearning-sahil"
TOPIC="iris-inference-data"
BATCH_SIZE=${1:-10}
DELAY=${2:-5}
DURATION=${3:-}

echo "Starting Pub/Sub producer..."
echo "Project: $PROJECT_ID"
echo "Topic: $TOPIC"
echo "Batch size: $BATCH_SIZE"
echo "Delay: ${DELAY}s"
echo "Duration: ${DURATION:-infinite}"
echo ""

DURATION_ARG=""
if [ -n "$DURATION" ]; then
  DURATION_ARG="--duration $DURATION"
fi

python "$SCRIPT_DIR/pubsub_producer.py" \
  --project-id "$PROJECT_ID" \
  --topic "$TOPIC" \
  --batch-size "$BATCH_SIZE" \
  --delay "$DELAY" \
  $DURATION_ARG
