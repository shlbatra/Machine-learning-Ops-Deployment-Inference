#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Export Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="$PROJECT_ROOT/deeplearning-sahil-e50332de6687.json"

# Run the BigQuery data loader script
python "$SCRIPT_DIR/bq_dataloader.py"