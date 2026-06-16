#!/bin/bash

# Uses application-default credentials (run `gcloud auth application-default login` once locally)
python -m feature_store.setup "$@"
