"""
Dataflow streaming pipeline for real-time Iris inference.
Reads from Pub/Sub, calls FastAPI ML service deployed via Kubeflow, writes predictions to BigQuery.
"""

import json
import logging
import argparse
from typing import Any, Dict, List
import requests
import time

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms import window
from apache_beam.io import ReadFromPubSub, WriteToBigQuery

# Constants
PROJECT_ID = "deeplearning-sahil"
REGION = "us-central1"
MODEL_NAME = "Iris-Classifier-XGBoost"
FASTAPI_SERVICE_NAME = "iris-classifier-xgboost-service"

# BigQuery schema for predictions
PREDICTION_SCHEMA = {
    "fields": [
        {"name": "sepal_length", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "sepal_width", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "petal_length", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "petal_width", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "sample_id", "type": "INTEGER", "mode": "REQUIRED"},
        {"name": "prediction", "type": "STRING", "mode": "REQUIRED"},
        {"name": "prediction_timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "model_service", "type": "STRING", "mode": "REQUIRED"},
        {"name": "processing_time", "type": "FLOAT", "mode": "NULLABLE"},
        {"name": "dataflow_processing_time", "type": "TIMESTAMP", "mode": "REQUIRED"},
    ]
}


class ParsePubSubMessage(beam.DoFn):
    """Parse JSON message from Pub/Sub."""

    def process(self, element):
        try:
            # Parse the Pub/Sub message
            message_data = json.loads(element.decode("utf-8"))

            # Validate required fields
            required_fields = [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ]
            if all(field in message_data for field in required_fields):
                yield message_data
            else:
                logging.warning(f"Missing required fields in message: {message_data}")

        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(f"Error parsing message: {e}, message: {element}")


class CallFastAPIService(beam.DoFn):
    """Call FastAPI ML service for inference."""

    def __init__(self, service_url: str):
        self.service_url = service_url
        self.predict_url = f"{service_url}/predict"

    def process(self, element):
        import time
        from datetime import datetime
        import requests

        start_time = time.time()

        try:
            # Prepare payload for FastAPI
            payload = {
                "instances": [
                    {
                        "SepalLengthCm": element["sepal_length"],
                        "SepalWidthCm": element["sepal_width"],
                        "PetalLengthCm": element["petal_length"],
                        "PetalWidthCm": element["petal_width"],
                    }
                ]
            }

            # Call FastAPI service
            response = requests.post(self.predict_url, json=payload, timeout=30)
            response.raise_for_status()

            # Parse response
            result_data = response.json()
            predictions = result_data.get("predictions", [])

            if predictions:
                prediction_result = predictions[0]
                predicted_class = str(prediction_result.get("prediction", "unknown"))
            else:
                predicted_class = "unknown"

            processing_time = time.time() - start_time

            # Create result record
            result = {
                "sepal_length": element["sepal_length"],
                "sepal_width": element["sepal_width"],
                "petal_length": element["petal_length"],
                "petal_width": element["petal_width"],
                "timestamp": element.get("timestamp", datetime.utcnow().isoformat()),
                "sample_id": element.get("sample_id", 0),
                "prediction": predicted_class,
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "model_service": self.service_url,
                "processing_time": processing_time,
            }

            logging.info(
                f"Prediction for sample {element.get('sample_id')}: {predicted_class}"
            )
            yield result

        except Exception as e:
            logging.error(f"Error calling FastAPI service: {e}, element: {element}")
            # Yield error record for monitoring
            yield {
                "sepal_length": element.get("sepal_length", 0.0),
                "sepal_width": element.get("sepal_width", 0.0),
                "petal_length": element.get("petal_length", 0.0),
                "petal_width": element.get("petal_width", 0.0),
                "timestamp": element.get("timestamp", datetime.utcnow().isoformat()),
                "sample_id": element.get("sample_id", 0),
                "prediction": "ERROR",
                "prediction_timestamp": datetime.utcnow().isoformat(),
                "model_service": f"ERROR: {str(e)}",
                "processing_time": time.time() - start_time,
            }


class AddProcessingMetadata(beam.DoFn):
    """Add processing metadata to records."""

    def process(self, element):
        from datetime import datetime

        # Add additional metadata
        element["dataflow_processing_time"] = datetime.utcnow().isoformat()

        yield element


def run_pipeline(argv=None):
    """Run the Dataflow streaming pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_topic",
        required=True,
        help="Pub/Sub topic to read from (projects/PROJECT/topics/TOPIC)",
    )
    parser.add_argument(
        "--output_table",
        required=True,
        help="BigQuery table to write to (PROJECT:DATASET.TABLE)",
    )
    parser.add_argument("--project_id", required=True, help="Project ID")
    parser.add_argument("--region", required=True, help="GCP Region")
    parser.add_argument("--service_url", required=True, help="FastAPI service URL")

    known_args, pipeline_args = parser.parse_known_args(argv)
    logging.info(f"Known args: {known_args}")
    logging.info(f"Pipeline args: {pipeline_args}")

    # Pipeline options - ensure project is set
    pipeline_options = PipelineOptions(pipeline_args)

    # Explicitly set the project for Dataflow
    from apache_beam.options.pipeline_options import GoogleCloudOptions

    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = known_args.project_id

    # Create pipeline
    with beam.Pipeline(options=pipeline_options) as pipeline:

        predictions = (
            pipeline
            | "Read from Pub/Sub" >> ReadFromPubSub(topic=known_args.input_topic)
            | "Parse JSON" >> beam.ParDo(ParsePubSubMessage())
            | "Call FastAPI Service"
            >> beam.ParDo(CallFastAPIService(known_args.service_url))
            | "Add Metadata" >> beam.ParDo(AddProcessingMetadata())
            | "Write to BigQuery"
            >> WriteToBigQuery(
                table=known_args.output_table,
                schema=PREDICTION_SCHEMA,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                additional_bq_parameters={
                    "timePartitioning": {"type": "DAY", "field": "prediction_timestamp"}
                },
            )
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run_pipeline()
