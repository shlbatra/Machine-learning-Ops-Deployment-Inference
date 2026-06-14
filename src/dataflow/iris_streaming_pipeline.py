"""
Dataflow streaming pipeline for real-time Iris inference.
Reads from Pub/Sub, calls FastAPI ML service deployed via Kubeflow, writes predictions to BigQuery.
"""

import json
import argparse
import asyncio
import logging
from typing import Any, Dict, List
import time

import aiohttp
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.util import BatchElements
from apache_beam.io import ReadFromPubSub, WriteToBigQuery

from ml_pipelines_kfp.log import get_logger

logger = get_logger(__name__)

PROJECT_ID = "deeplearning-sahil"
REGION = "us-central1"
MODEL_NAME = "Iris-Classifier-XGBoost"
FASTAPI_SERVICE_NAME = "iris-classifier-xgboost-service"

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
            message_data = json.loads(element.decode("utf-8"))

            required_fields = [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ]
            if all(field in message_data for field in required_fields):
                yield message_data
            else:
                logger.warning(f"Missing required fields in message: {message_data}")

        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Error parsing message: {e}, message: {element}")


class BatchCallFastAPIService(beam.DoFn):
    """Call FastAPI with a batch of instances using async HTTP."""

    def __init__(self, service_url, max_concurrent=4):
        self.service_url = service_url
        self.predict_url = f"{service_url}/predict"
        self.max_concurrent = max_concurrent

    def setup(self):
        self._loop = asyncio.new_event_loop()
        self._connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        self._session = aiohttp.ClientSession(connector=self._connector)

    def teardown(self):
        self._loop.run_until_complete(self._session.close())
        self._loop.close()

    def process(self, batch):
        results = self._loop.run_until_complete(self._call_async(batch))
        yield from results

    async def _call_async(self, batch):
        from datetime import datetime

        start_time = time.time()

        instances = [
            {
                "SepalLengthCm": e["sepal_length"],
                "SepalWidthCm": e["sepal_width"],
                "PetalLengthCm": e["petal_length"],
                "PetalWidthCm": e["petal_width"],
            }
            for e in batch
        ]

        try:
            async with self._session.post(
                self.predict_url,
                json={"instances": instances},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response.raise_for_status()
                result_data = await response.json()

            predictions = result_data.get("predictions", [])
            processing_time = time.time() - start_time

            results = []
            for element, pred in zip(batch, predictions):
                predicted_class = str(pred.get("prediction", "unknown"))
                results.append({
                    "sepal_length": element["sepal_length"],
                    "sepal_width": element["sepal_width"],
                    "petal_length": element["petal_length"],
                    "petal_width": element["petal_width"],
                    "timestamp": element.get("timestamp", datetime.utcnow().isoformat()),
                    "sample_id": element.get("sample_id", 0),
                    "prediction": predicted_class,
                    "prediction_timestamp": datetime.utcnow().isoformat(),
                    "model_service": self.service_url,
                    "processing_time": processing_time / len(batch),
                })
            return results

        except Exception as e:
            logging.error(f"Batch prediction failed ({len(batch)} instances): {e}")
            processing_time = time.time() - start_time
            return [
                {
                    "sepal_length": el.get("sepal_length", 0.0),
                    "sepal_width": el.get("sepal_width", 0.0),
                    "petal_length": el.get("petal_length", 0.0),
                    "petal_width": el.get("petal_width", 0.0),
                    "timestamp": el.get("timestamp", datetime.utcnow().isoformat()),
                    "sample_id": el.get("sample_id", 0),
                    "prediction": "ERROR",
                    "prediction_timestamp": datetime.utcnow().isoformat(),
                    "model_service": f"ERROR: {str(e)}",
                    "processing_time": processing_time,
                }
                for el in batch
            ]


class AddProcessingMetadata(beam.DoFn):
    """Add processing metadata to records."""

    def process(self, element):
        from datetime import datetime

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
    parser.add_argument(
        "--batch_size", type=int, default=50,
        help="Max instances per /predict call",
    )
    parser.add_argument(
        "--max_batch_duration_secs", type=float, default=1.0,
        help="Max seconds to buffer a partial batch before flushing",
    )
    parser.add_argument(
        "--no_wait", action="store_true",
        help="Submit the job and exit without waiting for it to finish",
    )

    known_args, pipeline_args = parser.parse_known_args(argv)
    logger.info(f"Known args: {known_args}")
    logger.info(f"Pipeline args: {pipeline_args}")

    pipeline_options = PipelineOptions(pipeline_args)

    from apache_beam.options.pipeline_options import GoogleCloudOptions

    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = known_args.project_id
    google_cloud_options.region = known_args.region

    pipeline = beam.Pipeline(options=pipeline_options)

    predictions = (
        pipeline
        | "Read from Pub/Sub" >> ReadFromPubSub(topic=known_args.input_topic)
        | "Parse JSON" >> beam.ParDo(ParsePubSubMessage())
        | "Batch Elements" >> BatchElements(
            min_batch_size=1,
            max_batch_size=known_args.batch_size,
            max_batch_duration_secs=known_args.max_batch_duration_secs,
        )
        | "Call FastAPI Batch"
        >> beam.ParDo(BatchCallFastAPIService(known_args.service_url))
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

    result = pipeline.run()
    if not known_args.no_wait:
        result.wait_until_finish()


if __name__ == "__main__":
    run_pipeline()
