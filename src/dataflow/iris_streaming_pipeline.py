"""
Dataflow streaming pipeline for real-time Iris inference via Feature Store.
Reads entity IDs from Pub/Sub, fetches features from the online store,
calls FastAPI ML service for predictions, writes results to BigQuery.

Decoupled from the feature ingestion pipeline (iris_feature_pipeline.py) —
this pipeline only reads from the online store and runs inference. Features
can be written by the streaming feature pipeline or batch ingest.py + sync.py.
"""

import json
import argparse
import asyncio
import logging
from datetime import datetime, timezone
import time

import aiohttp
import apache_beam as beam
from apache_beam.options.pipeline_options import GoogleCloudOptions, PipelineOptions
from apache_beam.transforms.util import BatchElements
from apache_beam.io import ReadFromPubSub, WriteToBigQuery
from dataflow.utils.online_store_reader import FetchFeaturesFromOnlineStore
from ml_pipelines_kfp.log import get_logger

logger = get_logger(__name__)

PROJECT_ID = "deeplearning-sahil"
REGION = "us-central1"
MODEL_NAME = "Iris-Classifier-XGBoost"
FASTAPI_SERVICE_NAME = "iris-classifier-xgboost-service"

FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]

PREDICTION_SCHEMA = {
    "fields": [
        {"name": "entity_id", "type": "STRING", "mode": "REQUIRED"},
        {"name": "sepal_length_cm", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "sepal_width_cm", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "petal_length_cm", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "petal_width_cm", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "prediction", "type": "STRING", "mode": "REQUIRED"},
        {"name": "prediction_timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "model_service", "type": "STRING", "mode": "REQUIRED"},
        {"name": "processing_time", "type": "FLOAT", "mode": "NULLABLE"},
        {"name": "dataflow_processing_time", "type": "TIMESTAMP", "mode": "REQUIRED"},
    ]
}


class ParsePubSubMessage(beam.DoFn):
    """Parse JSON message from Pub/Sub — only entity_id is required.

    Accepts messages with either 'entity_id' directly or 'sample_id'
    (converted to '{sample_id}_streaming' for backward compat with
    the feature ingestion pipeline's entity_id format).
    """

    def process(self, element):
        try:
            message_data = json.loads(element.decode("utf-8"))

            entity_id = message_data.get("entity_id")
            if not entity_id:
                sample_id = message_data.get("sample_id")
                if sample_id is not None:
                    entity_id = f"{sample_id}_streaming"
                    logger.info(f"Entity Id scored = {entity_id}")
                else:
                    logger.warning(f"Message missing entity_id and sample_id: {message_data}")
                    return

            yield {
                "entity_id": entity_id,
                "timestamp": message_data.get("timestamp"),
            }

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
        self._session = self._loop.run_until_complete(self._create_session())

    async def _create_session(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        return aiohttp.ClientSession(connector=connector)

    def teardown(self):
        self._loop.run_until_complete(self._session.close())
        self._loop.close()

    def process(self, batch):
        results = self._loop.run_until_complete(self._call_async(batch))
        yield from results

    async def _call_async(self, batch):
        start_time = time.time()

        instances = [
            {col: e[col] for col in FEATURE_COLUMNS}
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
                row = {col: element[col] for col in FEATURE_COLUMNS}
                row.update({
                    "entity_id": element["entity_id"],
                    "timestamp": element.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "prediction": predicted_class,
                    "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_service": self.service_url,
                    "processing_time": processing_time / len(batch),
                })
                logger.info(f"Row processed - {row}")
                results.append(row)
            return results

        except Exception as e:
            logging.error(f"Batch prediction failed ({len(batch)} instances): {e}")
            processing_time = time.time() - start_time
            return [
                {
                    "entity_id": el["entity_id"],
                    "sepal_length_cm": el.get("sepal_length_cm", 0.0),
                    "sepal_width_cm": el.get("sepal_width_cm", 0.0),
                    "petal_length_cm": el.get("petal_length_cm", 0.0),
                    "petal_width_cm": el.get("petal_width_cm", 0.0),
                    "timestamp": el.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "prediction": "ERROR",
                    "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_service": f"ERROR: {str(e)}",
                    "processing_time": processing_time,
                }
                for el in batch
            ]


class AddProcessingMetadata(beam.DoFn):
    """Add processing metadata to records."""

    def process(self, element):
        element["dataflow_processing_time"] = datetime.now(timezone.utc).isoformat()
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
        "--online_store_id",
        default="ml_online_store",
        help="Feature Online Store ID (default: ml_online_store)",
    )
    parser.add_argument(
        "--feature_view_id",
        default="iris_features",
        help="Feature View ID (default: iris_features)",
    )
    parser.add_argument(
        "--no_wait", action="store_true",
        help="Submit the job and exit without waiting for it to finish",
    )

    known_args, pipeline_args = parser.parse_known_args(argv)
    logger.info(f"Known args: {known_args}")
    logger.info(f"Pipeline args: {pipeline_args}")

    pipeline_options = PipelineOptions(pipeline_args)

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
        | "Fetch Features" >> beam.ParDo(
            FetchFeaturesFromOnlineStore(
                project_id=known_args.project_id,
                region=known_args.region,
                online_store_id=known_args.online_store_id,
                feature_view_id=known_args.feature_view_id,
                feature_columns=FEATURE_COLUMNS,
            )
        )
        | "Call FastAPI Batch"
        >> beam.ParDo(BatchCallFastAPIService(known_args.service_url))
        | "Add Metadata" >> beam.ParDo(AddProcessingMetadata())
        | "Write to BigQuery"
        >> WriteToBigQuery(
            table=known_args.output_table,
            schema=PREDICTION_SCHEMA,
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
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
