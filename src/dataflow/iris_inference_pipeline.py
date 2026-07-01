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
from apache_beam.io.gcp.bigquery import BigQueryWriteFn, RetryStrategy
from dataflow.utils.online_store_reader import FetchFeaturesFromOnlineStore
from dataflow.utils.dead_letter import DEAD_LETTER_TAG, build_dead_letter, write_dead_letters
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
        {"name": "features", "type": "STRING", "mode": "NULLABLE"},
        {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "prediction", "type": "STRING", "mode": "REQUIRED"},
        {"name": "class_probabilities", "type": "FLOAT", "mode": "REPEATED"},
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

    def __init__(self):
        self.parse_success = beam.metrics.Metrics.counter("ParsePubSubMessage", "parse_success")
        self.parse_error = beam.metrics.Metrics.counter("ParsePubSubMessage", "parse_error")
        self.missing_id = beam.metrics.Metrics.counter("ParsePubSubMessage", "missing_entity_id")

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
                    self.missing_id.inc()
                    logger.warning(f"Message missing entity_id and sample_id: {message_data}")
                    yield beam.pvalue.TaggedOutput(DEAD_LETTER_TAG, build_dead_letter(
                        pipeline="inference", stage="parse", error_type="missing_field",
                        error_message="Message missing entity_id and sample_id",
                        original_message=element,
                    ))
                    return

            self.parse_success.inc()
            yield {
                "entity_id": entity_id,
                "timestamp": message_data.get("timestamp"),
            }

        except (json.JSONDecodeError, AttributeError) as e:
            self.parse_error.inc()
            logger.error(f"Error parsing message: {e}, message: {element}")
            yield beam.pvalue.TaggedOutput(DEAD_LETTER_TAG, build_dead_letter(
                pipeline="inference", stage="parse", error_type="json_decode",
                error_message=e, original_message=element,
            ))


class BatchCallFastAPIService(beam.DoFn):
    """Call FastAPI with a batch of instances using async HTTP."""

    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2

    def __init__(self, service_url, max_concurrent=4):
        self.service_url = service_url
        self.predict_url = f"{service_url}/predict"
        self.max_concurrent = max_concurrent
        self.prediction_success = beam.metrics.Metrics.counter("BatchCallFastAPIService", "prediction_success")
        self.prediction_error = beam.metrics.Metrics.counter("BatchCallFastAPIService", "prediction_error")
        self.prediction_retry = beam.metrics.Metrics.counter("BatchCallFastAPIService", "prediction_retry")
        self.prediction_latency = beam.metrics.Metrics.distribution("BatchCallFastAPIService", "prediction_latency_ms")
        self.batch_size = beam.metrics.Metrics.distribution("BatchCallFastAPIService", "batch_size")

    def setup(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._session = self._loop.run_until_complete(self._create_session())

    async def _create_session(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(
            total=30, connect=10, sock_connect=10, sock_read=20,
        )
        return aiohttp.ClientSession(connector=connector, timeout=timeout)

    def teardown(self):
        self._loop.run_until_complete(self._session.close())
        self._loop.close()

    def process(self, batch):
        results, dead_letters = self._loop.run_until_complete(self._call_async(batch))
        yield from results
        for dl in dead_letters:
            yield beam.pvalue.TaggedOutput(DEAD_LETTER_TAG, dl)

    async def _call_async(self, batch):
        start_time = time.time()
        self.batch_size.update(len(batch))

        instances = [
            {col: e[col] for col in FEATURE_COLUMNS}
            for e in batch
        ]

        last_error = None
        retry_count = 0
        for attempt in range(self.MAX_RETRIES):
            try:
                async with self._session.post(
                    self.predict_url,
                    json={"instances": instances},
                ) as response:
                    response.raise_for_status()
                    result_data = await response.json()

                predictions = result_data.get("predictions", [])
                processing_time = time.time() - start_time

                self.prediction_latency.update(int(processing_time * 1000))
                self.prediction_success.inc(len(predictions))

                results = []
                for element, pred in zip(batch, predictions):
                    predicted_class = str(pred.get("class_", "unknown"))
                    features = {col: element[col] for col in FEATURE_COLUMNS}
                    row = {
                        "entity_id": element["entity_id"],
                        "features": json.dumps(features),
                        "timestamp": element.get("timestamp", datetime.now(timezone.utc).isoformat()),
                        "prediction": predicted_class,
                        "class_probabilities": pred.get("class_probabilities", []),
                        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
                        "model_service": self.service_url,
                        "processing_time": processing_time / len(batch),
                    }
                    logger.info(f"Row processed - {row}")
                    results.append(row)
                return results, []

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                retry_count += 1
                self.prediction_retry.inc()
                wait = self.RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    f"Batch prediction attempt {attempt + 1}/{self.MAX_RETRIES} "
                    f"failed ({len(batch)} instances): {e}. Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            except Exception as e:
                last_error = e
                break

        self.prediction_error.inc(len(batch))
        logging.error(f"Batch prediction failed after retries ({len(batch)} instances): {last_error}")
        error_type = "timeout" if isinstance(last_error, asyncio.TimeoutError) else "connection"
        dead_letters = [
            build_dead_letter(
                pipeline="inference", stage="predict", error_type=error_type,
                error_message=last_error, entity_id=el["entity_id"],
                retry_count=retry_count,
            )
            for el in batch
        ]
        return [], dead_letters


class AddProcessingMetadata(beam.DoFn):
    """Add processing metadata to records."""

    def process(self, element):
        element["dataflow_processing_time"] = datetime.now(timezone.utc).isoformat()
        yield element


class RaiseOnBigQueryError(beam.DoFn):
    """Raise an exception when BigQuery insert fails."""

    def process(self, element):
        table, row, errors = element
        raise RuntimeError(
            f"BigQuery insert failed for table={table}: {errors}. Row: {row}"
        )


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
        "--dead_letter_table",
        default=None,
        help="BigQuery dead letter table (PROJECT:DATASET.TABLE). If unset, dead letters are logged only.",
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

    parse_results = (
        pipeline
        | "Read from Pub/Sub" >> ReadFromPubSub(topic=known_args.input_topic)
        | "Parse JSON" >> beam.ParDo(ParsePubSubMessage()).with_outputs(
            DEAD_LETTER_TAG, main="parsed",
        )
    )

    fetch_results = (
        parse_results.parsed
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
        ).with_outputs(DEAD_LETTER_TAG, main="fetched")
    )

    predict_results = (
        fetch_results.fetched
        | "Batch for Prediction" >> BatchElements(
            min_batch_size=1,
            max_batch_size=known_args.batch_size,
            max_batch_duration_secs=known_args.max_batch_duration_secs,
        )
        | "Call FastAPI Batch"
        >> beam.ParDo(BatchCallFastAPIService(known_args.service_url)).with_outputs(
            DEAD_LETTER_TAG, main="predictions",
        )
    )

    predictions = (
        predict_results.predictions
        | "Add Metadata" >> beam.ParDo(AddProcessingMetadata())
        | "Write to BigQuery"
        >> WriteToBigQuery(
            table=known_args.output_table,
            schema=PREDICTION_SCHEMA,
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
            insert_retry_strategy=RetryStrategy.RETRY_NEVER,
            additional_bq_parameters={
                "timePartitioning": {"type": "DAY", "field": "prediction_timestamp"}
            },
        )
    )

    _ = (
        predictions[BigQueryWriteFn.FAILED_ROWS_WITH_ERRORS]
        | "Raise on BQ Error" >> beam.ParDo(RaiseOnBigQueryError())
    )

    if known_args.dead_letter_table:
        all_dead_letters = (
            (
                parse_results[DEAD_LETTER_TAG],
                fetch_results[DEAD_LETTER_TAG],
                predict_results[DEAD_LETTER_TAG],
            )
            | "Flatten Dead Letters" >> beam.Flatten()
        )
        write_dead_letters(
            all_dead_letters,
            table=known_args.dead_letter_table,
        )

    result = pipeline.run()
    if not known_args.no_wait:
        result.wait_until_finish()


if __name__ == "__main__":
    run_pipeline()
