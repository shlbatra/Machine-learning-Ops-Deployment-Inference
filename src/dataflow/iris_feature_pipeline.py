"""
Dataflow streaming pipeline for ingesting Pub/Sub messages into the Feature Store.
Reads from Pub/Sub, renames fields to canonical names, and dual-writes to:
  - BQ iris_features table (offline store, for training/batch)
  - Bigtable online store (real-time serving, sub-second latency)

No model calls — this pipeline only persists features.
"""

import json
import argparse
import logging
import uuid
from datetime import datetime, timezone

import apache_beam as beam
from apache_beam.options.pipeline_options import GoogleCloudOptions, PipelineOptions
from apache_beam.io import ReadFromPubSub, WriteToBigQuery
from apache_beam.transforms.util import BatchElements
from pydantic import ValidationError

from dataflow.models.iris_schema import PubSubIrisMessage
from dataflow.utils.online_store_writer import WriteToOnlineStore
from ml_pipelines_kfp.log import get_logger

logger = get_logger(__name__)

PROJECT_ID = "deeplearning-sahil"
REGION = "us-central1"

PUBSUB_TO_CANONICAL = {
    "sepal_length": "sepal_length_cm",
    "sepal_width": "sepal_width_cm",
    "petal_length": "petal_length_cm",
    "petal_width": "petal_width_cm",
}

FEATURE_TABLE_SCHEMA = {
    "fields": [
        {"name": "sepal_length_cm", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "sepal_width_cm", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "petal_length_cm", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "petal_width_cm", "type": "FLOAT", "mode": "REQUIRED"},
        {"name": "species", "type": "STRING", "mode": "NULLABLE"},
        {"name": "source", "type": "STRING", "mode": "REQUIRED"},
        {"name": "entity_id", "type": "STRING", "mode": "REQUIRED"},
        {"name": "feature_timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
    ]
}


class ParsePubSubMessage(beam.DoFn):
    """Parse and validate JSON message from Pub/Sub using Pydantic schema."""

    def process(self, element):
        try:
            message_data = json.loads(element.decode("utf-8"))
            validated = PubSubIrisMessage(**message_data)
            yield validated.model_dump()

        except ValidationError as e:
            logger.warning(f"Invalid message: {e}")
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Error parsing message: {e}, message: {element}")


class MapToFeatureRow(beam.DoFn):
    """Rename validated Pub/Sub fields to canonical names and add Feature Store metadata."""

    def process(self, element):
        row = {
            canonical: element[pubsub_key]
            for pubsub_key, canonical in PUBSUB_TO_CANONICAL.items()
        }

        sample_id = element.get("sample_id") or uuid.uuid4().hex[:8]
        logger.info(f"Processing entity_id: {sample_id}_streaming")
        row["species"] = None
        row["source"] = "streaming"
        row["entity_id"] = f"{sample_id}_streaming"
        row["feature_timestamp"] = datetime.now(timezone.utc).isoformat()

        yield row


def run_pipeline(argv=None):
    """Run the Dataflow streaming feature pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_topic",
        required=True,
        help="Pub/Sub topic to read from (projects/PROJECT/topics/TOPIC)",
    )
    parser.add_argument(
        "--output_table",
        required=True,
        help="BigQuery feature table (PROJECT:DATASET.TABLE)",
    )
    parser.add_argument("--project_id", required=True, help="GCP project ID")
    parser.add_argument("--region", required=True, help="GCP Region")
    parser.add_argument(
        "--online_batch_size",
        type=int,
        default=100,
        help="Max rows per online store write batch (default: 100)",
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
        "--no_wait",
        action="store_true",
        help="Submit the job and exit without waiting for it to finish",
    )

    known_args, pipeline_args = parser.parse_known_args(argv)
    logger.info(f"Known args: {known_args}")
    logger.info(f"Pipeline args: {pipeline_args}")

    pipeline_options = PipelineOptions(pipeline_args)

    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = known_args.project_id
    google_cloud_options.region = known_args.region

    p = beam.Pipeline(options=pipeline_options)

    feature_rows = (
        p
        | "Read from Pub/Sub" >> ReadFromPubSub(topic=known_args.input_topic)
        | "Parse JSON" >> beam.ParDo(ParsePubSubMessage())
        | "Map to Feature Row" >> beam.ParDo(MapToFeatureRow())
    )

    feature_rows | "Write to BQ (Offline Store)" >> WriteToBigQuery(
        table=known_args.output_table,
        schema=FEATURE_TABLE_SCHEMA,
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
    )

    (
        feature_rows
        | "Batch for Online Store"
        >> BatchElements(
            min_batch_size=1,
            max_batch_size=known_args.online_batch_size,
        )
        | "Write to Online Store (Bigtable)"
        >> beam.ParDo(
            WriteToOnlineStore(
                project_id=known_args.project_id,
                region=known_args.region,
                online_store_id=known_args.online_store_id,
                feature_view_id=known_args.feature_view_id,
                feature_columns=list(PUBSUB_TO_CANONICAL.values()),
            )
        )
    )

    result = p.run()
    if not known_args.no_wait:
        result.wait_until_finish()


if __name__ == "__main__":
    run_pipeline()
