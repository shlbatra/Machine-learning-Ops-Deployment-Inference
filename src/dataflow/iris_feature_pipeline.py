"""
Dataflow streaming pipeline for ingesting Pub/Sub messages into the Feature Store.
Reads from Pub/Sub, renames fields to canonical names, and writes to the
iris_features BQ table with WRITE_APPEND. Periodically triggers a FeatureView
sync so the online store stays fresh.

No model calls — this pipeline only persists features.
"""

import json
import argparse
import logging
import uuid

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromPubSub, WriteToBigQuery
from apache_beam.transforms.window import FixedWindows

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
        from pydantic import ValidationError
        from dataflow.models.iris_schema import PubSubIrisMessage

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
        from datetime import datetime, timezone

        row = {
            canonical: element[pubsub_key]
            for pubsub_key, canonical in PUBSUB_TO_CANONICAL.items()
        }

        sample_id = element.get("sample_id") or uuid.uuid4().hex[:8]
        row["species"] = None
        row["source"] = "streaming"
        row["entity_id"] = f"{sample_id}_streaming"
        row["feature_timestamp"] = datetime.now(timezone.utc).isoformat()

        yield row


class TriggerFeatureSync(beam.DoFn):
    """Trigger a FeatureView sync after each window completes."""

    def __init__(self, project_id, region, online_store_id, feature_view_id):
        self.project_id = project_id
        self.region = region
        self.online_store_id = online_store_id
        self.feature_view_id = feature_view_id

    def setup(self):
        from google.cloud.aiplatform_v1 import FeatureOnlineStoreAdminServiceClient

        self._client = FeatureOnlineStoreAdminServiceClient(
            client_options={"api_endpoint": f"{self.region}-aiplatform.googleapis.com"}
        )
        self._feature_view_name = (
            f"projects/{self.project_id}/locations/{self.region}"
            f"/featureOnlineStores/{self.online_store_id}"
            f"/featureViews/{self.feature_view_id}"
        )

    def process(self, count):
        try:
            response = self._client.sync_feature_view(
                feature_view=self._feature_view_name
            )
            logger.info(
                f"FeatureView sync triggered ({count} rows in window): "
                f"{response.feature_view_sync}"
            )
        except Exception as e:
            logger.error(f"FeatureView sync failed: {e}")
        yield count


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
        "--sync_interval_secs",
        type=int,
        default=300,
        help="How often to trigger FeatureView sync (seconds, default: 300)",
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

    from apache_beam.options.pipeline_options import GoogleCloudOptions

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

    feature_rows | "Write to Feature Table" >> WriteToBigQuery(
        table=known_args.output_table,
        schema=FEATURE_TABLE_SCHEMA,
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER,
    )

    (
        feature_rows
        | "Window for Sync" >> beam.WindowInto(
            FixedWindows(known_args.sync_interval_secs)
        )
        | "Count per Window" >> beam.combiners.Count.Globally().without_defaults()
        | "Trigger FeatureView Sync"
        >> beam.ParDo(
            TriggerFeatureSync(
                project_id=known_args.project_id,
                region=known_args.region,
                online_store_id=known_args.online_store_id,
                feature_view_id=known_args.feature_view_id,
            )
        )
    )

    result = p.run()
    if not known_args.no_wait:
        result.wait_until_finish()


if __name__ == "__main__":
    run_pipeline()
