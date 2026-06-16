import logging

import apache_beam as beam
from google.cloud.aiplatform_v1 import FeatureOnlineStoreServiceClient
from google.cloud.aiplatform_v1.types import (
    WriteFeatureValuesRequest,
    WriteFeatureValuesPayload,
    FeatureValue,
)

logger = logging.getLogger(__name__)


class WriteToOnlineStore(beam.DoFn):
    """Write a batch of feature rows directly to the Feature Store online store (Bigtable).

    Expects each batch element to be a dict with 'entity_id' and feature columns.
    Float features use double_value, string features use string_value.
    """

    def __init__(self, project_id, region, online_store_id, feature_view_id, feature_columns):
        self.project_id = project_id
        self.region = region
        self.online_store_id = online_store_id
        self.feature_view_id = feature_view_id
        self.feature_columns = feature_columns

    def setup(self):
        self._client = FeatureOnlineStoreServiceClient(
            client_options={"api_endpoint": f"{self.region}-aiplatform.googleapis.com"}
        )
        self._feature_view_name = (
            f"projects/{self.project_id}/locations/{self.region}"
            f"/featureOnlineStores/{self.online_store_id}"
            f"/featureViews/{self.feature_view_id}"
        )

    def process(self, batch):
        payloads = []
        for row in batch:
            feature_values = {}
            for col in self.feature_columns:
                val = row.get(col)
                if val is None:
                    continue
                if isinstance(val, (int, float)):
                    feature_values[col] = FeatureValue(double_value=float(val))
                else:
                    feature_values[col] = FeatureValue(string_value=str(val))

            payloads.append(
                WriteFeatureValuesPayload(
                    entity_id=row["entity_id"],
                    feature_values=feature_values,
                )
            )

        try:
            self._client.write_feature_values(
                request=WriteFeatureValuesRequest(
                    feature_view=self._feature_view_name,
                    payloads=payloads,
                )
            )
            logger.info(f"Wrote {len(payloads)} rows to online store")
        except Exception as e:
            logger.error(f"Online store write failed ({len(payloads)} rows): {e}")

        yield from batch
