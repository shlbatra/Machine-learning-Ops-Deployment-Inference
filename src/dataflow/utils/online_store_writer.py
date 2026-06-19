import logging

import apache_beam as beam
from google.cloud.aiplatform_v1beta1 import FeatureOnlineStoreServiceClient
from google.cloud.aiplatform_v1beta1.types import (
    FeatureViewDirectWriteRequest,
    FeatureViewDataKey,
    FeatureValue,
)

logger = logging.getLogger(__name__)

DataKeyAndFeatureValues = FeatureViewDirectWriteRequest.DataKeyAndFeatureValues
Feature = DataKeyAndFeatureValues.Feature
FeatureValueAndTimestamp = Feature.FeatureValueAndTimestamp


class WriteToOnlineStore(beam.DoFn):
    """Write a batch of feature rows directly to the Feature Store online store (Bigtable).

    Uses the v1beta1 feature_view_direct_write streaming RPC.
    Expects each batch element to be a dict with 'entity_id' and feature columns.
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

    def _build_entry(self, row):
        features = []
        for col in self.feature_columns:
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, (int, float)):
                fv = FeatureValue(double_value=float(val))
            else:
                fv = FeatureValue(string_value=str(val))
            features.append(
                Feature(
                    name=col,
                    value_and_timestamp=FeatureValueAndTimestamp(value=fv),
                )
            )
        return DataKeyAndFeatureValues(
            data_key=FeatureViewDataKey(key=row["entity_id"]),
            features=features,
        )

    def process(self, batch):
        entries = [self._build_entry(row) for row in batch]
        request = FeatureViewDirectWriteRequest(
            feature_view=self._feature_view_name,
            data_key_and_feature_values=entries,
        )

        try:
            responses = self._client.feature_view_direct_write(requests=iter([request]))
            # Drain the response stream to ensure the server has processed the write
            for resp in responses:
                logger.info(f"Direct write response: {resp}")
            logger.info(f"Wrote {len(entries)} rows to online store")
        except Exception as e:
            logger.error(f"Online store write failed ({len(entries)} rows): {e}")

        yield from batch
