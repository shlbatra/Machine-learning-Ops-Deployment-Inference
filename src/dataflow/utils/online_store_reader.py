import logging

import apache_beam as beam
from google.cloud.aiplatform_v1 import FeatureOnlineStoreServiceClient
from google.cloud.aiplatform_v1.types import (
    FetchFeatureValuesRequest,
    FeatureViewDataKey,
)

logger = logging.getLogger(__name__)


class FetchFeaturesFromOnlineStore(beam.DoFn):
    """Fetch feature values from the Feature Store online store by entity_id.

    Uses the v1 GA FeatureOnlineStoreServiceClient.fetch_feature_values() API
    for single-key lookups with millisecond latency (Bigtable-backed).

    v1 (GA) is used for reads — fetch_feature_values is a stable API.
    v1beta1 is only needed for writes (feature_view_direct_write).
    """

    def __init__(self, project_id, region, online_store_id, feature_view_id, feature_columns):
        self.project_id = project_id
        self.region = region
        self.online_store_id = online_store_id
        self.feature_view_id = feature_view_id
        self.feature_columns = set(feature_columns)

    def setup(self):
        self._client = FeatureOnlineStoreServiceClient(
            client_options={"api_endpoint": f"{self.region}-aiplatform.googleapis.com"}
        )
        self._feature_view_name = (
            f"projects/{self.project_id}/locations/{self.region}"
            f"/featureOnlineStores/{self.online_store_id}"
            f"/featureViews/{self.feature_view_id}"
        )

    def process(self, element):
        entity_id = element["entity_id"]

        try:
            response = self._client.fetch_feature_values(
                request=FetchFeatureValuesRequest(
                    feature_view=self._feature_view_name,
                    data_key=FeatureViewDataKey(key=entity_id),
                )
            )

            features = {}
            for pair in response.key_values.features:
                if pair.name in self.feature_columns:
                    features[pair.name] = pair.value.double_value

            if len(features) != len(self.feature_columns):
                missing = self.feature_columns - set(features.keys())
                logger.warning(f"Missing features for entity_id={entity_id}: {missing}")
                return

            element.update(features)
            yield element

        except Exception as e:
            logger.error(f"Feature fetch failed for entity_id={entity_id}: {e}")
