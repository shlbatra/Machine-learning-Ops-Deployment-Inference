import logging

import apache_beam as beam
from google.cloud.aiplatform_v1 import FeatureOnlineStoreAdminServiceClient

logger = logging.getLogger(__name__)


class TriggerFeatureSync(beam.DoFn):
    """Trigger a FeatureView sync after each window completes."""

    def __init__(self, project_id, region, online_store_id, feature_view_id):
        self.project_id = project_id
        self.region = region
        self.online_store_id = online_store_id
        self.feature_view_id = feature_view_id

    def setup(self):
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
