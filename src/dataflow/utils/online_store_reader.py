import asyncio
import logging

import apache_beam as beam
from google.cloud.aiplatform_v1 import FeatureOnlineStoreServiceAsyncClient
from google.cloud.aiplatform_v1.types import (
    FetchFeatureValuesRequest,
    FeatureViewDataKey,
)

logger = logging.getLogger(__name__)


class FetchFeaturesFromOnlineStore(beam.DoFn):
    """Fetch feature values from the Feature Store online store by entity_id.

    Processes batched elements (from BatchElements) and fetches all entity_ids
    concurrently using the async client with a semaphore-controlled concurrency limit.

    v1 (GA) is used for reads — fetch_feature_values is a stable API.
    v1beta1 is only needed for writes (feature_view_direct_write).
    """

    def __init__(self, project_id, region, online_store_id, feature_view_id,
                 feature_columns, max_concurrent=8, max_retries=2,
                 initial_backoff_secs=1.0):
        self.project_id = project_id
        self.region = region
        self.online_store_id = online_store_id
        self.feature_view_id = feature_view_id
        self.feature_columns = set(feature_columns)
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.initial_backoff_secs = initial_backoff_secs

    def setup(self):
        self._loop = asyncio.new_event_loop()
        self._client = FeatureOnlineStoreServiceAsyncClient(
            client_options={"api_endpoint": f"{self.region}-aiplatform.googleapis.com"}
        )
        self._feature_view_name = (
            f"projects/{self.project_id}/locations/{self.region}"
            f"/featureOnlineStores/{self.online_store_id}"
            f"/featureViews/{self.feature_view_id}"
        )

    def teardown(self):
        self._loop.close()

    def process(self, batch):
        results = self._loop.run_until_complete(self._fetch_batch(batch))
        yield from results

    async def _fetch_batch(self, batch):
        # Limits concurrent API calls to max_concurrent — without this,
        # a large batch would fire all requests at once and overwhelm the online store.
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._fetch_one(elem, semaphore) for elem in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if r is not None and not isinstance(r, Exception)]

    async def _fetch_one(self, element, semaphore):
        entity_id = element["entity_id"]
        async with semaphore:
            # Retry with exponential backoff — covers the race condition where the
            # feature pipeline hasn't written this entity's features to the online
            # store yet (e.g. 0.5s, then 1s before giving up).
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._client.fetch_feature_values(
                        request=FetchFeatureValuesRequest(
                            feature_view=self._feature_view_name,
                            data_key=FeatureViewDataKey(key=entity_id),
                        )
                    )

                    features = {}
                    for pair in response.key_values.features:
                        if pair.name in self.feature_columns:
                            features[pair.name] = pair.value.double_value

                    if len(features) == len(self.feature_columns):
                        element.update(features)
                        return element

                    if attempt < self.max_retries:
                        backoff = self.initial_backoff_secs * (2 ** attempt)
                        logger.info(f"Missing features for entity_id={entity_id}, retrying in {backoff}s")
                        await asyncio.sleep(backoff)
                    else:
                        missing = self.feature_columns - set(features.keys())
                        logger.warning(f"Missing features for entity_id={entity_id} after {self.max_retries} retries: {missing}")
                        return None

                except Exception as e:
                    if attempt < self.max_retries:
                        backoff = self.initial_backoff_secs * (2 ** attempt)
                        logger.warning(f"Feature fetch failed for entity_id={entity_id}, retrying in {backoff}s: {e}")
                        await asyncio.sleep(backoff)
                    else:
                        logger.error(f"Feature fetch failed for entity_id={entity_id} after {self.max_retries} retries: {e}")
                        return None
