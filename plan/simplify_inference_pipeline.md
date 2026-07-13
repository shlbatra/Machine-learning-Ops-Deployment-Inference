# Plan: Simplify iris_inference_pipeline — Remove Async from Feature Store Reads

## Problem

`FetchFeaturesFromOnlineStore` in `utils/online_store_reader.py` uses `asyncio` + `FeatureOnlineStoreServiceAsyncClient` with semaphore-controlled concurrency. This adds complexity (dedicated event loops per Beam worker, `asyncio.set_event_loop()` workaround for Beam's threading model, semaphore management) and is causing issues. The async pattern here is overkill because:

1. **Beam `BatchElements` already controls batch sizes** (max 50 items, 1s flush). Batches are small.
2. **Beam parallelism is at the worker level.** Multiple workers process batches concurrently — within-batch concurrency on the Feature Store adds marginal throughput but significant complexity.
3. The async `FeatureOnlineStoreServiceAsyncClient` requires careful event loop lifecycle management that conflicts with Beam's worker threading model.

**Keep async for:** `BatchCallFastAPIService` — the `aiohttp` HTTP call to FastAPI works correctly and the async pattern is simpler there (single POST per batch, no fan-out).

## Scope

| File | Change |
|---|---|
| `src/dataflow/utils/online_store_reader.py` | Replace async client with sync `FeatureOnlineStoreServiceClient`, remove `asyncio` entirely |

**No changes to:**
- `iris_inference_pipeline.py` — `BatchCallFastAPIService` stays async as-is
- `iris_feature_pipeline.py`, `online_store_writer.py`, FastAPI server
- Pipeline DAG structure, `BatchElements` config, BigQuery writes, CLI args

---

## Change: `FetchFeaturesFromOnlineStore` (utils/online_store_reader.py)

### Before (current — async)

```python
import asyncio
from google.cloud.aiplatform_v1 import FeatureOnlineStoreServiceAsyncClient

class FetchFeaturesFromOnlineStore(beam.DoFn):
    def __init__(self, ..., max_concurrent=8, ...):
        self.max_concurrent = max_concurrent
        ...

    def setup(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._client = FeatureOnlineStoreServiceAsyncClient(...)

    def teardown(self):
        self._loop.close()

    def process(self, batch):
        results = self._loop.run_until_complete(self._fetch_batch(batch))
        yield from results

    async def _fetch_batch(self, batch):
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._fetch_one(elem, semaphore) for elem in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if r is not None and not isinstance(r, Exception)]

    async def _fetch_one(self, element, semaphore):
        async with semaphore:
            for attempt in range(self.max_retries + 1):
                response = await self._client.fetch_feature_values(...)
                ...
                await asyncio.sleep(backoff)
```

### After (simplified — sync)

```python
import time
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

    Processes batched elements (from BatchElements) and fetches each entity_id
    sequentially using the sync client.

    v1 (GA) is used for reads — fetch_feature_values is a stable API.
    v1beta1 is only needed for writes (feature_view_direct_write).
    """

    def __init__(self, project_id, region, online_store_id, feature_view_id,
                 feature_columns, max_retries=1, initial_backoff_secs=0.5):
        self.project_id = project_id
        self.region = region
        self.online_store_id = online_store_id
        self.feature_view_id = feature_view_id
        self.feature_columns = set(feature_columns)
        self.max_retries = max_retries
        self.initial_backoff_secs = initial_backoff_secs

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
        for element in batch:
            result = self._fetch_one(element)
            if result is not None:
                yield result

    def _fetch_one(self, element):
        entity_id = element["entity_id"]
        for attempt in range(self.max_retries + 1):
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

                if len(features) == len(self.feature_columns):
                    element.update(features)
                    return element

                if attempt < self.max_retries:
                    backoff = self.initial_backoff_secs * (2 ** attempt)
                    logger.info(
                        f"Missing features for entity_id={entity_id}, "
                        f"retrying in {backoff}s"
                    )
                    time.sleep(backoff)
                else:
                    missing = self.feature_columns - set(features.keys())
                    logger.warning(
                        f"Missing features for entity_id={entity_id} "
                        f"after {self.max_retries} retries: {missing}"
                    )
                    return None

            except Exception as e:
                if attempt < self.max_retries:
                    backoff = self.initial_backoff_secs * (2 ** attempt)
                    logger.warning(
                        f"Feature fetch failed for entity_id={entity_id}, "
                        f"retrying in {backoff}s: {e}"
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"Feature fetch failed for entity_id={entity_id} "
                        f"after {self.max_retries} retries: {e}"
                    )
                    return None
```

### What changes

| Aspect | Before | After |
|---|---|---|
| Client | `FeatureOnlineStoreServiceAsyncClient` | `FeatureOnlineStoreServiceClient` |
| Event loop | `asyncio.new_event_loop()` + `set_event_loop()` in `setup()` | None |
| Concurrency | `asyncio.gather()` + `Semaphore(8)` | Sequential loop over batch |
| Sleep | `await asyncio.sleep()` | `time.sleep()` |
| `teardown()` | Closes event loop | Removed — sync client doesn't need it |
| `max_concurrent` param | Constructor arg, default 8 | Removed |
| `_fetch_batch()` | Async method orchestrating gather | Removed — logic inlined in `process()` |
| `_fetch_one()` | Async method with semaphore | Plain sync method |
| `max_retries` default | 2 (3 total attempts, up to 1s+2s=3s backoff) | 1 (2 total attempts, 0.5s backoff) |
| `initial_backoff_secs` default | 1.0 | 0.5 |

### Constructor signature change

```python
# Before
FetchFeaturesFromOnlineStore(
    project_id, region, online_store_id, feature_view_id,
    feature_columns, max_concurrent=8, max_retries=2, initial_backoff_secs=1.0
)

# After — max_concurrent removed, reduced retries and backoff
FetchFeaturesFromOnlineStore(
    project_id, region, online_store_id, feature_view_id,
    feature_columns, max_retries=1, initial_backoff_secs=0.5
)
```

The call site in `iris_inference_pipeline.py:259-266` does **not** pass `max_concurrent`, so no change is needed there.

---

## What we lose

| Capability | Impact | Mitigation |
|---|---|---|
| Concurrent feature fetches within a batch | ~50 sequential RPCs instead of 8-at-a-time concurrent | Batch sizes are small (max 50). Each Beam worker handles one batch at a time, but multiple workers run in parallel. At Iris scale this is negligible. |
| Longer retry window per entity | Was 2 retries with 1s+2s backoff (3s max). Now 1 retry with 0.5s backoff (0.5s max). | Reduces worst-case blocking from 3s to 0.5s per entity. Missing entities are dropped and retried on next Pub/Sub delivery — Pub/Sub ack deadline provides the outer retry loop. |

## What we gain

| Benefit | Detail |
|---|---|
| No event loop management | Removes the `asyncio.new_event_loop()` / `set_event_loop()` pattern that conflicts with Beam's threading model and causes the current issues |
| Simpler debugging | Stack traces are straightforward — no coroutine chains or gather tracebacks |
| No teardown needed | Sync client doesn't require explicit cleanup |
| Fewer lines | ~106 lines → ~75 lines |
| Stable on Beam workers | No async/threading interaction bugs |
| Less worker blocking | Worst-case retry sleep per entity drops from 3s to 0.5s — limits impact on subsequent elements in the batch |
