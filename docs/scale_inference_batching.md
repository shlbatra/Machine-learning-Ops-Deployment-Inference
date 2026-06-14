# Plan: Scale Inference — Micro-Batching + Async HTTP

## Context

The current Dataflow streaming pipeline (`iris_streaming_pipeline.py`) makes **one synchronous HTTP call per Pub/Sub message**. For every single iris sample, it:
1. Builds a payload with 1 instance
2. Calls `requests.post()` synchronously (blocking the worker thread for ~30-100ms of network round-trip)
3. Waits for the response before processing the next message

This means throughput is capped at ~10-30 messages/sec/worker, dominated by network latency — not model inference time. The FastAPI server already accepts `List[Dict]` and calls `model.predict(df)` on the full batch at once, so sending 50 instances costs nearly the same server-side time as sending 1.

Two changes fix this:
1. **Micro-batching** — use `BatchElements` with `max_batch_duration_secs` to collect up to 50 messages before calling `/predict` once, flushing partial batches after 1 second at low traffic
2. **Async HTTP** — use `aiohttp` instead of `requests` to overlap network I/O across concurrent batch calls within a worker

## Current pipeline flow

```
Pub/Sub → Parse JSON → [1 msg] → HTTP POST (1 instance) → Add Metadata → BigQuery
                        [1 msg] → HTTP POST (1 instance) → ...
                        [1 msg] → HTTP POST (1 instance) → ...
```

## Target pipeline flow

```
Pub/Sub → Parse JSON → BatchElements(max=50, flush=1s) → [up to 50 msgs] → async HTTP POST
                                                                             → unbundle → Add Metadata → BigQuery
```

No `FixedWindows` needed. `BatchElements` with `max_batch_duration_secs=1` uses Beam's stateful processing (State & Timers API) to batch across bundles. At high traffic, batches fill to 50 and flush immediately with near-zero latency. At low traffic, partial batches flush after at most 1 second — avoiding the problem where default `BatchElements` only batches within a bundle (which on Dataflow streaming is often size 1, making it a no-op).

## Changes

All changes are in a single file: `src/dataflow/iris_streaming_pipeline.py`

### 1. Add micro-batching to the pipeline

Insert `BatchElements` with `max_batch_duration_secs` between Parse and the HTTP call. This tells Beam: "collect up to 50 messages, flushing immediately when full or after 1 second if the batch is still partial."

```python
from apache_beam.transforms.util import BatchElements

# In the pipeline construction, replace the current chain:
predictions = (
    pipeline
    | "Read from Pub/Sub" >> ReadFromPubSub(topic=known_args.input_topic)
    | "Parse JSON" >> beam.ParDo(ParsePubSubMessage())
    | "Batch Elements" >> BatchElements(
        min_batch_size=1,
        max_batch_size=50,
        max_batch_duration_secs=1,
    )
    | "Call FastAPI Batch" >> beam.ParDo(
        BatchCallFastAPIService(known_args.service_url)
    )
    | "Add Metadata" >> beam.ParDo(AddProcessingMetadata())
    | "Write to BigQuery" >> WriteToBigQuery(...)
)
```

**Why these parameters:**
- `max_batch_size=50` — matches a reasonable HTTP payload size. The FastAPI server builds a pandas DataFrame from the instances; 50 rows is trivial. Going higher (500+) risks HTTP timeouts and large retry payloads.
- `max_batch_duration_secs=1` — activates the **stateful implementation** (Beam State & Timers API, requires Beam 2.52+). Without this, default `BatchElements` only batches within a single bundle — on Dataflow streaming, bundles are frequently size 1 at low throughput, making the transform a no-op. With it, elements are batched across bundles and partial batches flush after 1 second. The timing is best-effort — actual hold time may slightly exceed 1s.
- `min_batch_size=1` — ensures single messages still flow through at low traffic instead of blocking until 50 arrive.
- **No `FixedWindows` needed** — the `max_batch_duration_secs` timer replaces the fixed window. This avoids adding an artificial 5-second latency floor. At high traffic, batches fill to 50 and flush immediately. At low traffic, worst case is ~1 second.

**Tradeoff:** The stateful path requires internal keying, which triggers a shuffle (network transfer between workers). For small payloads like iris features this is negligible, but worth noting for larger payloads.

**Adding CLI args for tuning:**

```python
parser.add_argument("--batch_size", type=int, default=50,
                    help="Max instances per /predict call")
parser.add_argument("--max_batch_duration_secs", type=float, default=1.0,
                    help="Max seconds to buffer a partial batch before flushing")
```

### 2. Replace `CallFastAPIService` with `BatchCallFastAPIService`

The new DoFn receives a `List[Dict]` (a batch of parsed messages) instead of a single element. It builds one payload with all instances, makes one HTTP call, then unbundles the response back to individual records.

```python
class BatchCallFastAPIService(beam.DoFn):
    """Call FastAPI with a batch of instances in one request."""

    def __init__(self, service_url):
        self.service_url = service_url
        self.predict_url = f"{service_url}/predict"

    def setup(self):
        import requests
        self.session = requests.Session()

    def process(self, batch):
        import time
        from datetime import datetime

        start_time = time.time()

        # Build batch payload
        instances = []
        for element in batch:
            instances.append({
                "SepalLengthCm": element["sepal_length"],
                "SepalWidthCm": element["sepal_width"],
                "PetalLengthCm": element["petal_length"],
                "PetalWidthCm": element["petal_width"],
            })

        try:
            response = self.session.post(
                self.predict_url,
                json={"instances": instances},
                timeout=30,
            )
            response.raise_for_status()

            predictions = response.json().get("predictions", [])
            processing_time = time.time() - start_time

            # Unbundle: zip each input element with its prediction
            for element, pred in zip(batch, predictions):
                predicted_class = str(pred.get("prediction", "unknown"))
                yield {
                    "sepal_length": element["sepal_length"],
                    "sepal_width": element["sepal_width"],
                    "petal_length": element["petal_length"],
                    "petal_width": element["petal_width"],
                    "timestamp": element.get("timestamp", datetime.utcnow().isoformat()),
                    "sample_id": element.get("sample_id", 0),
                    "prediction": predicted_class,
                    "prediction_timestamp": datetime.utcnow().isoformat(),
                    "model_service": self.service_url,
                    "processing_time": processing_time / len(batch),
                }

        except Exception as e:
            logging.error(f"Batch prediction failed ({len(batch)} instances): {e}")
            # Yield error records for every element in the failed batch
            processing_time = time.time() - start_time
            for element in batch:
                yield {
                    "sepal_length": element.get("sepal_length", 0.0),
                    "sepal_width": element.get("sepal_width", 0.0),
                    "petal_length": element.get("petal_length", 0.0),
                    "petal_width": element.get("petal_width", 0.0),
                    "timestamp": element.get("timestamp", datetime.utcnow().isoformat()),
                    "sample_id": element.get("sample_id", 0),
                    "prediction": "ERROR",
                    "prediction_timestamp": datetime.utcnow().isoformat(),
                    "model_service": f"ERROR: {str(e)}",
                    "processing_time": processing_time,
                }
```

**Key details:**
- Uses `setup()` to create a persistent `requests.Session` — reuses TCP connections across batches instead of opening a new connection per call
- Error handling yields error records for every element in the failed batch (same behavior as current, but batched)
- `processing_time` is divided by batch size to give per-element time

### 3. Add async HTTP for concurrent batch calls

Once batching is in place, the next win is overlapping HTTP calls across batches. Within a single worker, multiple batches from the same window can be in-flight concurrently.

Replace `requests.Session` with `aiohttp` using Beam's async DoFn support (`beam.DoFn` with async `process_batch` isn't available in all versions). The practical approach for Beam 2.50+ is to use a thread pool inside the DoFn:

```python
class BatchCallFastAPIService(beam.DoFn):
    """Call FastAPI with a batch of instances, using async HTTP."""

    def __init__(self, service_url, max_concurrent=4):
        self.service_url = service_url
        self.predict_url = f"{service_url}/predict"
        self.max_concurrent = max_concurrent

    def setup(self):
        import aiohttp
        import asyncio
        self._loop = asyncio.new_event_loop()
        self._connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            loop=self._loop,
        )
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            loop=self._loop,
        )

    def teardown(self):
        self._loop.run_until_complete(self._session.close())
        self._loop.close()

    def process(self, batch):
        results = self._loop.run_until_complete(self._call_async(batch))
        yield from results

    async def _call_async(self, batch):
        import time
        from datetime import datetime

        start_time = time.time()

        instances = [
            {
                "SepalLengthCm": e["sepal_length"],
                "SepalWidthCm": e["sepal_width"],
                "PetalLengthCm": e["petal_length"],
                "PetalWidthCm": e["petal_width"],
            }
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
                results.append({
                    "sepal_length": element["sepal_length"],
                    "sepal_width": element["sepal_width"],
                    "petal_length": element["petal_length"],
                    "petal_width": element["petal_width"],
                    "timestamp": element.get("timestamp", datetime.utcnow().isoformat()),
                    "sample_id": element.get("sample_id", 0),
                    "prediction": predicted_class,
                    "prediction_timestamp": datetime.utcnow().isoformat(),
                    "model_service": self.service_url,
                    "processing_time": processing_time / len(batch),
                })
            return results

        except Exception as e:
            logging.error(f"Batch prediction failed ({len(batch)} instances): {e}")
            processing_time = time.time() - start_time
            return [
                {
                    "sepal_length": el.get("sepal_length", 0.0),
                    "sepal_width": el.get("sepal_width", 0.0),
                    "petal_length": el.get("petal_length", 0.0),
                    "petal_width": el.get("petal_width", 0.0),
                    "timestamp": el.get("timestamp", datetime.utcnow().isoformat()),
                    "sample_id": el.get("sample_id", 0),
                    "prediction": "ERROR",
                    "prediction_timestamp": datetime.utcnow().isoformat(),
                    "model_service": f"ERROR: {str(e)}",
                    "processing_time": processing_time,
                }
                for el in batch
            ]
```

**Why this pattern:**
- Beam DoFns run in threads, not an async event loop. Creating a dedicated `asyncio` loop in `setup()` and driving it with `run_until_complete()` is the standard approach for async I/O in Beam.
- `TCPConnector(limit=4)` caps concurrent connections per worker — prevents overwhelming the Cloud Run service.
- `setup()`/`teardown()` lifecycle ensures the session is created once per worker, not per bundle.

### 4. Add `aiohttp` dependency

`aiohttp` is already in `pyproject.toml` (line 20: `aiohttp>=3.9.3`), so no new dependency needed for the project install.

However, for Dataflow workers, the pipeline uses `--requirements_file` or `--setup_file` to install deps on workers. Currently the deploy script doesn't pass either — it relies on Beam's default packaging. You may need to add:

```bash
--setup_file ./setup.py
# or
--requirements_file requirements-dataflow.txt
```

where `requirements-dataflow.txt` contains:
```
aiohttp>=3.9.3
requests>=2.31.0
```

Alternatively, since `pyproject.toml` already declares `aiohttp` and the pipeline is installed via `pip install -e .`, passing `--setup_file` pointing to the project should work.

### 5. Remove old `CallFastAPIService` class

Delete the old single-element `CallFastAPIService` DoFn since it's fully replaced by `BatchCallFastAPIService`.

## Implementation options

You can do this in two phases or all at once:

| Phase | Change | Throughput gain | Complexity |
|---|---|---|---|
| **Phase 1** | Stateful micro-batching + `requests.Session` (step 1-2) | ~10-50x (amortize network latency over 50 msgs) | Low — straightforward Beam transforms |
| **Phase 2** | Async HTTP with `aiohttp` (step 3) | ~2-4x on top of phase 1 (overlap concurrent batches) | Medium — async event loop management |

Phase 1 alone gives the biggest win. Phase 2 adds concurrency on top. If traffic is moderate (~100s msgs/sec), phase 1 may be sufficient.

## File summary

| File | Action |
|---|---|
| `src/dataflow/iris_streaming_pipeline.py` | Add `BatchElements` with `max_batch_duration_secs`, replace `CallFastAPIService` with `BatchCallFastAPIService`, add `--batch_size` and `--max_batch_duration_secs` CLI args |
| `.github/workflows/deploy-dataflow.yaml` | Add `--batch_size` and `--max_batch_duration_secs` flags if overriding defaults |

No changes to `fastapi_server.py` — it already handles batched instances.

## Verification

1. **Unit test locally with DirectRunner (staging):**
   ```bash
   python src/dataflow/iris_streaming_pipeline.py \
       --input_topic projects/deeplearning-sahil/topics/iris-inference-data \
       --output_table deeplearning-sahil:ml_dataset.iris_predictions_streaming_staging \
       --project_id deeplearning-sahil \
       --region us-central1 \
       --service_url https://iris-classifier-xgboost-service-staging-zoxyfmo73q-uc.a.run.app \
       --runner DirectRunner \
       --streaming
   ```
   Run the pubsub producer in parallel and verify predictions land in the staging BigQuery table.

2. **Deploy to staging via GitHub Action:** Trigger the `Deploy Dataflow Streaming` workflow with `environment=staging` to validate on Dataflow workers.

3. **Check batch sizes in logs:** Look for `Batch prediction failed (N instances)` log pattern — if N is consistently 50, batching is working. At low traffic, N will be smaller (down to 1) which is expected.

4. **Compare throughput:** Before and after, monitor Dataflow job metrics:
   - Elements processed/sec (should increase ~10-50x)
   - System lag (should decrease — messages spend less time waiting)
   - Worker CPU (should decrease — less time idle on network I/O)

5. **Verify no data loss:** Count Pub/Sub acked messages vs BigQuery rows inserted over a time window. Should match.

6. **Promote to prod:** After staging validation, trigger the workflow with `environment=prod`.
