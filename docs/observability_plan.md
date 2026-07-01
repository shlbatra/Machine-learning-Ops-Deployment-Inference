# Observability Plan: Prometheus + Grafana for ML Ops

Stack choice: **Prometheus + Grafana** for the backend, **OpenTelemetry** for instrumentation.
Complements existing structured JSON logging (Cloud Logging) with real-time metrics, dashboards, and alerting.

> **Existing monitoring baseline** (see `docs/production_deployment_plan.md`): Airflow UI for DAG
> health, Vertex AI console for pipeline runs / model registry, BQ ad-hoc queries for prediction
> counts, Dataflow console for streaming job status / watermark lag, Cloud Run metrics for request
> latency / errors. This plan builds on top of that baseline — Prometheus/Grafana unifies these
> signals into a single pane with alerting, rather than checking 5 consoles manually.

---

## Stack Decision: Why OTel-First with Prometheus+Grafana Backend

### Alternatives Evaluated

| | **Prometheus + Grafana** | **SigNoz** | **Observe** |
|---|---|---|---|
| Architecture | Separate tools: Prometheus (metrics), Loki (logs), Tempo (traces), Grafana (dashboards) | Single platform: metrics + traces + logs in one ClickHouse backend | SaaS data-lake: logs + metrics + traces + AI SRE + knowledge graph |
| Instrumentation | Prometheus client libs or OpenTelemetry | OpenTelemetry-native only | OpenTelemetry + proprietary collectors |
| Signal correlation | Manual — stitch across backends with labels | Native — click from metric spike → correlated traces → logs | Native — data-lake joins + AI-driven root-cause |
| Self-hosted ops | Prometheus is simple, well-understood | ClickHouse is a real operational burden | SaaS-only, enterprise-priced |

### Why Not SigNoz Today

SigNoz is architecturally better (unified signals, OTel-native, no tool sprawl), but its main advantage — **correlated distributed traces** — can't be realized for this stack yet:

1. **OpenTelemetry tracing for Beam Python SDK is not production-ready.** Trace propagation through Dataflow workers (our heaviest component) doesn't work. The Java SDK has active OTel PRs, but Python support hasn't landed.
2. **GCP metric ingestion is less proven.** SigNoz's `googlecloudmonitoring` OTel receiver supports Cloud Run and Pub/Sub, but has no Dataflow-specific documentation. The Prometheus `stackdriver-exporter` is battle-tested for this path.
3. **Self-hosted SigNoz means running ClickHouse** — real ops burden for a project that doesn't yet have any observability infra.

### Why OTel-First Instrumentation

Even though the backend is Prometheus, we instrument with **OpenTelemetry SDK** (not `prometheus-client`) from day one:

- **Backend-agnostic.** OTel Collector exports to Prometheus today. When Beam Python gets OTel tracing support, switching the backend to SigNoz (or Observe, or Grafana Tempo) is a Collector config change — zero re-instrumentation.
- **Auto-instrumentation.** `opentelemetry-instrumentation-fastapi` auto-instruments all HTTP requests, traces, and metrics with one line. No manual middleware needed.
- **Industry convergence.** OTel is the CNCF standard. Prometheus client libs are stable but a dead-end for traces/logs.
- **Future-proof.** When we add tracing (Phase 4), the OTel SDK is already in place — we just enable the trace exporter.

### Decision

Instrument with **OpenTelemetry SDK + auto-instrumentation**. Export metrics to **Prometheus** via the **OTel Collector's Prometheus exporter**. Visualize with **Grafana**. Re-evaluate SigNoz when Beam Python SDK gets OTel tracing support.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Pub/Sub     │────▶│  Dataflow Jobs   │────▶│  BigQuery    │
│  (messages)  │     │  (Beam workers)  │     │  (sink)      │
└─────────────┘     └───────┬──────────┘     └─────────────┘
                            │
                    Beam Metrics → Cloud Monitoring
                            │
                            ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  FastAPI     │────▶│  OTel Collector  │────▶│  Prometheus      │────▶│  Grafana     │
│  (OTel SDK) │     │  (receive +      │     │  (scrape + store)│     │  (dashboards)│
└─────────────┘     │   export)        │     └───────┬──────────┘     └─────────────┘
                    └──────┬───────────┘             │
                           │                         ▼
                    stackdriver-exporter      ┌──────────────────┐
                    (GCP metrics bridge)      │  Alertmanager    │
                                              │  (Slack/PD)      │
                                              └──────────────────┘
```

**Collection patterns:**
- **FastAPI service:** OTel SDK auto-instrumentation → OTel Collector → Prometheus (via Collector's Prometheus exporter)
- **Dataflow workers:** Beam custom metrics → Cloud Monitoring → stackdriver-exporter → Prometheus
- **GCP platform metrics:** stackdriver-exporter scrapes Cloud Monitoring for Dataflow/Pub/Sub/Cloud Run/Bigtable

---

## Pillar 1: Latency

Track histograms (not averages) — you need p50/p95/p99 to catch tail latency.

### Instrumentation Points

| Metric Name | Type | Where | What It Measures |
|---|---|---|---|
| `pubsub_message_age_seconds` | Histogram | `ParsePubSubMessage.process()` | Time from Pub/Sub publish to Dataflow pickup |
| `feature_fetch_latency_seconds` | Histogram | `online_store_reader._fetch_one()` | Per-entity Bigtable round-trip (including retries) |
| `feature_write_latency_seconds` | Histogram | `online_store_writer.process()` | Bigtable direct-write batch latency |
| `fastapi_predict_latency_seconds` | Histogram | `BatchCallFastAPIService._call_async()` | FastAPI `/predict` HTTP round-trip per batch |
| `fastapi_request_duration_seconds` | Histogram | FastAPI middleware | Per-request latency at the API layer (all endpoints) |
| `end_to_end_latency_seconds` | Histogram | `AddProcessingMetadata.process()` | Pub/Sub publish timestamp → prediction complete |
| `model_load_latency_seconds` | Gauge | `fastapi_server.load_model()` | Time to download + deserialize model at startup |

### Labels (dimensions to slice by)

- `pipeline`: `feature` | `inference`
- `environment`: `staging` | `prod`
- `entity_id_prefix`: first segment (e.g. `streaming`, `batch`) — NOT full entity_id (cardinality explosion)
- `status`: `success` | `error` | `retry`

### Implementation — FastAPI Service (OTel SDK)

OTel auto-instrumentation handles request latency, status codes, and trace context
automatically — no manual middleware needed. We add custom metrics only for
ML-specific signals (prediction latency, batch size, model load time).

```python
# In fastapi_server.py
from opentelemetry import metrics, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
import time

# --- OTel setup (run once at startup, before app creation) ---
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="otel-collector:4317", insecure=True),
    export_interval_millis=10_000,
)
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
trace.set_tracer_provider(TracerProvider())

meter = metrics.get_meter("fastapi-inference")

# Auto-instruments all HTTP requests (latency, status, method, path)
# Replaces the manual metrics_middleware and /metrics endpoint
FastAPIInstrumentor.instrument_app(app)

# Custom ML-specific metrics
prediction_latency = meter.create_histogram(
    name="fastapi.predict.duration",
    description="Model prediction latency (compute only)",
    unit="s",
)
predictions_total = meter.create_counter(
    name="fastapi.predictions.total",
    description="Total predictions served",
)
batch_size_hist = meter.create_histogram(
    name="fastapi.predict.batch_size",
    description="Number of instances per /predict call",
)
model_load_time = meter.create_gauge(
    name="fastapi.model.load_duration",
    description="Time taken to load the model at startup",
    unit="s",
)

# --- Usage in /predict endpoint ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        start = time.perf_counter()
        df = pd.DataFrame(i.model_dump() for i in request.instances)
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        duration = time.perf_counter() - start

        prediction_latency.record(duration)
        predictions_total.add(len(predictions), {"status": "success"})
        batch_size_hist.record(len(request.instances))

        results = [
            Prediction(class_=int(pred), class_probabilities=proba.tolist())
            for pred, proba in zip(predictions, probabilities)
        ]
        return PredictionResponse(predictions=results)
    except Exception as e:
        predictions_total.add(1, {"status": "error"})
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
```

> **No `/metrics` endpoint needed.** OTel SDK pushes metrics to the OTel Collector via OTLP gRPC.
> The Collector then exposes them for Prometheus to scrape. This decouples the app from the
> metrics backend — swap the Collector's exporter config to switch from Prometheus to SigNoz
> without touching application code.

### Implementation — Dataflow (Beam Custom Metrics)

Beam has built-in metrics (Counters, Distributions, Gauges) that Dataflow surfaces in Cloud Monitoring. For Prometheus, two options:

**Option A: Beam metrics → Cloud Monitoring → Prometheus (via stackdriver-exporter)**
- Lowest effort. Use `beam.metrics.Metrics` natively, then run `stackdriver-exporter` sidecar to expose them as Prometheus metrics.

**Option B: Pushgateway (direct push from DoFn)**
- More control. Push from `setup()`/`teardown()` or periodically from `process()`.
- Risk: Pushgateway is designed for batch jobs, not long-running streaming. Use with `grouping_key` per worker.

**Recommended: Option A** — use Beam's native metrics, export to Prometheus via stackdriver-exporter. Simpler, no Pushgateway to manage.

```python
# In online_store_reader.py (sync client — see _fetch_one)
from apache_beam.metrics import Metrics

class FetchFeaturesFromOnlineStore(beam.DoFn):
    def __init__(self, ...):
        ...
        self.fetch_latency = Metrics.distribution(self.__class__.__name__, "feature_fetch_latency_ms")
        self.fetch_success = Metrics.counter(self.__class__.__name__, "feature_fetch_success")
        self.fetch_failure = Metrics.counter(self.__class__.__name__, "feature_fetch_failure")
        self.fetch_retry = Metrics.counter(self.__class__.__name__, "feature_fetch_retry")

    def _fetch_one(self, element):
        entity_id = element["entity_id"]
        start = time.monotonic()
        retry_count = 0
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
                    elapsed_ms = (time.monotonic() - start) * 1000
                    self.fetch_latency.update(int(elapsed_ms))
                    self.fetch_success.inc()
                    element.update(features)
                    element["feature_fetch_latency_ms"] = elapsed_ms
                    element["feature_fetch_retry_count"] = retry_count
                    return element

                if attempt < self.max_retries:
                    retry_count += 1
                    self.fetch_retry.inc()
                    backoff = self.initial_backoff_secs * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    self.fetch_failure.inc()
                    return None

            except Exception as e:
                if attempt < self.max_retries:
                    retry_count += 1
                    self.fetch_retry.inc()
                    backoff = self.initial_backoff_secs * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    self.fetch_failure.inc()
                    return None
```

---

## Pillar 2: Cost

### What to Track

| Cost Driver | How to Measure | Metric / Data Source |
|---|---|---|
| Dataflow worker-hours | Beam metrics: worker count, uptime | `dataflow_worker_count` (Cloud Monitoring, export to Prometheus) |
| Dataflow vCPU-hours | Worker type × count × time | Cloud Monitoring `job/elapsed_time`, `job/current_num_vcpus` |
| Bigtable read units | Feature fetches per prediction | `feature_fetch_success` counter × avg read size |
| Bigtable write units | Feature ingestion writes | `online_store_write_success` counter × avg batch size |
| BigQuery streaming inserts | Rows written to predictions + features tables | `bq_rows_written_total` counter, label by `table` |
| FastAPI compute | Cloud Run instance-seconds | Cloud Run built-in metrics → Prometheus via stackdriver-exporter |
| Pub/Sub messages | Messages published + delivered | Pub/Sub built-in metrics → Prometheus via stackdriver-exporter |

### Cost Attribution Labels

Every metric should carry a `pipeline` label (`feature` | `inference` | `training`) so Grafana dashboards can show cost per pipeline.

### Implementation

Most cost signals come from GCP built-in metrics. Use **stackdriver-exporter** to bridge them into Prometheus:

```yaml
# stackdriver-exporter config
monitoring.metrics-type-prefixes:
  - dataflow.googleapis.com/job
  - bigtable.googleapis.com/server
  - pubsub.googleapis.com/topic
  - pubsub.googleapis.com/subscription
  - run.googleapis.com/container
  - bigquery.googleapis.com/storage
```

For application-level cost attribution, add counters in the Beam DoFns:

```python
# In online_store_writer.py
self.rows_written = Metrics.counter(self.__class__.__name__, "online_store_rows_written")
self.batches_written = Metrics.counter(self.__class__.__name__, "online_store_batches_written")

# In iris_inference_pipeline.py (BQ write)
# BigQuery write metrics come from Dataflow's built-in BQ sink metrics
```

### Grafana Cost Dashboard

- Row 1: **Daily cost estimate** — compute (Dataflow workers × $/hr) + storage (BQ bytes × $/TB) + reads (Bigtable ops × $/M)
- Row 2: **Cost per prediction** — total daily cost / predictions_total
- Row 3: **Cost breakdown by pipeline** — stacked bar chart, feature vs. inference vs. training
- Row 4: **Cost trend** — 7-day rolling average to spot creep

---

## Pillar 3: Errors

### Error Counters

| Metric Name | Type | Where | What It Catches |
|---|---|---|---|
| `pubsub_parse_error_total` | Counter | `ParsePubSubMessage` (both pipelines) | JSON decode failures, missing fields |
| `pydantic_validation_error_total` | Counter | `ParsePubSubMessage` (feature pipeline) | Schema mismatch on incoming messages |
| `feature_fetch_failure_total` | Counter | `online_store_reader` | Entities dropped after max retries |
| `feature_fetch_missing_total` | Counter | `online_store_reader` | Incomplete feature sets (partial data in Bigtable) |
| `prediction_error_total` | Counter | `BatchCallFastAPIService` | FastAPI call failures (timeout, 5xx, connection refused) |
| `online_store_write_failure_total` | Counter | `online_store_writer` | Bigtable write errors |
| `fastapi_error_total` | Counter | FastAPI middleware | 4xx and 5xx responses |
| `model_load_failure_total` | Counter | `fastapi_server.load_model()` | Model download or deserialization failures |

### Labels for Error Metrics

- `error_type`: `json_decode` | `missing_field` | `validation` | `timeout` | `connection` | `5xx` | `bigtable_error`
- `pipeline`: `feature` | `inference`
- `recoverable`: `true` | `false` (did we retry and succeed, or drop it?)

### Dead Letter Pattern

Currently, failed entities are silently dropped (return `None` in reader) or written as `ERROR` prediction rows (in `BatchCallFastAPIService`). Replace with a dead letter queue:

```python
# Option A: Dead letter Pub/Sub topic
# Route failed entities to a separate topic for later inspection + replay

# Option B: Dead letter BigQuery table
# Write failed entities with error context to a dedicated table:
DEAD_LETTER_SCHEMA = {
    "fields": [
        {"name": "entity_id", "type": "STRING"},
        {"name": "pipeline", "type": "STRING"},          # feature | inference
        {"name": "stage", "type": "STRING"},              # parse | fetch | predict | write
        {"name": "error_type", "type": "STRING"},
        {"name": "error_message", "type": "STRING"},
        {"name": "original_message", "type": "STRING"},   # raw Pub/Sub payload
        {"name": "timestamp", "type": "TIMESTAMP"},
        {"name": "retry_count", "type": "INTEGER"},
    ]
}
```

**Recommendation:** Use a dead letter BQ table (`ml_dataset.dead_letters`). Easier to query than a Pub/Sub topic, and you can build a Grafana dashboard panel that shows recent dead letters with drill-down.

### Error Rate Alerts (Alertmanager)

```yaml
# alertmanager rules
groups:
  - name: ml-pipeline-errors
    rules:
      - alert: HighPredictionErrorRate
        expr: rate(prediction_error_total[5m]) / rate(predictions_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Prediction error rate > 5% for 5 minutes"

      - alert: FeatureFetchFailureSpike
        expr: rate(feature_fetch_failure_total[5m]) > 10
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Feature fetch failures spiking — check Bigtable / online store"

      - alert: NoPredicitionsFlowing
        expr: rate(predictions_total[10m]) == 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Zero predictions in last 10 minutes — pipeline may be down"

      - alert: HighEndToEndLatency
        expr: histogram_quantile(0.95, rate(end_to_end_latency_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p95 end-to-end latency exceeds 10 seconds"
```

---

## Pillar 4: Investigations

### Trace ID Propagation

Attach a correlation ID to every entity as it flows through the system. Use the Pub/Sub `message_id` as the natural trace ID.

```
Pub/Sub message (message_id=abc123)
  → ParsePubSubMessage: log {trace_id: abc123, stage: parse, entity_id: 42_streaming}
  → FetchFeatures:      log {trace_id: abc123, stage: fetch, entity_id: 42_streaming, latency_ms: 12}
  → BatchCallFastAPI:   log {trace_id: abc123, stage: predict, entity_id: 42_streaming, prediction: setosa}
  → WriteToBigQuery:    log {trace_id: abc123, stage: write, entity_id: 42_streaming}
```

Implementation — propagate `message_id` through the pipeline:

```python
# In ParsePubSubMessage, capture the Pub/Sub message ID
# Beam's ReadFromPubSub can output (message_data, attributes) or PubsubMessage objects
# Switch to ReadFromPubSub(topic=..., with_attributes=True) to get message_id

class ParsePubSubMessage(beam.DoFn):
    def process(self, element):
        # element is now a PubsubMessage with .data and .attributes and .message_id
        message_id = element.message_id  # or element.attributes.get("trace_id")
        message_data = json.loads(element.data.decode("utf-8"))
        ...
        yield {
            "entity_id": entity_id,
            "trace_id": message_id,  # propagate through all stages
            "timestamp": message_data.get("timestamp"),
        }
```

### Feature Snapshot at Prediction Time

Feature values are **already stored** in the `features` column as a JSON string (`iris_inference_pipeline.py:142-145`):

```python
features = {col: element[col] for col in FEATURE_COLUMNS}
row = {
    ...
    "features": json.dumps(features),  # all 4 feature values serialized
}
```

You can already query what the model saw with `JSON_EXTRACT` in BQ. What's **missing** is the metadata to fully trace a prediction:

| Field | Why It's Needed | Where to Capture |
|---|---|---|
| `trace_id` | Correlate a prediction back to its Pub/Sub message across all pipeline stages | `ParsePubSubMessage` — requires `ReadFromPubSub(with_attributes=True)` to get `message_id` |
| `feature_fetch_latency_ms` | Debug slow predictions — is it the feature store or the model? | `online_store_reader._fetch_one()` — wrap the fetch call with `time.monotonic()` |
| `feature_fetch_retry_count` | Know if predictions relied on retried (potentially stale) feature fetches | `online_store_reader._fetch_one()` — increment counter per retry in the backoff loop |
| `prediction_retry_count` | Know if FastAPI call needed retries (network issues, overload) | `BatchCallFastAPIService._call_async()` — track `attempt` value on success |

To add these, propagate them through the element dict and extend `PREDICTION_SCHEMA`:

```python
# Add to PREDICTION_SCHEMA["fields"]:
{"name": "trace_id", "type": "STRING", "mode": "NULLABLE"},
{"name": "feature_fetch_latency_ms", "type": "FLOAT", "mode": "NULLABLE"},
{"name": "feature_fetch_retry_count", "type": "INTEGER", "mode": "NULLABLE"},
{"name": "prediction_retry_count", "type": "INTEGER", "mode": "NULLABLE"},
```

This lets you answer: "What did the model see when it predicted X for entity Y at time T, and how healthy was the path to get there?"

### Investigation Queries

> **Note:** Batch inference uses `WRITE_APPEND` — the `iris_predictions` table accumulates
> across runs. Always filter by `prediction_timestamp` to isolate a single run's results.

```sql
-- Trace a single entity through the system (streaming)
SELECT *
FROM `ml_dataset.iris_predictions_streaming`
WHERE entity_id = '42_streaming'
ORDER BY prediction_timestamp DESC
LIMIT 10;

-- Latest batch inference run results
SELECT *
FROM `ml_dataset.iris_predictions`
WHERE prediction_timestamp = (
  SELECT MAX(prediction_timestamp) FROM `ml_dataset.iris_predictions`
);

-- Compare batch run sizes over time (catches data drift in source table)
SELECT
  prediction_timestamp,
  COUNT(*) as row_count,
  COUNT(DISTINCT prediction) as distinct_classes
FROM `ml_dataset.iris_predictions`
GROUP BY prediction_timestamp
ORDER BY prediction_timestamp DESC
LIMIT 10;

-- Find all dead letters in the last hour
SELECT *
FROM `ml_dataset.dead_letters`
WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
ORDER BY timestamp DESC;

-- Correlate features written vs. features read (freshness gap)
SELECT
  p.entity_id,
  p.prediction_timestamp,
  f.feature_timestamp,
  TIMESTAMP_DIFF(p.prediction_timestamp, f.feature_timestamp, SECOND) as feature_age_seconds
FROM `ml_dataset.iris_predictions_streaming` p
JOIN `ml_dataset.iris_features` f
  ON p.entity_id = f.entity_id
WHERE p.prediction_timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
ORDER BY feature_age_seconds DESC;

-- Prediction distribution over time (drift detection)
SELECT
  TIMESTAMP_TRUNC(prediction_timestamp, HOUR) as hour,
  prediction,
  COUNT(*) as count
FROM `ml_dataset.iris_predictions_streaming`
WHERE prediction_timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
GROUP BY hour, prediction
ORDER BY hour, prediction;

-- Diagnose slow predictions using feature snapshot metadata (after Phase 4)
SELECT
  entity_id,
  trace_id,
  feature_fetch_latency_ms,
  feature_fetch_retry_count,
  prediction_retry_count,
  processing_time,
  JSON_EXTRACT(features, '$.sepal_length_cm') as sepal_length
FROM `ml_dataset.iris_predictions_streaming`
WHERE prediction_timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
  AND (feature_fetch_retry_count > 0 OR prediction_retry_count > 0)
ORDER BY feature_fetch_latency_ms DESC;
```

### Grafana Investigation Dashboard

- **Entity Lookup Panel**: text input for entity_id, shows full trace across stages
- **Dead Letters Panel**: table of recent failures with error_type, stage, message
- **Feature Freshness Panel**: heatmap of feature_age_seconds — how stale are features at prediction time
- **Prediction Distribution Panel**: time-series of class distribution — visual drift detection

---

## Grafana Dashboard Layout

### Dashboard 1: Pipeline Health (ops team daily view)

| Row | Panels |
|---|---|
| **Row 1: Traffic** | Predictions/sec (timeseries), Messages in Pub/Sub (gauge), Active Dataflow workers (gauge) |
| **Row 2: Latency** | End-to-end p50/p95/p99 (timeseries), Feature fetch latency (heatmap), FastAPI latency (timeseries) |
| **Row 3: Errors** | Error rate % (timeseries), Dead letters/hr (stat), Error breakdown by type (pie chart) |
| **Row 4: Cost** | Daily estimated cost (stat), Cost per prediction (timeseries), Cost by pipeline (stacked bar) |

### Dashboard 2: ML Model Health (ML team weekly view)

| Row | Panels |
|---|---|
| **Row 1: Predictions** | Prediction class distribution over time (stacked area), Prediction volume (timeseries) |
| **Row 2: Features** | Feature freshness distribution (histogram), Missing features rate (timeseries) |
| **Row 3: Model** | Model version in production (stat), Time since last training (stat), Accuracy if ground truth available (timeseries) |

### Dashboard 3: Investigation (on-call, ad-hoc)

| Row | Panels |
|---|---|
| **Row 1: Entity Trace** | Variable: entity_id → full trace table |
| **Row 2: Dead Letters** | Recent failures table, filterable by stage/error_type |
| **Row 3: Feature Correlation** | Feature values at prediction time vs. training distribution |

---

## Infrastructure Setup

### Deployment (Docker Compose for local dev, GKE for prod)

```yaml
# docker-compose.observability.yml (local development)
services:
  # Receives OTLP from FastAPI, exposes /metrics for Prometheus to scrape
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    ports:
      - "4317:4317"   # OTLP gRPC (FastAPI pushes here)
      - "8889:8889"   # Prometheus exporter (Prometheus scrapes here)
    volumes:
      - ./observability/otel-collector.yml:/etc/otelcol-contrib/config.yaml

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./observability/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./observability/alert_rules.yml:/etc/prometheus/alert_rules.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./observability/grafana/provisioning:/etc/grafana/provisioning
      - ./observability/grafana/dashboards:/var/lib/grafana/dashboards

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./observability/alertmanager.yml:/etc/alertmanager/alertmanager.yml

  # Bridge GCP metrics into Prometheus
  stackdriver-exporter:
    image: prometheuscommunity/stackdriver-exporter:latest
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/credentials/sa.json
      - STACKDRIVER_EXPORTER_GOOGLE_PROJECT_ID=deeplearning-sahil
      - STACKDRIVER_EXPORTER_MONITORING_METRICS_TYPE_PREFIXES=dataflow.googleapis.com/job,bigtable.googleapis.com/server,pubsub.googleapis.com,run.googleapis.com
    ports:
      - "9255:9255"
    volumes:
      - ./deeplearning-sahil-e50332de6687.json:/credentials/sa.json
```

### OTel Collector Config

```yaml
# observability/otel-collector.yml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  batch:
    timeout: 10s

exporters:
  prometheus:
    endpoint: 0.0.0.0:8889
    namespace: fastapi

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
    # Uncomment when Beam Python gets OTel tracing support:
    # traces:
    #   receivers: [otlp]
    #   processors: [batch]
    #   exporters: [otlp/signoz]  # swap backend without re-instrumenting
```

### Prometheus Scrape Config

```yaml
# observability/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  # OTel Collector exposes FastAPI metrics via its Prometheus exporter
  - job_name: "otel-collector"
    static_configs:
      - targets: ["otel-collector:8889"]
    scrape_interval: 10s

  - job_name: "stackdriver-exporter"
    static_configs:
      - targets: ["stackdriver-exporter:9255"]
    scrape_interval: 60s
```

---

## Implementation Order

### Phase 1: FastAPI Metrics (lowest effort, highest signal) -- DONE
1. ~~Add OTel dependencies to `requirements.fastapi.txt`~~ (see Dependencies below)
2. ~~Add OTel SDK setup + `FastAPIInstrumentor.instrument_app(app)` to `fastapi_server.py`~~
3. ~~Add custom ML metrics: prediction latency, batch size, predictions total~~ (see Implementation above)
4. ~~Deploy OTel Collector + Prometheus + Grafana (docker-compose locally)~~ — see `observability/`
5. ~~Build Dashboard 1 (Pipeline Health)~~ — see `observability/grafana/dashboards/pipeline-health.json`

### Phase 2: Beam / Dataflow Metrics -- DONE
1. ~~Add `Metrics.counter()` and `Metrics.distribution()` to all DoFns~~
2. ~~Deploy stackdriver-exporter to bridge Dataflow metrics → Prometheus~~
3. Add feature fetch latency, error counters, write latency to dashboards (Phase 5 — Grafana panels)

### Phase 3: Error Handling + Dead Letters -- DONE
1. ~~Create `ml_dataset.dead_letters` BQ table~~ — run DDL manually in BQ console
2. ~~Add dead letter routing to `ParsePubSubMessage`, `FetchFeaturesFromOnlineStore`, `BatchCallFastAPIService`~~ — via Beam tagged outputs
3. ~~Remove `ERROR` prediction rows pattern~~ — replaced with dead letter side output
4. ~~Add Alertmanager rules for error rate thresholds~~ — see `observability/alert_rules.yml`
5. ~~Build Dead Letters panel in Grafana~~ — see `observability/grafana/dashboards/dead-letters.json`

### Phase 4: Tracing + Investigation -- DONE
Feature values are already stored as JSON in the `features` column — this phase adds the missing tracing and diagnostic metadata.

1. ~~Switch to `ReadFromPubSub(with_attributes=True)` to capture `message_id` as `trace_id`~~ — see `iris_inference_pipeline.py` `ParsePubSubMessage`
2. ~~Propagate `trace_id` through all pipeline stages (element dict → BQ row)~~ — flows from parse → fetch → predict → BQ row
3. ~~Instrument `online_store_reader._fetch_one()` to capture `feature_fetch_latency_ms` and `feature_fetch_retry_count` per entity~~ — see `online_store_reader.py`
4. ~~Capture `prediction_retry_count` in `BatchCallFastAPIService._call_async()` (the `attempt` value on success)~~ — see `iris_inference_pipeline.py`
5. ~~Extend `PREDICTION_SCHEMA` with the 4 new fields and update BQ table DDL~~ — `trace_id`, `feature_fetch_latency_ms`, `feature_fetch_retry_count`, `prediction_retry_count`
6. ~~Build Investigation dashboard (entity lookup, feature correlation, slow-prediction diagnosis)~~ — see `observability/grafana/dashboards/investigation.json`
7. Create BQ views for common investigation queries (deferred — queries documented in dashboard text panel)

### Phase 5: Cost Attribution
1. Deploy stackdriver-exporter with Dataflow/Bigtable/BQ/Pub/Sub metric prefixes
2. Add `pipeline` label to all custom metrics
3. Build cost dashboard with per-pipeline attribution
4. Set up cost anomaly alerts (daily spend > 2× rolling average)

---

## Dependencies to Add

```
# requirements.fastapi.txt (add)
opentelemetry-api>=1.25.0
opentelemetry-sdk>=1.25.0
opentelemetry-instrumentation-fastapi>=0.46b0
opentelemetry-exporter-otlp-proto-grpc>=1.25.0

# pyproject.toml (add to beam/dataflow deps — Beam metrics are native, no OTel deps needed)
# No additional deps for Phase 2 (Beam Metrics are built-in)
# OTel deps only needed if/when Beam Python gets OTel tracing support (Phase 4+)
```

---

## File Structure

```
observability/
├── otel-collector.yml                # OTel Collector: receive OTLP, export to Prometheus
├── prometheus.yml                    # Scrape config (scrapes OTel Collector + stackdriver-exporter)
├── alert_rules.yml                   # Alertmanager rules
├── alertmanager.yml                  # Notification routing (Slack, PagerDuty)
├── docker-compose.observability.yml  # Local dev stack
└── grafana/
    ├── provisioning/
    │   ├── datasources/
    │   │   └── prometheus.yml        # Auto-configure Prometheus datasource
    │   └── dashboards/
    │       └── dashboards.yml        # Auto-load dashboard JSON
    └── dashboards/
        ├── pipeline-health.json      # Dashboard 1
        ├── ml-model-health.json      # Dashboard 2
        └── investigation.json        # Dashboard 3
```
