# Observability Plan: Prometheus + Grafana for ML Ops

Stack choice: **Prometheus + Grafana** — portable, open-source, industry-standard.
Complements existing structured JSON logging (Cloud Logging) with real-time metrics, dashboards, and alerting.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Pub/Sub     │────▶│  Dataflow Jobs   │────▶│  BigQuery    │
│  (messages)  │     │  (Beam workers)  │     │  (sink)      │
└─────────────┘     └───────┬──────────┘     └─────────────┘
                            │
                    pushgateway (batch)
                            │
                            ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  FastAPI     │────▶│  Prometheus      │────▶│  Grafana     │
│  /metrics   │     │  (scrape + store)│     │  (dashboards)│
└─────────────┘     └───────┬──────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │  Alertmanager    │
                    │  (Slack/PD)      │
                    └──────────────────┘
```

**Two collection patterns:**
- FastAPI service: native `/metrics` endpoint (Prometheus scrapes directly)
- Dataflow workers: push to Prometheus Pushgateway (short-lived workers can't be scraped reliably), OR use Beam custom metrics + a sidecar exporter

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

### Implementation — FastAPI Service

```python
# In fastapi_server.py
from prometheus_client import Histogram, Counter, Gauge, generate_latest
from starlette.responses import Response
import time

REQUEST_LATENCY = Histogram(
    "fastapi_request_duration_seconds",
    "Request latency",
    ["method", "endpoint", "status_code"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_LATENCY = Histogram(
    "fastapi_predict_latency_seconds",
    "Model prediction latency (compute only)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

MODEL_LOAD_TIME = Gauge(
    "model_load_latency_seconds",
    "Time taken to load the model at startup",
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions served",
    ["model_type", "status"],
)

BATCH_SIZE = Histogram(
    "prediction_batch_size",
    "Number of instances per /predict call",
    buckets=[1, 5, 10, 25, 50, 100],
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).observe(duration)
    return response

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

### Implementation — Dataflow (Beam Custom Metrics)

Beam has built-in metrics (Counters, Distributions, Gauges) that Dataflow surfaces in Cloud Monitoring. For Prometheus, two options:

**Option A: Beam metrics → Cloud Monitoring → Prometheus (via stackdriver-exporter)**
- Lowest effort. Use `beam.metrics.Metrics` natively, then run `stackdriver-exporter` sidecar to expose them as Prometheus metrics.

**Option B: Pushgateway (direct push from DoFn)**
- More control. Push from `setup()`/`teardown()` or periodically from `process()`.
- Risk: Pushgateway is designed for batch jobs, not long-running streaming. Use with `grouping_key` per worker.

**Recommended: Option A** — use Beam's native metrics, export to Prometheus via stackdriver-exporter. Simpler, no Pushgateway to manage.

```python
# In online_store_reader.py
from apache_beam.metrics import Metrics

class FetchFeaturesFromOnlineStore(beam.DoFn):
    def __init__(self, ...):
        ...
        self.fetch_latency = Metrics.distribution(self.__class__.__name__, "feature_fetch_latency_ms")
        self.fetch_success = Metrics.counter(self.__class__.__name__, "feature_fetch_success")
        self.fetch_failure = Metrics.counter(self.__class__.__name__, "feature_fetch_failure")
        self.fetch_retry = Metrics.counter(self.__class__.__name__, "feature_fetch_retry")

    async def _fetch_one(self, element, semaphore):
        entity_id = element["entity_id"]
        start = time.monotonic()
        async with semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._client.fetch_feature_values(...)
                    # ... existing logic ...
                    if len(features) == len(self.feature_columns):
                        elapsed_ms = (time.monotonic() - start) * 1000
                        self.fetch_latency.update(int(elapsed_ms))
                        self.fetch_success.inc()
                        element.update(features)
                        return element

                    if attempt < self.max_retries:
                        self.fetch_retry.inc()
                        # ... existing backoff ...
                    else:
                        self.fetch_failure.inc()
                        return None

                except Exception as e:
                    if attempt < self.max_retries:
                        self.fetch_retry.inc()
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

Currently you write the prediction to BQ but not the feature values the model actually saw. Add them:

```python
# In BatchCallFastAPIService._call_async(), the feature values are already in `batch`
# Include them in the BQ prediction row (you partially do this already for FEATURE_COLUMNS)
# Also add: trace_id, feature_fetch_latency_ms, retry_count
```

This lets you answer: "What did the model see when it predicted X for entity Y at time T?"

### Investigation Queries

```sql
-- Trace a single entity through the system
SELECT *
FROM `ml_dataset.iris_predictions_streaming`
WHERE entity_id = '42_streaming'
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

  # For Dataflow/Beam batch metrics
  pushgateway:
    image: prom/pushgateway:latest
    ports:
      - "9091:9091"
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
  - job_name: "fastapi-inference"
    static_configs:
      - targets: ["fastapi-service:8080"]
    metrics_path: "/metrics"
    scrape_interval: 10s

  - job_name: "stackdriver-exporter"
    static_configs:
      - targets: ["stackdriver-exporter:9255"]
    scrape_interval: 60s

  - job_name: "pushgateway"
    static_configs:
      - targets: ["pushgateway:9091"]
    honor_labels: true
```

---

## Implementation Order

### Phase 1: FastAPI Metrics (lowest effort, highest signal)
1. Add `prometheus-client` to `requirements.fastapi.txt`
2. Add middleware + `/metrics` endpoint to `fastapi_server.py`
3. Instrument: request latency, prediction count, batch size, error count
4. Deploy Prometheus + Grafana (docker-compose locally, Cloud Run / GKE for prod)
5. Build Dashboard 1 (Pipeline Health)

### Phase 2: Beam / Dataflow Metrics
1. Add `Metrics.counter()` and `Metrics.distribution()` to all DoFns
2. Deploy stackdriver-exporter to bridge Dataflow metrics → Prometheus
3. Add feature fetch latency, error counters, write latency to dashboards

### Phase 3: Error Handling + Dead Letters
1. Create `ml_dataset.dead_letters` BQ table
2. Add dead letter routing to `ParsePubSubMessage`, `FetchFeaturesFromOnlineStore`, `BatchCallFastAPIService`
3. Remove `ERROR` prediction rows pattern — use dead letters instead
4. Add Alertmanager rules for error rate thresholds
5. Build Dead Letters panel in Grafana

### Phase 4: Tracing + Investigation
1. Switch to `ReadFromPubSub(with_attributes=True)` to capture message_id as trace_id
2. Propagate trace_id through all pipeline stages
3. Store feature snapshot at prediction time in BQ
4. Build Investigation dashboard (entity lookup, feature correlation)
5. Create BQ views for common investigation queries

### Phase 5: Cost Attribution
1. Deploy stackdriver-exporter with Dataflow/Bigtable/BQ/Pub/Sub metric prefixes
2. Add `pipeline` label to all custom metrics
3. Build cost dashboard with per-pipeline attribution
4. Set up cost anomaly alerts (daily spend > 2× rolling average)

---

## Dependencies to Add

```
# requirements.fastapi.txt (add)
prometheus-client>=0.20.0

# pyproject.toml (add to beam/dataflow deps)
prometheus-client>=0.20.0
```

---

## File Structure

```
observability/
├── prometheus.yml                    # Scrape config
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
