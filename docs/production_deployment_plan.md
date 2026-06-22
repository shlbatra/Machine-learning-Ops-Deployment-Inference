# Plan: Production Deployment

## Context

Everything has been built and tested in staging:
- Cloud Composer 2 environment is running with 4 DAGs (training + inference, staging + prod)
- Workload Identity and RBAC are configured
- CI/CD builds images and syncs DAGs on every push
- Pipeline scripts use `google.auth.default()` (no key files)
- Staging DAGs triggered manually, validated end-to-end

In production, everything runs on automated schedules with no manual intervention.

---

## What's Already in Place

| Component | Status | Notes |
|-----------|--------|-------|
| Composer 2 environment | Running | `ml-pipelines-composer`, shared across staging/prod |
| Prod training DAG | Deployed | `iris_training_prod`, schedule `0 6 * * *` (daily 6am UTC) |
| Prod inference DAG | Deployed | `iris_batch_inference_prod`, schedule `0 8 * * *` (daily 8am UTC) |
| Docker images (`main` tag) | Built by CI | Pushed on every merge to `main` |
| DAG sync | Automated | CI pushes `dags/*.py` to Composer GCS bucket on every push |
| Workload Identity | Configured | KPO pods auth as `kfp-mlops@` SA |
| RBAC | Configured | Airflow SA can manage pods + events in `composer-user-workloads` |
| Feature Store | Running | Offline (BQ) + online (Bigtable) serving |
| Cloud Run FastAPI | Deployed | `iris-classifier-xgboost-service` (prod) |

---

## Steps to Go Live

### Step 1: Verify `main` Images Exist in Artifact Registry

```bash
# Confirm all 3 images have the main tag after CI completes
gcloud artifacts docker images list \
  us-docker.pkg.dev/deeplearning-sahil/sahil-experiment-docker-images \
  --filter="tags:main" \
  --format="table(package, tags, createTime)"
```

Expected: `ml-pipelines-kfp-image:main`, `fastapi-ml-generic:main`, `dataflow-beam:main`

### Step 2: Verify Prod DAGs Are Active in Airflow

Open the Airflow UI:

```bash
gcloud composer environments describe ml-pipelines-composer \
  --location us-central1 \
  --format='value(config.airflowUri)'
```

Check:
- `iris_training_prod` — schedule `0 6 * * *`, unpaused
- `iris_batch_inference_prod` — schedule `0 8 * * *`, unpaused

Both DAGs are currently **paused by default** (Airflow pauses new DAGs). Unpause them only after Step 3.

### Step 3: Manual Prod Validation Run

Before enabling schedules, trigger each prod DAG once manually to validate:

```bash
# Training
gcloud composer environments run ml-pipelines-composer \
  --location us-central1 \
  trigger_dag -- iris_training_prod

# Wait for training to complete (~15-20 min), then inference
gcloud composer environments run ml-pipelines-composer \
  --location us-central1 \
  trigger_dag -- iris_batch_inference_prod
```

Verify:
- [ ] Training DAG completes successfully in Airflow UI
- [ ] Vertex AI pipeline `pipeline-iris-prod` shows SUCCESS
- [ ] Model `Iris-Classifier-XGBoost` has a new version with `blessed` alias in Vertex AI Model Registry
- [ ] Cloud Run service `iris-classifier-xgboost-service` is updated and healthy
- [ ] Inference DAG completes successfully
- [ ] `ml_dataset.iris_predictions` table has new rows with current `prediction_timestamp`

### Step 4: Unpause Prod DAGs

Once manual runs succeed, enable the automated schedules in the Airflow UI by toggling the pause switch for `iris_training_prod` and `iris_batch_inference_prod`.

### Step 5: Deploy Prod Dataflow Streaming Pipelines

Deploy the two long-running Dataflow jobs for prod:

**Feature ingestion pipeline** (Pub/Sub → BQ + Bigtable):
- Go to GitHub Actions → "Deploy Dataflow Feature Pipeline" → Run workflow
- Select: environment=`prod`, region=`us-central1`, machine_type=`e2-standard-2`

**Streaming inference pipeline** (Pub/Sub → online store → FastAPI → BQ):
- Go to GitHub Actions → "Deploy Dataflow Streaming" → Run workflow
- Select: environment=`prod`, region=`us-central1`, machine_type=`e2-standard-2`

Verify:
- [ ] Both jobs show as `Running` in Dataflow console
- [ ] Publish test events: `./scripts/run_pubsub_producer.sh 5 2 30`
- [ ] Check `ml_dataset.iris_features` has new rows (feature pipeline)
- [ ] Check `ml_dataset.iris_predictions_streaming` has new rows (inference pipeline)

### Step 6: Pause Staging DAGs (Optional)

If staging DAGs aren't needed for daily runs, pause them to avoid accidental triggers:

```bash
gcloud composer environments run ml-pipelines-composer \
  --location us-central1 \
  dags pause -- iris_training_staging

gcloud composer environments run ml-pipelines-composer \
  --location us-central1 \
  dags pause -- iris_batch_inference_staging
```

They remain available for manual testing anytime — just trigger from the UI.

---

## Production Schedule Summary

| What | When | How | Writes to |
|------|------|-----|-----------|
| Training pipeline | Daily 6am UTC | Composer DAG → KPO → Vertex AI | Model Registry, Cloud Run |
| Batch inference | Daily 8am UTC | Composer DAG → KPO → Vertex AI | `ml_dataset.iris_predictions` |
| Feature ingestion | Continuous | Dataflow (streaming) | `iris_features` (BQ) + Bigtable |
| Streaming inference | Continuous | Dataflow (streaming) | `iris_predictions_streaming` (BQ) |
| Image builds | On merge to `main` | GitHub Actions CI | Artifact Registry |
| DAG sync | On every push | GitHub Actions CI | Composer GCS bucket |

---

## Monitoring Checklist (Daily)

### Airflow UI
- Check DAG run history — both prod DAGs should show green (success) for today
- Check task duration — training ~15-20 min, inference ~5-10 min
- Review any failed runs and retry if transient

### Vertex AI Console
- Pipeline runs: `pipeline-iris-prod` should have a daily run
- Model Registry: `Iris-Classifier-XGBoost` should have `blessed` alias on latest version

### BigQuery
```sql
-- Latest batch predictions
SELECT prediction_timestamp, COUNT(*) as row_count
FROM `deeplearning-sahil.ml_dataset.iris_predictions`
GROUP BY prediction_timestamp
ORDER BY prediction_timestamp DESC
LIMIT 5;

-- Latest streaming predictions
SELECT prediction_timestamp, COUNT(*) as row_count
FROM `deeplearning-sahil.ml_dataset.iris_predictions_streaming`
WHERE prediction_timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
GROUP BY prediction_timestamp
ORDER BY prediction_timestamp DESC
LIMIT 5;
```

### Dataflow Console
- Both streaming jobs should show status `Running`
- Check watermark lag — should be < 1 minute under normal load
- Check error rate in job metrics

### Cloud Run
- `iris-classifier-xgboost-service` should show healthy instances
- Check request latency and error rate in Cloud Run metrics

---

## Rollback

### Bad model deployed
If training produces a bad model:
1. In Vertex AI Model Registry, move the `blessed` alias back to the previous version
2. Re-run the deploy step manually or trigger the training DAG with known-good data

### DAG broken after merge
1. Revert the merge commit on `main`
2. CI will rebuild images and re-sync DAGs automatically
3. Or manually upload the last working DAG: `gcloud storage cp <fixed_dag.py> <DAGS_BUCKET>/`

### Dataflow job crashed
Re-deploy via GitHub Actions workflow dispatch — same as Step 6.

### Composer environment unhealthy
```bash
gcloud composer environments describe ml-pipelines-composer \
  --location us-central1 \
  --format="value(state)"
```
Should return `RUNNING`. If in error state, check Cloud Logging for Composer environment logs.
