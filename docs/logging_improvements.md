# Logging Improvements Plan

## Problem

The project uses `print()` for ~90% of its logging across KFP pipeline components. There are no log levels, no timestamps, and one security issue (`print(credentials)` in deploy.py). The non-component files (server.py, pubsub_producer.py, dataflow) each use a different logging setup with no shared config. `google-cloud-logging` is already in `pyproject.toml` but unused.

## Current State

| Layer | Approach | Files |
|---|---|---|
| KFP pipeline components | Raw `print()` | deploy.py, evaluation.py, inference.py, get_model.py, register.py |
| KFP pubsub consumer | `logging.getLogger(__name__)` | pubsub_bq_consumer.py |
| FastAPI server | `getLogger()` (root, no formatter) | server.py |
| Dataflow pipeline | `logging.info()` functional API | iris_streaming_pipeline.py |
| Pub/Sub producer | `logging.getLogger(__name__)` + `basicConfig` | pubsub_producer.py |
| BQ dataloader | Raw `print()` | bq_dataloader.py |

## Approach

### 1. Create a shared logging helper

**New file:** `src/ml_pipelines_kfp/log.py`

```python
import json
import logging
import sys


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        # Merge any extra_data passed via logger.info("msg", extra={"extra_data": {...}})
        extra_data = getattr(record, "extra_data", None)
        if extra_data and isinstance(extra_data, dict):
            log_entry.update(extra_data)
        return json.dumps(log_entry)


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

Usage with extra data:
```python
logger = get_logger(__name__)

# Basic
logger.info("Model loaded")
# → {"severity": "INFO", "message": "Model loaded", "module": "server"}

# With extra context
logger.info("Prediction complete", extra={"extra_data": {"batch_size": 50, "latency_ms": 12.3}})
# → {"severity": "INFO", "message": "Prediction complete", "module": "server", "batch_size": 50, "latency_ms": 12.3}
```

Use `get_logger(__name__)` in all non-component files (server.py, pubsub_producer.py, bq_dataloader.py, dataflow). Located at the package root so it's importable by any subpackage (`iris_xgboost`, `dataflow`, future model packages).

### 2. Switch KFP components from `python:3.10` to project Docker image

All KFP components except `deploy.py` and `schema.py` use `base_image="python:3.10"` with inline `packages_to_install`. This means they can't import project modules (like `log.py`) and must reinstall dependencies on every run.

Switch them to `base_image=IMAGE_NAME` (the project Docker image built by CI/CD), which already has all dependencies and the project package installed. This enables importing `get_logger()` and any other project code inside component functions.

**Files to update (7 components):**

| File | Current `base_image` |
|---|---|
| `components/data.py` | `python:3.10` |
| `components/models.py` (2 components) | `python:3.10` |
| `components/evaluation.py` | `python:3.10` |
| `components/register.py` | `python:3.10` |
| `components/inference.py` | `python:3.10` |
| `components/get_model.py` | `python:3.10` |
| `components/pubsub_bq_consumer.py` | `python:3.10-slim` |

**For each file:**
- Add `from ml_pipelines_kfp.iris_xgboost.constants import IMAGE_NAME`
- Replace `@component(base_image="python:3.10", packages_to_install=[...])` with `@component(base_image=IMAGE_NAME)`
- Remove the `packages_to_install` list (no longer needed)

**Already using `IMAGE_NAME`** (no changes needed): `deploy.py`, `schema.py`

### 3. KFP components: use `get_logger()` from `log.py`

With the project image as base, components can now import the shared logger directly:

```python
from ml_pipelines_kfp.log import get_logger

logger = get_logger(__name__)
logger.info("Model loaded")
```

Cloud Logging auto-parses the `severity` field from JSON output, enabling log-level filtering in GCP Console.

Then:
- **Remove** all debug noise prints (`df.head()`, `df.columns`, `type()`, raw object dumps)
- **Convert** progress/info prints to `logger.info(...)`
- **Convert** error prints to `logger.error(...)`
- **Delete** `print(credentials)` in deploy.py (security fix)

### 4. Fix FastAPI server logging (server.py)

```python
from ml_pipelines_kfp.log import get_logger

log = get_logger(__name__)
```

Remove the manual `StreamHandler` setup and root logger usage. Keep `LOG_LEVEL` env var support by adding:
```python
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
```

### 5. Standardize non-component files

**bq_dataloader.py** — replace `print()` with `get_logger(__name__)` from `log.py`.

**iris_streaming_pipeline.py** — replace functional `logging.info()`/`logging.error()` with a named logger from `log.py`.

**pubsub_producer.py** — replace `basicConfig()` with `get_logger()` from `log.py` for consistent JSON formatting.

### 6. Leave test files as-is

- Test files use `print()` for human-readable output. Standard practice.
- `pubsub_bq_consumer.py` already uses proper `logging.getLogger(__name__)` pattern.

## File-by-file changes

| File | Action |
|---|---|
| `ml_pipelines_kfp/log.py` | **New** — shared JSON formatter + `get_logger()` |
| `components/data.py` | Switch `base_image` to `IMAGE_NAME`, remove `packages_to_install` |
| `components/models.py` | Switch `base_image` to `IMAGE_NAME` for both `decision_tree` and `random_forest`, remove `packages_to_install` |
| `components/evaluation.py` | Switch `base_image` to `IMAGE_NAME`, remove `packages_to_install`. Remove 2 debug prints. Convert model-selection prints to `logger.info()` |
| `components/register.py` | Switch `base_image` to `IMAGE_NAME`, remove `packages_to_install`. Convert 1 print to `logger.info()` |
| `components/inference.py` | Switch `base_image` to `IMAGE_NAME`, remove `packages_to_install`. Remove 7 debug prints. Convert to `logger.info()` |
| `components/get_model.py` | Switch `base_image` to `IMAGE_NAME`, remove `packages_to_install`. Remove 7 debug prints. Fix broken f-string on line 38: `f"Could not find model with f{model_name}"` to `f"Could not find model: {model_name}"` |
| `components/deploy.py` | Already uses `IMAGE_NAME`. **Security:** remove `print(credentials)` on line 44. Remove debug prints. Convert ~18 info/error prints to `logger.info()`/`logger.error()` |
| `components/pubsub_bq_consumer.py` | Switch `base_image` to `IMAGE_NAME`, remove `packages_to_install` |
| `iris_xgboost/bq_dataloader.py` | Replace 4 prints with logger from `log.py` |
| `dataflow/iris_streaming_pipeline.py` | Use named logger from `log.py` instead of functional `logging.info()` |
| `iris_xgboost/pubsub_producer.py` | Use `get_logger()` from `log.py` instead of `basicConfig()` |

**No changes needed:** `schema.py` (already uses `IMAGE_NAME`), test files.

## Verification

```bash
# 1. Syntax check all modified files
python -m py_compile src/ml_pipelines_kfp/log.py
python -m py_compile src/ml_pipelines_kfp/iris_xgboost/server.py
# ... etc for each modified file

# 2. Run existing tests
python -m pytest test/

# 3. Verify security fix
grep -rn "print(credentials)" src/
# Should return nothing

# 4. Verify debug prints removed
grep -rn "print(df\.\|print(type(" src/ml_pipelines_kfp/iris_xgboost/pipelines/components/
# Should return nothing

# 5. Spot-check JSON output
python -c "from ml_pipelines_kfp.log import get_logger; get_logger('test').info('hello')"
# Should output: {"severity": "INFO", "message": "hello", "module": "log"}
```
