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

**New file:** `src/ml_pipelines_kfp/iris_xgboost/log.py`

```python
import json
import logging
import sys


class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "severity": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        })


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

Use `get_logger(__name__)` in all non-component files (server.py, pubsub_producer.py, bq_dataloader.py, dataflow).

### 2. KFP components: structured JSON via print

KFP lightweight components using `base_image="python:3.10"` can't import project modules. Define a local helper inside each component function body:

```python
def _log(severity, message):
    import json
    print(json.dumps({"severity": severity, "message": message}))
```

Cloud Logging auto-parses the `severity` field, enabling log-level filtering in GCP Console.

Then:
- **Remove** all debug noise prints (`df.head()`, `df.columns`, `type()`, raw object dumps)
- **Convert** progress/info prints to `_log("INFO", ...)`
- **Convert** error prints to `_log("ERROR", ...)`
- **Delete** `print(credentials)` in deploy.py (security fix)

### 3. Fix FastAPI server logging (server.py)

```python
from ml_pipelines_kfp.iris_xgboost.log import get_logger

log = get_logger(__name__)
```

Remove the manual `StreamHandler` setup and root logger usage. Keep `LOG_LEVEL` env var support by adding:
```python
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
```

### 4. Standardize non-component files

**bq_dataloader.py** — replace `print()` with `get_logger(__name__)` from `log.py`.

**iris_streaming_pipeline.py** — replace functional `logging.info()`/`logging.error()` with a named logger from `log.py`.

**pubsub_producer.py** — replace `basicConfig()` with `get_logger()` from `log.py` for consistent JSON formatting.

### 5. Leave test files and pubsub_bq_consumer.py as-is

- Test files use `print()` for human-readable output. Standard practice.
- `pubsub_bq_consumer.py` already uses proper `logging.getLogger(__name__)` pattern.

## File-by-file changes

| File | Action |
|---|---|
| `iris_xgboost/log.py` | **New** — shared JSON formatter + `get_logger()` |
| `components/deploy.py` | **Security:** remove `print(credentials)` on line 44. Remove debug prints. Convert ~18 info/error prints to `_log()` |
| `components/inference.py` | Remove 7 debug prints (`df.head()`, `df.columns`, `len()` dumps). Convert 1 info print to `_log("INFO", ...)` |
| `components/get_model.py` | Remove 7 debug prints. Fix broken f-string on line 38: `f"Could not find model with f{model_name}"` → `f"Could not find model: {model_name}"`. Convert to `_log("ERROR", ...)` |
| `components/evaluation.py` | Remove 2 debug prints (`print(dt_accuracy)`, `print(rf_accuracy)`). Convert 2 model-selection prints to `_log("INFO", ...)` |
| `components/register.py` | Convert 1 print to `_log("INFO", "Model uploaded successfully")` (don't dump the full result object) |
| `iris_xgboost/server.py` | Use `get_logger(__name__)` from `log.py`, remove manual StreamHandler |
| `iris_xgboost/bq_dataloader.py` | Replace 4 prints with logger from `log.py` |
| `dataflow/iris_streaming_pipeline.py` | Use named logger from `log.py` instead of functional `logging.info()` |
| `iris_xgboost/pubsub_producer.py` | Use `get_logger()` from `log.py` instead of `basicConfig()` |

**No changes needed:** `data.py`, `models.py`, `schema.py`, `pubsub_bq_consumer.py`, test files.

## Verification

```bash
# 1. Syntax check all modified files
python -m py_compile src/ml_pipelines_kfp/iris_xgboost/log.py
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
python -c "from ml_pipelines_kfp.iris_xgboost.log import get_logger; get_logger('test').info('hello')"
# Should output: {"severity": "INFO", "message": "hello", "module": "log"}
```
