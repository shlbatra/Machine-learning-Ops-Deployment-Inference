# Plan: GPU Support for Vertex AI Pipelines (Training & Inference)

## Context

The current KFP pipeline components (`models.py`, `inference.py`, `evaluation.py`) run on Vertex AI's default CPU-only `e2-standard-4` machine type. This plan adds the ability to run specific pipeline steps on GPU-accelerated machines, configured at the KFP component level so Vertex AI provisions the right hardware automatically.

### Where GPU resources are configured

GPU resources belong on the **Vertex AI pipeline components** (the `@component` steps), not on the Composer KPO pod. The KPO pod only compiles and submits the pipeline — Vertex AI provisions the actual compute for each step.

```
Composer KPO pod (CPU-only, lightweight)
  → submits pipeline to Vertex AI
    → Vertex AI provisions per-step compute:
        load_data          → CPU (default)
        decision_tree      → GPU (if configured)
        random_forest      → GPU (if configured)
        choose_best_model  → CPU (default)
        inference_model    → GPU (if configured)
```

### KFP resource configuration API

KFP v2 provides these methods on any pipeline task:

```python
# Machine type
task.set_machine_type("n1-standard-8")

# GPU accelerator
task.set_accelerator_type("NVIDIA_TESLA_T4")
task.set_accelerator_count(1)

# Combined example in pipeline definition
dt_op = decision_tree(
    train_dataset=data_op.outputs["train_dataset"]
).set_display_name("Decision Tree") \
 .set_machine_type("n1-standard-8") \
 .set_accelerator_type("NVIDIA_TESLA_T4") \
 .set_accelerator_count(1)
```

These are set in the **pipeline function** (where tasks are wired together), not inside the `@component` function itself.

---

## 1. Approach: Pipeline-level resource configuration

Add machine type and GPU settings to the pipeline definitions in `iris_pipeline_training.py` and `iris_pipeline_inference.py`. Pass resource config as pipeline parameters so they can be overridden per-run without code changes.

### Why pipeline-level, not component-level

- Components stay reusable — same `decision_tree()` component can run on CPU or GPU depending on the pipeline
- Resource config is a deployment concern, not a logic concern
- Easy to override via Airflow DAG params or CLI args

---

## 2. Training pipeline changes

### `iris_pipeline_training.py`

Add resource configuration to the compute-heavy steps (model training). Lightweight steps (data loading, schema, evaluation) stay on default CPU.

```python
@kfp.dsl.pipeline(name=f"{PIPELINE_NAME}-training", pipeline_root=PIPELINE_ROOT)
def pipeline(
    project_id: str,
    location: str,
    bq_dataset: str,
    bq_feature_table: str,
    machine_type: str = "e2-standard-4",
    accelerator_type: str = "",
    accelerator_count: int = 0,
):
    # ... existing component imports ...

    # Data loading — CPU is fine
    data_op = load_data_from_feature_store(
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_feature_table=bq_feature_table,
    ).set_display_name("Load data from Feature Store")

    schema_load = load_schema(repo_root=REPO_ROOT).set_display_name(
        "Load schema relevant to model"
    )

    # Model training — GPU-capable
    dt_op = decision_tree(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Decision Tree") \
     .set_machine_type(machine_type)

    rf_op = random_forest(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Random Forest") \
     .set_machine_type(machine_type)

    # Conditionally add GPU if specified
    if accelerator_type and accelerator_count > 0:
        dt_op.set_accelerator_type(accelerator_type)
        dt_op.set_accelerator_count(accelerator_count)
        rf_op.set_accelerator_type(accelerator_type)
        rf_op.set_accelerator_count(accelerator_count)

    # Evaluation and registration — CPU is fine
    # ... rest stays the same ...
```

**Note**: The `if accelerator_type` conditional won't work inside a KFP pipeline function because pipeline params are `PipelineChannel` objects at compile time, not Python strings. Two options:

#### Option A: Separate pipeline functions (recommended)

Define `pipeline_cpu()` and `pipeline_gpu()`, or pass resource config as compile-time constants rather than pipeline parameters:

```python
# In __main__ block, before compilation:
MACHINE_TYPE = coalesce(cli.machine_type, "e2-standard-4")
ACCELERATOR_TYPE = coalesce(cli.accelerator_type, "")
ACCELERATOR_COUNT = int(coalesce(cli.accelerator_count, "0"))

@kfp.dsl.pipeline(name=f"{PIPELINE_NAME}-training", pipeline_root=PIPELINE_ROOT)
def pipeline(project_id: str, location: str, bq_dataset: str, bq_feature_table: str):
    # ... same as today ...

    dt_op = decision_tree(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Decision Tree") \
     .set_machine_type(MACHINE_TYPE)

    rf_op = random_forest(
        train_dataset=data_op.outputs["train_dataset"]
    ).set_display_name("Random Forest") \
     .set_machine_type(MACHINE_TYPE)

    if ACCELERATOR_TYPE:
        dt_op.set_accelerator_type(ACCELERATOR_TYPE)
        dt_op.set_accelerator_count(ACCELERATOR_COUNT)
        rf_op.set_accelerator_type(ACCELERATOR_TYPE)
        rf_op.set_accelerator_count(ACCELERATOR_COUNT)

    # ... rest unchanged ...
```

This works because `MACHINE_TYPE` etc. are resolved as Python variables before `@kfp.dsl.pipeline` compiles. The conditional is plain Python, not a KFP pipeline conditional.

---

## 3. Inference pipeline changes

### `iris_pipeline_inference.py`

Same pattern — add machine type and optional GPU to the `inference_model` step:

```python
MACHINE_TYPE = coalesce(cli.machine_type, "e2-standard-4")
ACCELERATOR_TYPE = coalesce(cli.accelerator_type, "")
ACCELERATOR_COUNT = int(coalesce(cli.accelerator_count, "0"))

@kfp.dsl.pipeline(name=f"{PIPELINE_NAME}-inference", pipeline_root=PIPELINE_ROOT)
def pipeline(
    project_id: str,
    location: str,
    bq_dataset: str,
    bq_feature_table: str,
    bq_table_predictions: str,
):
    get_model_op = get_model(
        project_id=project_id, location=location, model_name=MODEL_NAME
    ).set_display_name("Get Model")

    inference_op = (
        inference_model(
            project_id=project_id,
            location=location,
            model=get_model_op.outputs["latest_model"],
            bq_dataset=bq_dataset,
            bq_feature_table=bq_feature_table,
            bq_table_predictions=bq_table_predictions,
        )
        .set_display_name("Inference Model")
        .set_machine_type(MACHINE_TYPE)
        .after(get_model_op)
    )

    if ACCELERATOR_TYPE:
        inference_op.set_accelerator_type(ACCELERATOR_TYPE)
        inference_op.set_accelerator_count(ACCELERATOR_COUNT)
```

---

## 4. CLI argument changes

Add three new CLI args to both pipeline scripts:

```python
parser.add_argument("--machine-type", default="e2-standard-4",
                    help="Vertex AI machine type (e.g., n1-standard-8, e2-standard-4)")
parser.add_argument("--accelerator-type", default="",
                    help="GPU type (e.g., NVIDIA_TESLA_T4, NVIDIA_TESLA_V100, NVIDIA_L4)")
parser.add_argument("--accelerator-count", default="0",
                    help="Number of GPUs per step")
```

### Usage examples

```bash
# CPU-only (default, same as today)
python -m ml_pipelines_kfp.iris_xgboost.pipelines.iris_pipeline_training

# GPU training with T4
python -m ml_pipelines_kfp.iris_xgboost.pipelines.iris_pipeline_training \
    --machine-type n1-standard-8 \
    --accelerator-type NVIDIA_TESLA_T4 \
    --accelerator-count 1

# GPU inference with L4
python -m ml_pipelines_kfp.iris_xgboost.pipelines.iris_pipeline_inference \
    --machine-type g2-standard-8 \
    --accelerator-type NVIDIA_L4 \
    --accelerator-count 1
```

---

## 5. DAG changes

Add GPU params to the Airflow DAGs so resource config can be overridden per-run from the Airflow UI:

```python
# In both DAGs, add to params:
params={
    # ... existing params ...
    "machine_type": Param("e2-standard-4", type="string",
                          description="Vertex AI machine type"),
    "accelerator_type": Param("", type="string",
                              description="GPU type (blank for CPU-only)"),
    "accelerator_count": Param("0", type="string",
                               description="Number of GPUs"),
},

# In KPO arguments, add:
arguments=[
    # ... existing args ...
    "--machine-type", "{{ params.machine_type }}",
    "--accelerator-type", "{{ params.accelerator_type }}",
    "--accelerator-count", "{{ params.accelerator_count }}",
],
```

Scheduled runs use CPU defaults. Trigger manually with GPU params when needed.

---

## 6. GPU machine type + accelerator combinations

| Use case | Machine type | Accelerator | Cost/hr (approx) |
|---|---|---|---|
| CPU-only (default) | `e2-standard-4` | none | ~$0.13 |
| Light GPU (small models) | `n1-standard-4` | `NVIDIA_TESLA_T4` x1 | ~$0.55 |
| Medium GPU (training) | `n1-standard-8` | `NVIDIA_TESLA_T4` x1 | ~$0.70 |
| Heavy GPU (large models) | `n1-standard-16` | `NVIDIA_TESLA_V100` x1 | ~$2.90 |
| Inference-optimized | `g2-standard-8` | `NVIDIA_L4` x1 | ~$1.00 |

**Note**: GPU machine types must use `n1-*` (for T4/V100/A100) or `g2-*` (for L4). The default `e2-*` does not support GPUs.

---

## 7. Docker image considerations

The current `ml-pipelines-kfp-image` uses scikit-learn and XGBoost (CPU). For GPU training:

- **XGBoost GPU**: XGBoost supports `tree_method="gpu_hist"` out of the box — just install `xgboost` with CUDA support in the Docker image
- **PyTorch / TensorFlow**: Would need a GPU-enabled base image (e.g., `nvidia/cuda:12.x-runtime-ubuntu22.04` or `tensorflow/tensorflow:latest-gpu`)
- **Current Iris models (DecisionTree, RandomForest)**: scikit-learn does not support GPU — these would not benefit from GPU hardware. GPU makes sense when moving to XGBoost `gpu_hist` or deep learning

---

## 8. Implementation order

1. Add `--machine-type`, `--accelerator-type`, `--accelerator-count` CLI args to both pipeline scripts
2. Add `.set_machine_type()` calls to compute-heavy steps in both pipeline definitions
3. Add conditional `.set_accelerator_type()` / `.set_accelerator_count()` when GPU is specified
4. Add GPU params to both Airflow DAGs
5. Test with CPU defaults (no behavior change)
6. Test with `--machine-type n1-standard-4 --accelerator-type NVIDIA_TESLA_T4 --accelerator-count 1`
7. Update Docker image with CUDA support when switching to GPU-native models

---

## 9. File changes

| File | Change |
|---|---|
| `src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_training.py` | Add CLI args, set machine type + GPU on training steps |
| `src/ml_pipelines_kfp/iris_xgboost/pipelines/iris_pipeline_inference.py` | Add CLI args, set machine type + GPU on inference step |
| `dags/iris_training_dag.py` | Add `machine_type`, `accelerator_type`, `accelerator_count` params |
| `dags/iris_batch_inference_dag.py` | Add `machine_type`, `accelerator_type`, `accelerator_count` params |

### Unchanged

- All `@component` functions (`models.py`, `inference.py`, etc.) — resource config is set on the task in the pipeline, not inside the component
- Docker images — no change needed until switching to GPU-native model code
- `setup_composer.sh` — KPO pod stays CPU-only
