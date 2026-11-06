# QPMIL-VL

This adapter runs the complete continual QPMIL-VL recipe from
`can-can-ya/QPMIL-VL`, commit `3a7a7698582dec866d43eb748f8c3599f7be4391`.
It uses external, private author code and an actual user-authorized CONCH
checkpoint. Neither the author's ND code nor model weights are redistributed.

The default `QPMIL-VL` alias selects
`qpmilvl_conch_max_query_patient_reference`. The other original query choice is
`qpmilvl_conch_mean_query_patient_reference`. Both use the full pretrained
110,129,664-parameter, 12-layer, 768-hidden, 12-head CONCH text tower, a pool of
20 keys and 24-token prompts, five matched prototypes, minibatch consensus,
frequency-based old-task penalties, and current-task-only tunable class vectors.
The full source CoCa constructor, including its frozen visual tower and unused
caption decoder, preserves source initialization order. Its declared timm
version is **1.0.11**; run this method in that isolated environment. Other methods
keep their own recorded dependencies.

Use CONCH's **512-dimensional contrastive projection** features. Record the
actual extraction normalization and exact checkpoint SHA256. Features from a
different, unaligned FM are not interchangeable inputs for this method. The
manifest binds ordered original patch IDs, feature bytes, clinical descriptions,
class templates, global patients, partitions, task order and CONCH checkpoint.
The concatenation of task classes must equal the dataset's global class order.
The first task needs at least two classes; every task needs TRAIN/VAL/TEST
patients for every class and at least 16 actual TRAIN slides.

With the task bundle prepared, the standard interface is:

```bash
milplus setup --method QPMIL-VL
milplus check --data /data/continual_task --method QPMIL-VL
milplus run --data /data/continual_task --method QPMIL-VL --device cuda:0 --output runs/qpmil
milplus evaluate --run runs/qpmil --device cuda:0
```

`prepare-qpmilvl --data DATA --schema SCHEMA.yaml --output MANIFEST.json` binds
an existing feature bundle. The schema contains `schema_version: 1`, ordered
`tasks` (each with `id` and `classes`), `slide_tasks`, `provenance`,
`class_descriptions` (class to nonempty description list), `templates` (one
`CLASSNAME` per template) and `conch` (`path`, `sha256`, and `feature_encoding`
with `model: CONCH`, `proj_contrast: true`, and actual boolean `normalize`).
Then register `qpmilvl: {manifest: MANIFEST.json}` under that encoder in task.yaml.

`demo-qpmilvl --output DATA --weights CHECKPOINT --sha256 SHA` creates explicitly
synthetic bags for the original four task/class names and all 22 original text
templates. It never represents actual TCGA slides or actual feature extraction.

Training retains CE + 0.5 matching loss + 0.5 class-similarity loss, norm clipping
at 1, original zero-gradient parameter exclusion before Adam (lr .001, eps 1e-8,
weight decay .0005), plateau LR and source early stopping. The default budget is
12 epochs per task. Selection follows the original `load_best_ckpt: false`:
**the final actual task epoch**, not validation-best weights. The combined
validation monitor is (loss + 1 - accuracy)/2. Every actual epoch preserves
learned parameters, stale gradients, frequency histories, optimizer moments,
scheduler, early stopping, all loader generators and Python/NumPy/Torch/CUDA RNG.

Original training makes an update after 16 bags and drops an incomplete tail.
This remains explicit in `dropped_native_incomplete_tail_slides`; inputs with
fewer than 16 TRAIN slides per task fail. No smaller replacement minibatch is
silently used. Frozen model weights are reloaded by their verified hash rather
than duplicated in every learned-state checkpoint.

The patient-safe port defers TEST decoding until explicit evaluation. Old-task
VAL diagnostics do not select models. No external telemetry or tSNE plotting is
started; the original debugging plots can consume NumPy randomness and require
more than 30 samples. Primary evaluation uses all-seen global classes; a
separate task-masked diagnostic is labelled explicitly. A matrix, forgetting and
backward-transfer summaries, and per-slide patient-bound predictions are saved.

Software, independent wheel, both native query budgets and whole-library
acceptance remain distinct evidence. Synthetic engineering checks do not claim
real WSI experiments or paper-score reproduction.

Install in a dedicated environment with `python -m pip install ".[qpmilvl]"` from the checkout, or `pip install "mil-baseline-puls[qpmilvl]"` for a released package. This exact timm1.0.11 extra is isolated from methods pinned to timm0.9.12.
