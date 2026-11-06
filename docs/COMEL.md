# CoMEL complete continual reference

Author: https://github.com/Hyun1A/CoMEL, fixed commit
`9fd667994eb57e3960e36970a9509a8217d84a22`.
Paper: https://arxiv.org/abs/2507.02395.
The pinned repository has no LICENSE file. It is fetched privately by `setup`,
not redistributed in this package. The adapter loads the original model and
training definitions directly from the verified Git objects.

This is a continual task protocol. It requires two or more explicitly ordered
binary tasks, each with an actual negative/positive class pair and independent
patient-grouped TRAIN, VAL and TEST partitions. Global targets are 0/1 for task0,
2/3 for task1, and so on. The sequence in the manifest is binding. A three-class
RCC problem cannot be renamed a binary task, and one ordinary BRCA classifier
does not exercise this protocol. Supply real tasks and their definitions.

## Input

Use the ordinary frozen-feature task bundle, with global classes listed in task
order. Every feature bag has real ordered unique `patch_ids` in its HDF5 file or
an explicit `patch_ids_path` in the feature index. Feature dimensions are read
from the actual FM. The original UNI case uses1024. Nothing synthesizes spatial
coordinates. Optional `instance_labels_path` contains aligned binary integer
patch ground truth for VAL/TEST only; TRAIN patch ground truth is rejected.

The schema passed to `prepare-comel` contains `schema_version: 1`, `tasks` (a
list of `{id, classes: [negative_name, positive_name]}`), `slide_tasks` (exact
slide-to-task ID mapping) and a nonempty source `provenance` string.

```bash
milplus setup --method comel
milplus prepare-comel --data /data/tasks --schema /data/sequence.yaml \
  --output /data/tasks/comel_manifest.json
```

Add `comel: {manifest: comel_manifest.json}` to the chosen encoder in task.yaml.
The manifest binds every actual feature, patch identity, optional coordinate and
evaluation annotation byte, original patient/label/partition metadata, FM and
dimension. Preparation audits the entire input; training only decodes the
current task's TRAIN features and seen tasks' VAL features. TEST is decoded only
by explicit evaluation. Future-task integrity hashes are not training samples.

```bash
milplus run --data /data/tasks --method CoMEL --output /runs/comel --device cuda:0
milplus evaluate --run /runs/comel --subset test --device cuda:0
```

## Two explicit variants

`comel_owlora_source_patient_reference` retains the source GDAT, BPPL, weighted
LoRA, per-task new classifier, optimizer ownership, orthogonality loss and
post-backward projected gradients. The released `save_initial_basis` computes
SVD and truncates the base weight but does not assign U/Vh/S to the new initial
LoRA basis. Consequently its up matrix remains zero and its down matrix random;
these matrices are nevertheless used in later gradient projections. This
behavior is retained under the source profile.

`comel_owlora_svd_basis_repaired` explicitly makes that missing assignment. It
preserves the same constructor RNG consumption, ranks, source energy rule and
base-weight truncation. It is a disclosed repair, not a claim that original
paper scores have been reproduced.

Both preserve512 hidden units, two GDAT blocks/eight heads, actual region16
padding/grouping, .25 embedding and .1 transformer dropout, tanh pooling, no
qkv bias, and effective rank16 (48 for qkv). The released constructor ignores
the launcher `--rank8`; this adapter records the actual rank16. All model widths
remain intact when explicit small software epoch budgets are used.

## Full training, recovery and evaluation

The default is100 epochs **per task**, Adam1e-4/weight decay1e-5, original cosine
schedule, batch1, no gradient clipping, no AMP and no patch shuffle. Task0 trains
all parameters; subsequent optimizers contain only the latest LoRA parameters
and classifiers. Non-optimized base/previous-LoRA gradients remain accumulated
exactly as in the source and are included in checkpoints. BPPL uses the actual
positive/correct/.95-confident gate and adaptive threshold. TRAIN patch ground
truth never contributes to its loss.

The original loop saves final fixed-epoch task models rather than honoring its
early-stop flag. We preserve that selection and snapshot before each after-task
transition, then separately checkpoint the post-transition graph. Checkpoints
include every head/LoRA rank/counter/freeze flag, all active and inactive gradients,
model modes, Adam moments, cosine state, tau2 type/value, sampler RNG, training
order, complete history and validation matrix. Epoch-boundary interruption and
task-boundary interruption are both real paths; recovery never reruns SVD on
learned weights or recreates initial task states.

Patient-disjoint VAL replaces the source default that can alias TEST to VAL.
The manifest fixes the actual task count instead of copying a five-task flag
onto a four-organ list. Global-head accuracy is reported without a task mask;
per-task AUROC uses the global positive softmax column. This avoids source
evaluation failures when some global classes are absent in a per-task subset.
Optional localization uses real held-out patch annotations, the literal native
attention score `(a-min(a))/(max(a)+1e-10)`, and a VAL-only Youden threshold fixed
at each task completion. Per-slide foreground IoU/Dice, mean IoU and patch AUROC
are explicitly named. Missing annotations are marked unavailable. TEST thresholds
are never fitted. The final report includes the full lower-triangular task
performance matrix, average final accuracy, backward transfer and forgetting.

```bash
# Synthetic engineering checks only; not new WSI or paper experiments.
milplus demo-comel --output /tmp/comel_demo --tasks 3 --patches 17 --dimension 1024
milplus run --data /tmp/comel_demo --method CoMEL --epochs 2 --output /tmp/comel_run
```

Software checks, independent installed-wheel checks, full native100-epoch
conditions and whole-library admission are separate evidence. A runnable source
adapter or successful synthetic experiment alone does not establish paper-score
reproduction or finish the164-method goal.
