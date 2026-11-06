# AKD-PMP complete continual protocol

Official source: https://github.com/lxr2/akd-pmp,
commit `a1c83d2efe31bd713b0f7999715297700bfdbb80`.
Paper: https://arxiv.org/abs/2505.10649.
Original SmoothTop1SVM dependency: https://github.com/oval-group/smooth-topk,
commit `12c1645f187e2fa0c05f47bf1fe48864d4bd2707`.
Author code is fetched privately; no third-party source is bundled here.

## Complete method

The adapter loads both original full-width models: CLAM-SB with512 hidden units,
256 attention units, .25 dropout, gated attention, k8 instance selection,
subtyping, and actual smooth SVM instance loss; and the author's two-layer
TransMIL512 with eight heads,256 Nystrom landmarks, six inverse iterations,
original positional convolutions and the actual attention/intermediate outputs.
Feature dimensions come from the selected FM; the native TCGA and skin cases
use1024 and512 respectively. No fake spatial coordinates are created.

The entire author current-bag plus memory-replay training loop is loaded from
the pinned `one_fold` AST, as are validation, memory refill, native reservoir
classes and pseudo-bag selection. Single-device Fabric operations are performed
as literal full-float PyTorch moves/backward. Logging is observed separately.

CLAM retains .7 bag CE + .3 smooth SVM instance loss, attention KD (T2) and
10x logit KD (T2) on old pseudo-bags. TransMIL retains10x logit KD (T2),10x
two-layer attention KD (T1), and10x two intermediate CLS-feature MSE terms.
Original KL batchmean reduction is retained, without adding a temperature-square
factor. The default source recipe does not distill on current-task samples.
No clipping, extra scheduler or weight normalization is invented.

## Four explicit author recipes

| Profile suffix | Native reference settings |
|---|---|
| `clam_tcga_patient_reference` | Adam2e-4/wd1e-5,50 maximum epochs/task, balanced memory300, weighted replacement sampling,10% pseudo-bags |
| `transmil_tcga_patient_reference` | Same TCGA optimizer/budget/memory; actual TransMIL KD terms |
| `clam_skin_patient_reference` | Adam1e-4/wd0,40/50/50 maximum epochs, ordinary reservoir42, entire original bags |
| `transmil_skin_patient_reference` | Adam1e-5/1e-6/1e-6,wd0,20/40/25 maximum epochs, ordinary reservoir42, entire original bags |

All IDs start `akdpmp_`. The `AKD-PMP` alias resolves the complete TransMIL TCGA
reference. Defaults follow the README launch commands and YAMLs, not a claim to
exactly reproduce every paper table. In particular, skin CLAM's first-stage YAML
budget40 differs from the paper's prose50. Paper-vs-source conditions should be
reported explicitly. All original PMP selectors (`random`, `max`, `maxmin`,
`maxminrand`, `maxrand`) are available as explicit `config.distill_method`; the
README default is `maxrand`, while the paper's strongest CLAM table condition
uses `maxminrand`. The requested variant is recorded in the run recipe.

For TCGA `maxrand`, the original top-half plus random-half calculation, integer
rounding, and potentially duplicated indices are retained. Stored pseudo-bags
include original row positions, parent feature and patch-identity hashes. Skin
source settings do not compress their already smaller bags. Class-balance quota
adjustment, the source's reservoir behavior, epoch permutation and proportional
interleaved replay schedule run unchanged.

## Real task inputs

Use an ordinary frozen-feature bundle with patient-disjoint TRAIN/VAL/TEST. Each
bag has actual ordered unique `patch_ids` in HDF5 or a `patch_ids_path` column.
The preparation schema contains `schema_version: 1`, `tasks` (ordered
`{id, classes}` entries), `slide_tasks` (exact slide-to-task ID mapping), and an
explicit `provenance` string. Global class order is exactly the concatenation of
task classes. Singleton incremental tasks are supported; inventing extra epochs
for an incompletely specified five-task YAML is rejected.

```bash
milplus setup --method AKD-PMP
milplus prepare-akdpmp --data /data/actual_tasks --schema /data/sequence.yaml \
  --output /data/actual_tasks/akdpmp_manifest.json
```

Add `akdpmp: {manifest: akdpmp_manifest.json}` under the chosen encoder. All actual
input/identity/partition/FM bytes are bound. Model training decodes only the
current task's TRAIN/VAL and previously selected memory; TEST is decoded only by
explicit evaluation. Integrity hashing of future/TEST files does not provide
their features or labels to the model.

```bash
milplus run --data /data/actual_tasks --method AKD-PMP --output /runs/akdpmp --device cuda:0
milplus evaluate --run /runs/akdpmp --subset test --device cuda:0
```

## Model selection and exact recovery

Original VAL loss is accumulated in CPU float32 as in torchmetrics.MeanMetric.
Early stopping uses patience10 and earliest stopping after floor(budget/2),
including original tie behavior. The selected task model is reloaded before
memory refill and becomes the next frozen teacher. The original seed reset
before buffer population is retained. DataLoader generator1, weighted sampler
RNG, source memory permutations/counters, all memory features/indices, teacher
weights/gradients, student parameters/gradients, Adam moments, early-stop state,
model modes, selections and full history are checkpointed. Inference evaluates
every selected task model against all seen tasks without task-ID masking, then
reports the complete accuracy matrix, average final accuracy, BWT and forgetting.
Per-task AUROC is unavailable for singleton-class subsets instead of being zero.

Compatibility changes are explicit: `np.Inf` becomes `np.inf`; SmoothTop1SVM's
original closures are rebound after `_apply` so Fabric/model.to actually uses
the moved label buffer; feature I/O uses one process while keeping native
collation, generator and sampling; patients/VAL are audited and TEST is withheld.
Model/loss formulas, widths and update order are not replaced.

```bash
# Explicitly synthetic software checks, not WSI results.
milplus demo-akdpmp --output /tmp/akd_demo --preset tcga --patches 41
milplus run --data /tmp/akd_demo --method AKD-PMP --epochs 2 --output /tmp/akd_run
```

Full method admission additionally requires independently installed-wheel,
native complete-budget and whole-library validation. Synthetic agreement is not
paper-score or new TCGA/BRCA/COAD reproduction.

## CPU numerical runtime

Canonical `milplus run` CPU execution re-executes that one command before torch
imports with `MKL_CBWR=COMPATIBLE`, `MKL_DYNAMIC=FALSE`, `OMP_DYNAMIC=FALSE`
and the requested original thread count. Evaluation of that run uses its same
policy. It does not change the parent shell, another MIL method, CONCH feature
extraction, or CUDA execution. Model dimensions, float32 precision, objective
terms, update order, dataset and budgets remain unchanged.

Default MKL dispatch produced identical sampler/RNG state but different first
training-step rounding in independent processes; source CLI/API exact comparison
failed and its artifacts are retained. Explicit CNR is the supported CPU runtime
for exact independent-process recovery. Library API users launch their Python
process with the same environment before importing torch. Direct API execution
in an unpinned CPU process remains available as the author/default condition;
its run request records that condition and does not claim bitwise reproducibility.
The complete four-profile/default-seed wheel comparator and every original exact
assertion are retained. See Intel's CNR documentation:
https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-1/mkl-cbwr-get.html
