# MIL-Adapter

This adapter executes the full original author models from
`cvblab/MIL-Adapter`, commit `ade11bb19c3f09a35da9e8255a87a079217aef48`.
There is no LICENSE file at that commit. The code is fetched into a private
source cache and is not redistributed inside this package.

## Actual configurations

The author implements 26 conditions, all registered separately:

| Adaptation | Aggregators | Initialization |
| --- | --- | --- |
| ZSMIL | BGAP, BGMP, ABMIL, TransMIL, WIKGMIL, ILRAMIL, RRTMIL | ZS and random |
| TaskRes | BGAP, BGMP, ABMIL, TransMIL | Actual text prototype residual |
| CLIPAdapter | BGAP, BGMP, ABMIL, TransMIL | Frozen text, trainable residual MLP |
| TIPAdapter | BGAP, BGMP, ABMIL, TransMIL | Frozen text/onehot, trainable support keys |

The `MIL-Adapter` alias selects `miladapter_taskres_abmil`. An explicit execution
ID selects another condition. The global choices in the original argparse do
not make the missing TaskRes/CLIP/TIP graph/low-rank/RRT combinations implemented.
The `LR` choice is not constructed by the author. TITAN prototypes are absent.
Neither is represented as a working profile here.

All aggregators use the complete author's native512D CONCH/768D KEEP implementations. In
particular, RRT retains its regional and cross-region implementation, ILRA the
64 latent vectors and low-rank attention, WiKG topk6 neighborhood interactions,
TransMIL both Nyström layers and PPEG, and ABMIL its original384D gated attention.
They are not replaced with the namesake generic MIL classes in this library.

TaskRes starts its trainable residual from the text matrix, then computes
`text + .25 * residual`, as the author does. CLIPAdapter is the complete
nativeD→D/4→D bias-free/ReLU MLP mixed `.2 * adapted + .8 * embedding`.
TIP initializes every support key with the NumPy mean of its **original bag**,
even when the inference aggregator is TransMIL or ABMIL. The keys are trained;
onehot support labels and text are frozen. Its beta5/alpha1 exponential cache
and temperature56.347694396972656 are unchanged.

## Source repairs and protocol extension

The public ZSMIL ZS branch references an undefined global `logit_scale`; this
port uses its existing `self.logit_scale`. The public CLIP/TIP ABMIL branches
treat the returned `(embedding, attention)` tuple as a tensor; this port takes
the original embedding. The source hardcodes CLIPAdapter's input512 even though its actual KEEP prototypes are768D. An explicit native-width port uses768→192→768 for KEEP, retaining reduction4 and residual. CUDA-only tensor placement becomes ordinary module
device movement. Plain constant text tensors are registered as identical
buffers so checkpoints and CPU/GPU movement retain them. No loss or model
equation is otherwise replaced. Source defects and repairs remain explicit in
every run's `method_protocol.json`.

The public program samples k slides per class from its complete pool and calls
the remainder validation; it supplies no independent test partition. Our
patient-disjoint extension applies the **same NumPy classwise choice without
replacement** to the registered TRAIN pool only. VAL and TEST remain separate.
Every eligible TRAIN patient must have one slide in that pool; users with
multiple slides first register an explicit outcome-independent slide choice.
This route does not silently reinterpret slide shots as patient shots.

Supported k values are2/4/8/16, default8. Insufficient eligible patients fail
clearly; no fallback to a smaller k. The support seed is the run seed. ZSMIL's
source constructor resets Torch seed42 before building the classifier and
aggregator; other adapters retain the run seed. Python shuffling operates on
the **previous epoch's order**, as in the original `shuffle_data`.

The source default recipe is20epochs, AdamW1e-3/weight_decay1e-5/betas(.9,.999),
cosine T_max=epochs, one full bag per step and CE sum. There is no patch cap,
gradient clipping, early stopping or validation-based checkpoint selection.
Only the final epoch is selected; balanced accuracy is reported after training.
Checkpoint recovery includes the exact support IDs and order, native model,
trainable cache keys, fixed text/onehot labels, active parameter identities,
AdamW moments, cosine phase and all global RNG states.

## Text prototypes and data preparation

The author distributes six actual matrices: CONCH and KEEP for BRCA(IDC/ILC),
NSCLC(LUAD/LUSC), RCC(CHRCC/CCRCC/PRCC). Import verifies the bytes against the
fixed Git object. The exact generating prompts, text-ensemble code and encoder
weight revision are not recovered from those matrices. We report these limits
and do not claim to have regenerated the text ensemble or established an exact
undisclosed visual/text weight match.

```bash
milplus setup --method MIL-Adapter
milplus import-miladapter-prototypes --project BRCA --encoder CONCH --output /data/brca_text_v1
milplus prepare-miladapter --data /data/brca --metadata /data/brca/miladapter_metadata.json --output /data/brca/miladapter_v1/manifest.json
milplus run --data /data/brca --method MIL-Adapter --output /results/brca_taskres
```

Metadata must identify `kind: miladapter_original_prototypes_and_support_pool`,
ordered source class names, the imported `text_assets` path, feature source,
actual preprocessing, task semantics, `synthetic: false`,
`alignment_policy: author_native_visual_text_space`,
`alignment_evidence: author_released_prototypes_weight_revision_undisclosed`, and
`support_policy: source_classwise_sampling_from_train_pool_only_patient_disjoint_extension`.
The selected encoder declares model `CONCH` or `KEEP` and its actual weight
SHA256. Bind the prepared path in `encoders.<fm>.miladapter.manifest`.
Arbitrary unrelated FMs are not accepted as if their features aligned with the
author's text prototypes. Generation of new task-specific prototypes is not
silently fabricated by this implementation.

An explicit software fixture is available:

```bash
milplus demo --miladapter --patients 20 --output /tmp/miladapter_demo
milplus run --data /tmp/miladapter_demo --method MIL-Adapter --output /tmp/miladapter_run
```

These are synthetic visual bags with actual author-distributed text matrices,
not real WSI, actual extracted CONCH/KEEP features or clinical benchmark scores.
RCC fixtures require at least30patients. The separate installed-wheel validator
requires all26 profiles × both FM dimensions (52 runs), full512/768D/native aggregators, complete257patch bags,
default k8 and20epochs, actual original prototypes, full cache/state updates,
final-only selection and repeated independent checkpoint evaluation. Passing
software/native-budget gates does not establish paper metric reproduction.

## BRCA native CONCH experiment completed, 2026-11-06

The separately identified native CONCH group finished all856 original slide
bags and2422295 unchanged coordinate rows at01:30:28Z. Complete archives and
row identities were reverified before the existing scheduler accepted the
MIL-Adapter request; frozen UNI results are not replaced or merged with it.

The actual TaskRes/ABMIL recipe completed20 epochs and320 AdamW updates.
The author-native eight shots per class select16 distinct TRAIN patients
using sampling_seed2026 from the unchanged742-patient TRAIN pool. This is
the explicit source few-shot protocol, not742 supervised updates per epoch.
All16 support cases are TRAIN-only; full bags, native512 widths, LR.001,
weight decay1e-5, cosine scheduling and final-epoch-only selection remain.
Neither validation nor TEST selects the checkpoint. The preserved fitting
status correctly says test_evaluated=false; a separate receipt records the
subsequent locked45-patient TEST evaluation at01:32:49Z.

TEST AUROC is.9870689655, accuracy.9333333333, balanced accuracy.9482758621
and macro-F1.9298701299. All60 currently completed methods pass independent
TEST identity/probability/metric recomputation with zero errors.
These are preliminary single-split results, not original paper reproduction;
the author prototype generation prompts and historical weight revision remain
undisclosed. Selected checkpoint SHA256:
`276f3d74519da059bf6ece20b589393c87bbe23b060827ad0510c6fc0b030b5a`.

Unmodified request/status/inputs/protocol/environment/history and the native
input contract are retained in`results/miladapter/source_run_evidence_r1/`,
with`audit_receipt.json`. The prior59-method audit was preserved before
`completed_predictions_audit.json` was updated to60; earlier failures remain.
