# MUSE: released-source condition

Author code: <https://github.com/JiahaoXu-god/CVPR2026_MUSE>, commit
`0bd0e1abb402e772ed71cf208cfbd4b870175a00`.
Paper: <https://arxiv.org/html/2602.20873v1>.
The pinned tree has no LICENSE; the package privately loads the original Git
definitions and does not redistribute their source or language-bank files.

`muse_released_source` preserves the complete released implementation. It has
88,985,107 parameters at native 512D input: the full twelve-layer 768D CONCH text
path, sixteen shared context tokens, eight noisy experts with top-two routing,
eight-head visual/text attention, and a learned visual adapter. It retains all
300 original descriptions per class and all twenty retrieved views, with a
separate optimizer update for every view. Its supported cohorts are BRCA
(IDC/ILC), NSCLC (LUAD/LUSC), and CAMELYON (normal/tumor), with explicit label
order and patient-safe splits. Source-supported 768/1024/1536D visual adapter
conditions are separately identified in input metadata.

The released recipe uses Adam at .0001, weight decay .00001, batch one, constant
LR, up to 200 epochs, strict validation error selection, patience twenty, and
the original earliest stopping condition `epoch_index > 80`. The instantiated
plateau scheduler is never stepped by the original entry, so this port does
not add a schedule. The CLI's optional source SGD and weighted sampling are
explicit model settings. Original CPU-zero-worker/GPU-four-worker loading and
even the unused random coordinate draws are preserved; these tensors are not
presented as physical patch coordinates.

The released validation callback reads private example paths and writes a
diagnostic export. These files are not available to this benchmark, and no
private path is accessed. The callback nevertheless resets Python random state
and consumes 651 noisy-router draws. The port reproduces those exact draw
shapes/order and Python effects without the private diagnostic. An independent
execution of the literal callback with isolated synthetic diagnostic I/O
verifies exact Torch/Python RNG equivalence. This proves the RNG port, not the
content of the omitted visualization.

The released router remains noisy in evaluation. Repeated benchmark inference
starts from the saved run's explicit seed; it does not silently disable noise.
Checkpoints contain the complete text/adapter/expert parameters, fixed language
and prompt buffers, positive/negative retrieval caches, per-epoch bag/view
histories, actual sparse parameter update counts, optimizer states, and RNG.
An expert can be unused in a view, so its Adam step count need not equal the
total number of updates. Counts are checked against actual non-None gradients.

## Source and paper differences

This released-source condition is not claimed as a numerically reproduced or
fully reconciled paper protocol. The audit records these concrete differences:

- The source replaces sixteen token positions without first prepending context
  placeholders to its short class names. The actual fixed class prefix and
  suffix tensors are identical between the two classes in the checked BRCA
  model. The paper describes retaining the class name after context tokens.
- The source's recursive initializer reinitializes text linear and normalization
  layers after its original fixed knowledge caches have been computed. The
  projection/position parameters remain loaded, while many text weights change.
- Source cosine retrieval selects the least similar positive-class entries;
  the paper describes selecting the most similar entries.
- The released training loop processes the retrieved order directly, whereas
  the paper describes a shuffled queue.
- The source sums the main and auxiliary logits; the paper gives their mean.
- Source top-patch filtering is applied on the auxiliary path. Its manual
  auxiliary head reshape and second attention scaling also differ from the
  paper's multi-head equations.

These differences remain visible in `method_protocol.json`. A reconciled paper
variant and its evidence are separate work; replacing these equations silently
would destroy the released-source reference. Formal admission remains pending.

## Running

```bash
python -m milplus demo --muse --muse-weights /absolute/path/to/conch.pth \
  --muse-cohort tcga_brca --patients 24 --output /tmp/muse_demo
python -m milplus run --data /tmp/muse_demo --method MUSE --output /tmp/muse_run
```

This example has four synthetic TRAIN subjects per class and a full twenty-view
update for each visit; it does not supply clinical results. For a real task,
prepare its standard bundle, then run `prepare-muse --data TASK --metadata META
--output VERSION/manifest.json` and register that manifest as the encoder's
`muse.manifest`. The metadata binds all feature bytes/order, labels, patients,
splits, complete knowledge files, actual checkpoint and source condition.

Initial construction encodes the complete knowledge banks and can require
substantial RAM. The current full-model checks run in the 220GiB Worker.
Two-epoch, native512/257patch source checks have completed actual interruption,
resume and two independent selected-checkpoint reloads. Full source-loop
comparison, installed-wheel, full200-epoch-limit CUDA budgets, full regression,
and the paper/source reconciliation have distinct acceptance evidence.

## Actual BRCA UNI1024 input admission (2026-11-06)

The locked BRCA task now has a complete real input version at GPU workspace
`inputs_native_muse_uni1024`. It retains all856 patients,742/69/45 splits and
2,422,295 unchanged UNI1024 patch rows. There is exactly one slide per patient;
the original MUSE slide training/evaluation units are explicit and not relabeled
as patient-unit fields. This is the named
`uni_frozen_muse_released_source_adapter1024` condition, not native CONCH visual
features or a reproduction of the paper's original FM condition.

The pinned original model itself declares `input_dim=1024` and its learned
`Linear(input_dim,512)` visual adapter. No extra projection or model rewrite was
introduced. Its actual BRCA knowledge CSVs contain all300 IDC and300 ILC
descriptions; their complete pinned byte hashes, original model source, actual
CONCH text checkpoint and all856 UNI files are bound in the manifest and
independently rehashed by the input admission gate. No new clinical descriptions
or region labels were generated, and no patient labels were fitted during asset
preparation. All source/paper differences described above remain in force.

The full input contract completed at2026-11-05T21:07:09Z. After148 inherited/new
queue-control cases passed on the actual node, plus a separate real856-file
intake dry check, only the queue owner was handed from V8 to V9 at21:14:57Z.
All13 then-live training identities remained unchanged. At21:15:42Z the same
queue accepted `muse_native_uni1_20261106.json`, with the unchanged
`muse_released_source`200-epoch maximum,20 separate retrieved-view updates,
Adam1e-4/wd1e-5 and source validation-error stopping rules.

This is `ready_for_training_attempt`, not a model initialization, CUDA fit,
completed clinical result or final package admission. Complete original language
cache initialization can be large:160GiB host/96GiB GPU reservations and exclusive
GPU admission are conservative guards, not measured clinical peak claims. Banks,
views, bags and budgets must not be reduced to satisfy them. No live method or
runtime code was modified.

Evidence in the BRCA shared directory: `native_muse_uni_preparation.json`,
`muse_full856_actual_intake_dry_check_r1.json`,
`scheduler_muse_controls_all_actual_r1.xml`, `scheduler_v9_handoff_launch.json`,
and `verified_input_intake_receipts/muse_native_uni1_20261106.json`.
The native contract SHA256 is
`f6e5d59a69ef25bf050bc30809c948e98870e61f9455aee93001a32a4fdf566f`.
Seven task/manifest/receipt files totaling about2.30MB are backed up byte-exactly
under `muse_native_uni_inputs_backup_r1`; original absolute bindings remain
unchanged, and no UNI features or pretrained weights were duplicated.

## Bounded CPU initialization diagnostic (2026-11-06)

A separate CUDA-hidden child attempted the unchanged real856-patient UNI1024
constructor with gradients enabled and the original full text batches. Before
starting, observed host availability was53,641,232,384 bytes and outstanding
live-fit reservations were21,045,432,320 bytes. Only this diagnostic child was
given an additional20GiB virtual-address-space allowance and an8GiB host guard;
the queue and existing fits were not changed.

The actual first `TextEncoder` input was `[300,128,768]` float32 with gradients
enabled and a gradient-requiring input. It did not return: the CPU allocator
reported `ENOMEM` for a471,859,200-byte allocation at23:37:02Z. Thus the complete
300+300+600+2 initialization did **not** pass. The intentional `RLIMIT_AS` makes
this a bounded diagnostic failure, not proof of a kernel OOM, a measured full
RAM peak, a GPU-capacity result, or justification for lowering the160/96GiB
reservations. No smaller bank, detached/no-grad initialization, model change,
optimizer update, or GPU execution was substituted. MUSE remains ready and is
not counted as a completed clinical method.

Evidence: `measure_muse_original_cpu_initialization_capacity_r1.py`,
`muse_original_cpu_initialization_capacity_launch_r1.json`,
`muse_original_cpu_initialization_capacity_failed_r1.json`, and
`muse_original_cpu_initialization_capacity_diagnostic_audit_r1.json`.
