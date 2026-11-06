# PathRWKV: complete heterogeneous training and explicit source repairs

The two execution profiles are `pathrwkv_source` and
`pathrwkv_repaired`; `PathRWKV` resolves to the repaired profile. They run the
full TimeMix/ChannelMix backbone, task-specific projection and heads, sampling,
losses, optimizer, scheduler, validation selection and checkpoint continuation.
They are not a generic cross-entropy wrapper around an RWKV layer.

Primary sources are the [author repository](https://github.com/Puzzle-Logic/PathRWKV)
at `8818fa5711666bd8d30672f02bdcd98634389b70`, Apache-2.0, and the
[paper, arXiv2503.03199v5](https://arxiv.org/html/2503.03199v5).
Author definitions and CUDA kernels are loaded from verified Git objects in an
external checkout. The documented source environment is Python3.12/Torch2.9.1/
CUDA12.8. The benchmark port uses its declared environment; this is not a claim
that the original Lightning environment was reproduced unchanged.

## Native network and recipe

Defaults retain hidden768, two blocks, 64-dimensional heads, ReLU/input
dropout0.25, one projected768-dimensional feature per task and maximum pooling.
Each classification task has a C-way head; each regression or survival task
has a scalar head. The author's global Xavier initialization is retained,
including its overwrite of the TimeMix constructor's zero output projection.

TRAIN uses a complete random permutation of each slide, truncates or zero-pads
both features and coordinates to2000 tiles, and does not mask padded tiles.
Batches contain four slides, are shuffled, and drop the final incomplete
batch. `num_workers=0` is explicit; source multiworker identity is not claimed.
Sequential VAL/TEST loaders still consume their base RNG seed, as in PyTorch's
original DataLoader. This draw is retained for exact subsequent epoch sampling.
VAL and TEST use all available tiles, in original row order.

The joint objective sums all observed task losses: classification CE, scalar
SmoothL1(beta1) and Cox partial likelihood. No invented task balancing is added.
Defaults are AdamW(lr1e-4,weight_decay0.01),100 epochs, cosine LambdaLR with
final factor0.1, BF16 mixed precision and no clipping. Paper validation-side
learning-rate search is an experiment-level budget, not automatically run by
one CLI invocation. Reduced tiles, dimensions or epochs must be reported as
changed conditions.

The first strict minimum of **distinct VAL joint loss** selects a checkpoint.
Early stopping independently uses min_delta1e-5 and patience10. Small strict
improvements may update the selected checkpoint without resetting patience.
The common engine advances the scheduler at its documented epoch boundary,
before validation/checkpoint serialization; no Lightning callback-state pickle
identity is claimed. Model, AdamW, scheduler, AMP scaler, RNG, early stopping,
configuration/input contracts and update counters support exact epoch-boundary
continuation in the benchmark engine. Outer TEST never selects a checkpoint.

## Source behavior and named corrections

`pathrwkv_source` retains two observed source behaviors. Its Cox function gets
`risk[B,1]` and `event[B]`, so multiplication broadcasts to[B,B]; event masking
therefore differs from the intended per-patient Cox loss. Missing regression
targets use-1, which also ignores a legitimate observed-1 value. The source
route reports these facts and rejects a batch whose sentinels erase every
objective. It does not claim historical paper scores were produced by this bug.

The repaired route uses a separate boolean observed mask, preserves valid-1
regression targets, flattens risk and uses Breslow handling of tied times.
All-censored tasks contribute an independent zero leaf, matching the source's
absence of model gradient; no gradient or AdamW decay is invented for a wholly
unobserved head. Actual no-op batches are recorded separately from updates.

Source chunked inference carries the WKV matrix but resets both TimeMix and
ChannelMix's shifted previous normalized token at every chunk boundary.
`pathrwkv_source` preserves this behavior. The repaired route carries the WKV
matrix **and both token-shift states**, projects raw inputs chunk by chunk and
maintains the task-specific global maximum. This matches full-sequence
evaluation within numerical tolerance and bounds feature activations on the
GPU by the declared chunk budget. PT reading still materializes one CPU bag;
H5/NPY read only the requested chunk. No bounded-total-CPU-memory claim is made
for PT or the common input inventory.

The source's1200²×768 position table alone exceeds4GB. Its exact separable
CPU sine/cosine values are retained as a1200×384 cache, without interpolation
or approximated positions. Invalid coordinates are rejected per axis rather
than allowed to wrap through the original flattened index.

## Input and heterogeneous targets

Ordinary classification/ordinal, regression and survival task bundles are
supported. Ordinal labels here use the author's CE classifier, not an invented
cumulative ordinal head. Standard survival reporting averages slide risks per
patient, while source training and validation loss remain slide based.
Standardized probabilities use float32 softmax of emitted logits; BF16-quantized
TorchMetrics probabilities and undefined-metric fallback zeros are not copied.

Real row-aligned coordinates are mandatory. TRIDENT xy coordinates must declare
their order, origin and physical patch pitch. For example:

```yaml
encoders:
  uni:
    # Other normal feature-index/weights/preprocessing fields are required.
    spatial:
      unit: level0_pixels
      origin: top_left
      patch_size: 256
      coordinate_order: xy
      provenance: "actual TRIDENT extraction configuration/version"
```

The adapter converts declared top-left/center and pitch to the author's224-grid
yx coordinates. It does not fabricate a square grid from feature order.
`patch_size` is the actual physical grid pitch in the declared coordinate unit;
overlapping/irregular sampling requires an explicitly justified position mapping,
not an assumed224/256 scale. Choose a larger explicit `grids` budget if necessary.

Mixed targets additionally use `task_type: multitask` and an explicit table:

```yaml
multitask:
  schema_version: 1
  table: targets.csv
  tasks:
    - name: diagnosis
      type: classification
      column: diagnosis
      classes: [negative, positive]
      unit: patient
      label_provenance: "versioned pathology labels"
    - name: expression
      type: regression
      column: expression
      unit: patient
      label_provenance: "versioned assay and preprocessing"
    - name: OS
      type: survival
      time_column: OS_time
      event_column: OS_event
      time_unit: months
      unit: patient
      label_provenance: "versioned endpoint definition"
```

`targets.csv` must contain each declared slide exactly once. Missing values are
empty cells; survival time/event are jointly missing or observed. Unknown
classes, nonfinite values, inconsistent patient outcomes and wholly unlabeled
included slides fail validation. Each declared task needs observed TRAIN labels.
Per-endpoint metrics use only observed labels and their declared slide/patient
unit; undefined C-index/AUROC remains null. Different task metrics are not averaged
into a fabricated single score.

## Execution and validation boundary

```bash
milplus setup --method PathRWKV
milplus run --data /path/to/prepared_task --method PathRWKV \
  --device cuda --output /path/to/new_run
milplus evaluate --run /path/to/new_run --device cuda
```

The default `backend: native` compiles the exact pinned CUDA kernels with64-wide
heads and2000-token kernel limit. It fails when CUDA/toolchain requirements are
not met and never falls back silently. Native stateful kernels are certified
only for no-grad inference; their source backward is not used for training.
For an explicitly identified CPU reference experiment set `model.backend: torch`
and `model.precision: '32'`. This is a differentiable tensor transcription,
checked against independent closed-form causal sums and every input gradient;
it must not be reported as executing the native CUDA kernels.

Validation evidence is recorded separately in `docs/validation` and
[VALIDATION.md](VALIDATION.md). Short synthetic runs demonstrate software
mechanisms, not completed100epoch experiments, natural BRCA/COAD training,
clinical performance or paper-number reproduction.

## 0.52 acceptance evidence

The fresh complete CPU suite passed2292cases without failures/errors/skips; all205profiles actually trained, saved, reloaded and predicted. The PathRWKV focused set has53cases. Independent original-source native768 trainers matched16AdamWupdates across1024/1536inputs, including all weights/optimizer/scheduler/RNG. The extracted CPU wheel completed four native768/three-head two-epoch runs.

Actual H200 CUDA evidence covers both original kernels, all five input gradients, matrix carry, strict FP32 tensor-reference full-model gradients, bitwise independent original-native FP32/BF16 outputs/loss/all60parameter gradients/AdamW updates, corrected full-versus-stream inference, and four complete two-epoch native CLI train/selected-reload/repeated-prediction runs (32updates). T=1 decay has an analytically absent reference edge; precisely that structural zero is tested, with no relaxation of longer-sequence gradients or tolerances. An additional actual default four-slide by2000-tile BF16 native training step matched original-source outputs/loss/all gradients/updated weights/AdamW moments bitwise. Source stateful backward remains outside the supported training contract.

The BF16 pure-tensor subgradient comparison did NOT pass:73/1769472MTL-weight elements exceeded the unchanged0.035threshold, with9quantized max-selection channels differing. Its failed R4 report and R5 diagnostic remain in the development evidence directory. Native BF16 acceptance instead uses strict bitwise independent original-source CUDA fidelity, not a relaxed tensor tolerance. Reports are pathrwkv_cpu.json/pathrwkv_cuda.json/pathrwkv_wheel_052.json in docs/validation. This is software acceptance on synthetic data, not a completed100epoch natural WSI experiment or paper-score reproduction.

The preceding CUDA and wheel evidence belongs to the accepted0.52 runtime identity. Later releases preserve it as historical method evidence, without relabeling it as GPU verification of an entire newer package.
