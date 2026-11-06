# LNPL-MIL: actual RGB/LPA, weak training, SP and TOD

The public entry runs all stages; a feature-only TOD head is not LNPL-MIL.
The source at `1dc4e703a2b86dd0950e0b75aa7d45337d2127cb` is loaded privately.
Its repository has no verified LICENSE; original code/weights are not bundled.

The released weak-classifier recipe is absent. This implementation explicitly
reconstructs a supervised reference with actual observed TRAIN patch labels,
full ImageNet-initialized ResNet18, SGD/Nesterov0.9/LR0.02, cosine50epochs,
batch256, horizontal flips, ImageNet normalization and minimum TRAIN patch CE.
That reference must not be called the original unpublished training recipe.
Scoring uses the selected weak probabilities and a separate untouched/frozen
ImageNet ResNet18 branch for512D features, bound to actual ordered patch IDs,
coordinates, image bytes, initialization and patient partitions.

Two full execution conditions are available:

- `lnplmil_released_with_reference_weak`: original one-shot HNSW SP50/ROI0.4,
  original minimum-size rule, Top200 score ordering, released convolution input
  replacement and released CE overwrite.
- `lnplmil_paper_with_reference_weak` (alias `LNPL-MIL`): iterative disjoint HNSW,
  full retained clusters, Top200, Eq2 convolution residual and0.001 distribution
  auxiliary CE over four strict-positive fraction intervals.

Both preserve the full four-layer256D/eight-head/FF1024/512D-input200-instance
TOD, literal Lookahead/RAdam2e-4/wd1e-5/k6/alpha0.5, gradient accumulation2,
VAL-bag-CE selection, maximum200epochs and patience10. CUDA uses explicit
float16 autocast/GradScaler; CPU float32 is a separate software condition.
The implemented end point is binary slide diagnosis. The original paper's
COAD survival task is not claimed by these profiles.

Public runs explicitly use `indexThreadQty=1` for HNSW index construction.
Original M16/post0/efConstruction400/query-ef90, approximate search, ROI and TopK
equations remain. This canonical construction policy is recorded in every
selection manifest/protocol; it is distinct from literal source default-thread
execution, which remains available in the component API. A preserved failed
cross-process check found identical weak models and all original scored features
but different selected patch identities under parallel construction. Four fresh
processes now verify both SP conditions exactly. No numerical tolerance was
relaxed and no patch population or model width was reduced.

Public candidate pools are also ordered by actual level0 X/Y before the unchanged
source `torch.topk`. Full native paper testing retained the exact same patch
multisets in all six bags but exposed a different HNSW neighbor-return order
among repeated/tied visual instances. Canonical XY ordering fixes that declared
tie condition while preserving every score, feature, coordinate and repeated
instance. Literal default-order component execution remains separate. Original
failed native checkpoints and all strict numerical assertions are preserved.

Native precision16 keeps GradScaler dynamic-loss-scaling behavior: an overflow
skips a real optimizer update and reduces scale. Actual Lookahead steps count
updates; skipped attempts are recorded separately. CPU nonfinite gradients still
fail. The full4,304,095parameter TOD and actual200x515 inputs have passed a CUDA
overflow-skip probe with every model tensor unchanged; that probe is not an
epoch-budget result.

## One-time data preparation

Create an ordinary task bundle with real slide labels, global patient IDs and
TRAIN/VAL/TEST partitions. The encoder index points to actual HDF5 RGB arrays;
its dimension is3. Each file must contain RGB uint8 `[N,224,224,3]`, unique
ordered patch IDs, original level0 integer coordinates, observed labelled patch
indices and actual patch classes. N must be at least200. The source discrete
position grid covers coordinates below240000pixels. Only TRAIN can provide LPA.
The schema records measured magnification/MPP, image/annotation provenance,
weak classes and positive-class IDs, and a directory containing the checksum-
verified official ResNet18 weights and support source. Two or nine weak-class
constructors exist; clinical nine-class/survival validation remains separate.

```bash
milplus prepare-lnplmil --data /data/diagnosis --schema /data/schema.yaml \
  --output lnplmil_manifest.json
```

Attach `lnplmil: {manifest: lnplmil_manifest.json}` to the selected encoder in
task.yaml. Preparation is explicit and does not overwrite old manifests.
Observed labels are never filled from a slide label. An ordinary TRIDENT feature
folder alone does not satisfy the additional RGB/LPA input requirements.

## Complete public execution

```bash
milplus run --data /data/diagnosis --method LNPL-MIL --device cuda:0 --output runs/lnpl
milplus evaluate --run runs/lnpl --device cuda:0
```

The default budgets are50weak epochs and200maximum TOD epochs/patience10.
Optional YAML keys are `weak_epochs`, `weak_batch_size`, `epochs`, `patience`
and `threads`. Smaller budgets are explicit software checks, not native results.
There is no hidden/patch-count/model-width override.

`--stop-after` counts real weak optimizer updates followed by real TOD slide
minibatches. Recovery retains model/BN, optimizer/slow buffers, scaler,
accumulated gradients, scheduler, order/offset/history and all RNG states.
Full requests bind source/runtime/data/recipe identities. Run-local cache
generations are committed by manifest/checksum pointers only after verification;
uncommitted partial generations remain preserved. Distinct runs can have
different absolute cache paths and encoded HDF5/checkpoint hashes. Each binding
is checked against its own request, while learned states and decoded inputs
are compared exactly; incidental filenames are not scientific outputs.

## Failures and acceptance

The literal released SP can fail with `np.stack([])` on tied/empty eligible ROI
before its minimum-size fallback. This remains a recorded explicit rejection,
not an invented unfiltered-bag fallback. The paper path similarly rejects an
insufficient selected population. The original failed constant-image check is
preserved alongside the successful401-patch varied RGB software fixture.

The frozen parent passed32focused checks and complete source/paper software
pipelines. Public integration, native50/200 budgets, installed-wheel and full
library checks are distinct gates on this new source identity. Registration as
`adapted_under_verification` does not certify any unexecuted gate, clinical WSI
experiment or paper-score reproduction. Formal admission remains false.
