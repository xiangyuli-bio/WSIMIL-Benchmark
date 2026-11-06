# DeepGraphSurv: complete adaptive graph survival candidate

This is an **unaccepted0.57 candidate** while complete-library, all-profile and
independent native-budget gates run. The accepted main repository remains the
source of formal method counts. Software tests do not establish natural WSI or
paper-score reproduction.

Author source: <https://github.com/uta-smile/DeepGraphSurv>, commit
`3149752deba68067f1a39b9f3d212cb16ffaf560`. The README identifies Li et al.,
MICCAI2018, *Graph CNN for Survival Analysis on Whole Slide Pathological Images*.
No license file was found at this commit; original definitions are obtained from
the user's pinned checkout rather than bundled in this repository.

The execution ID is `deepgraphsurv_agcn`; `DeepGraphSurv` resolves to it. It
explicitly selects `basic_agcn_wsi_survival2(use_attn=True)`. The author top-level
script currently hardcodes `MODEL_NAME='chebyshev'` despite its AGCN CLI default;
running that script unchanged is a different network branch.

## Inputs and representation

Use a survival task bundle with real patient/slide mapping, actual time units,
fixed patient-separated splits and full-dimensional frozen patch features.
Physical coordinates are not required: the original graph is a **feature-space
graph**, rather than a physical-neighbor graph. Multiple WSIs per patient are
supported. All WSIs use the same registered representation and feature order.

The author extraction pipeline uses ImageNet VGG16 `fc2` features and TRAIN-only
PCA128 with whitening, sampling10percent of each training WSI using
`StratifiedShuffleSplit(random_state=1)`. That original representation can be
supplied as an explicitly documented encoder condition. The FM benchmark port
retains **all** registered FM channels, while keeping original attention and
backbone widths. It does not truncate a1024/1536dimensional FM to128 or claim
arbitrary FM inputs are the original VGG/PCA experiment. Automated execution of
the old Python2/OpenCV/VGG preprocessing script is not claimed.

Prepare a schema:

```json
{
  "schema_version": 1,
  "kind": "deepgraphsurv_feature_graph",
  "synthetic": false,
  "representation": "frozen_features_full_dimension",
  "representation_provenance": "Actual registered FM, weight version and extraction protocol",
  "selection_provenance": "Actual included diagnostic slides and source order",
  "max_nodes": 500,
  "sampling_seed": 1,
  "slide_order": {
    "train": ["actual_train_slide_ids"],
    "val": ["actual_validation_slide_ids"],
    "test": ["actual_test_slide_ids"]
  }
}
```

```bash
milplus setup --method DeepGraphSurv
milplus prepare-deepgraphsurv --data /data/task \
  --schema /data/task/deepgraphsurv_schema.json \
  --output /data/task/deepgraphsurv_assets
```

Bind `encoders.<fm>.deepgraphsurv.manifest` in `task.yaml` to the resulting
manifest, then run:

```bash
milplus run --data /data/task --method DeepGraphSurv --output /data/run
```

Each subset's order must exactly cover its selected WSIs without duplication.
The task must explicitly filter slides with fewer than4patches, matching the
source exclusion. A new preparation directory is required; existing assets are
not overwritten. Feature identity, selected indices, graph arrays and contents
are bound to hashes. The loader can independently rebuild each graph from the
unchanged original source using `assets.load(row,rebuild=True)`.

## Complete graph and native equations

The source loader permutes patches even for TEST, takes at most500 actual nodes
and produces one graph per WSI. The adapter freezes this sampling once, including
its seed and raw indices. It does not split a WSI into invented independent
samples or pad a short WSI with fake graph nodes. Original `-1` padding remains
part of native batch input.

The graph cutoff uses all lower-triangle feature distances, **including diagonal
zeros**, at `int(.3*n_pairs)`; edges use strict `<`. The actual source adds
self-loops, symmetrically normalizes adjacency, and then computes the
unnormalized degree-minus-normalized-adjacency Laplacian. It is not replaced by
generic kNN, a physical graph, `I-A`, or a rescaled Chebyshev Laplacian.

The full native model contains graph attention32→1, backbone256→256→128, K3
Chebyshev filters, learned residual feature graphs at every layer, population BN
and EMA,20percent dropout, and tanh64→tanh1 log risk. Literal details retained:

- The source uses a plus-Gram formula in its learned graph, column L2
  normalization with epsilon on squared norm, and column-sum graph degrees.
- Each layer adds its residual to the original intrinsic Laplacian.
- Attention divides by1000, applies sigmoid and a node softmax over the padded
  maximum. Actual-node gathering does not renormalize lost padding mass.
- Dropout is enabled in the original network even during inference.
- BN uses population variance, epsilon1e-3 and EMA states initially zero.

The TensorFlow1 equations are ported to Torch rather than requiring TF in the
runtime environment. Independent tests execute the unchanged author functions
in TensorFlow2.15.1 `compat.v1`; removed `tf.contrib` Xavier initialization is
mapped to its compatible Glorot initializer. Identical original weights and
actual TF-generated dropout masks permit comparison of outputs, literal Cox,
input/all30parameter gradients and BN EMA at128/1024/1536input widths. A separate
actual TF1Adam `apply_gradients` comparison uses identical captured original
loss gradients to isolate optimizer arithmetic and prevent read/update races.

## Full source training and explicit corrections

The adapter executes the actual original `train_one_epoch` function through a
session I/O bridge. Native defaults are100epochs,64graph batches, max500nodes,
base LR2e-4 and TF1Adam. The outer author loop overrides its CLI LR default with
2e-4; no optimizer weight decay or collected layer/graph regularizers enter the
source Cox objective.

A training window must contain observed and censored patients. Otherwise the
source shifts its start by1 until a valid full window exists; a final incomplete
window is dropped. Accepted batches are sorted by descending float32 clinical
time. Literal sum-Cox uses ordered `log(cumsum(exp(logrisk)))`, with source tie
ordering; it is not replaced by Breslow ties or event-mean normalization.
The original reporting denominator remains `floor(N/batch_size)`.

LR is clipped at1e-8 and decays by `.8**floor(update*batch_size/2000)`.
BN decay is `min(.99,1-.5*.5**floor(update*batch_size/2000))`. The optimizer
retains TF1's global float32 beta powers and epsilon outside uncorrected
`sqrt(v)`; a generic Torch/Keras Adam is not substituted.

Documented boundary/protocol corrections:

- Explicit clinical IDs replace TCGA-prefix/NLST-filename lookups and hardcoded
  cohort truncation. The provider repairs the original five-return/four-unpack
  mismatch without changing epoch windows or updates.
- Full attention, BN, optimizer, counters and RNG are checkpointed. The original
  saver omits all8attention tensors and BN/optimizer state.
- Source evaluation tiles partial batches and trims duplicate outputs; the
  small-subset `p_feature` variable is repaired. Duplicates do not enter metrics.
- Undefined batch-CI diagnostics remain null. They do not become invented
  scores and do not crash otherwise valid Cox updates or singleton-tail
  inference. A selected global validation patient CI must be defined.
- The original every5epoch schedule after epoch0 selects on **VAL**, preserving
  the first maximum. Outer TEST is evaluated only on explicit request.
- Inference dropout remains enabled with a declared per-subset/completed-epoch
  random stream, isolated from TRAIN RNG, so a selected checkpoint can be
  evaluated repeatedly. Original unseeded inference is stochastic.

Patient risk is `mean(exp(logrisk))` across the patient's WSIs: the source first
computes `-exp(logrisk)` as prognosis and then averages it. Averaging log risks
would change this rule. Full-precision clinical times and native float32 times
are both retained; WSI and patient CI diagnostics remain separate.

The original provider assembles padded population arrays, which can consume
substantial memory with wide FMs and many WSIs. This cost belongs in benchmark
resource accounting. Native synthetic validation uses actual small graphs with
the original500node cap and padding; it does not establish500real-node WSI
performance or GPU throughput.

## Verification state

Current candidate:19focused checks passed, including independent original
TensorFlow comparisons at all three widths, literal graphs, original source
epochs, complete-state resume, patient aggregation, undefined diagnostics,
source event-window exclusions, six-epoch selected-checkpoint reload and CLI.
Two independent-wheel1024/1536native-budget runs are in progress. Complete
current-library and all-profile acceptance remain pending. CUDA, natural
BRCA/COAD data and paper numbers have not been verified.
