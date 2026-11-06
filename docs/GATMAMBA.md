# GAT-Mamba: spatial graphs and the complete survival training recipe

The execution profiles are `gatmamba_source` and
`gatmamba_edge_conditioned`. `GAT-Mamba` resolves to the source profile.
Version0.54 acceptance includes all2343current CPU cases,209actual profile runs and four independent-wheel original-budget/native-width training runs. CUDA and natural WSI verification remain separate.

The original source is [rina-ding/gat-mamba](https://github.com/rina-ding/gat-mamba)
at `18979497990a66ffe6a75880f0336f85914552f0` (GPL-3.0). Its declared Mamba
dependency is version1.2.0.post1, separately pinned to
`state-spaces/mamba@34076d664838588a3c97727b263478ab9f621a07` (Apache-2.0).
Both sources are externally fetched and read from verified Git objects.

## Actual model and native source behavior

Native defaults are a64-dimensional projection of the entire FM feature,
16-dimensional xy sinusoidal position embedding,80-channel single-head GAT,
one80-channel Mamba with state16, convolution4 and expand1, residual MLP and
three batch-normalization modules. The patient graph is mean-pooled and a
40/20/scalar MLP predicts an unbounded log risk. The native original input is
1024-dimensional UNI; this port explicitly parameterizes the input projection
and slice for all registered FM channels. It never silently truncates a wider
FM to1024. Such FM substitution is an experimental condition, not the original
paper's UNI result.

The source computes a16-dimensional embedding of21 tissue-pair categories plus
a projection of two continuous edge distances. Its `GATConv`, however, omits
`edge_dim`. Under the declared PyG2.6.1 environment those edge attributes do
not enter predictions: changing them all leaves risks bitwise unchanged, and
edge encoders have no data gradients. The source's all-parameter L1 still
updates those weights. This is retained in `gatmamba_source`, without claiming
that a different historical paper environment behaved identically.

`gatmamba_edge_conditioned` explicitly supplies `edge_dim=16` and uses PyG's
declared singleton batch-normalization behavior for very small graphs. It is
a named wiring correction, not a silent replacement of the source baseline.
Original single-head channel wiring is retained; the inconsistent original
multihead shape configuration is rejected explicitly.

## Physical graphs and additional information

Every WSI needs true row-aligned level0-pixel xy coordinates, FM features and
six-class LUAD subtype predictions in the exact order:
`lepidic,acinar,papillary,micropapillary,solid,nontumor`.
The original source uses a three-model pretrained subtype ensemble. A different
classifier must be declared explicitly, with weight hashes, revision,
training population and ensemble rule. This extra information and computation
must be accounted for in a benchmark. Applying a LUAD classifier to another
cancer is a distinct transfer condition, not evidence of meaningful tissue
typing in that cancer.

The classifier registration declares either `fit_subset: train` with unique
TRAIN global patient IDs in `training_patients`, or
`fit_subset: external_declared` with its actual training population. VAL/TEST
fitting is rejected. Unknown external cohort exposure must remain explicitly
unknown; declaring an external classifier does not prove absence of overlap.

Preparation reorders **all arrays together** by actual y/x row-major position,
realizing the source's natural r/c tile-name ordering. It does not infer a
square lattice from the feature row index. Center coordinates are translated
to top-left using the declared physical patch pitch. Noninteger, negative or
incompatible coordinate frames are rejected rather than rounded.

The source graph is eight-neighbor sklearn KNN in actual pixel space,
converted to an undirected PyG graph with mean reduction. Edge features are
the original21 unordered subtype pairs, pixel L2 distance and the full FM
feature L2 distance. Distances use float64 then the source float32 graph
representation. The source route requires at least10 tiles per WSI, matching
the original explicit filtering. The corrected route uses min(8,N-1) for
smaller graphs; it does not duplicate nodes or silently remove patients.

Multiple slides of a patient remain separate spatial graphs with no fabricated
cross-slide edges. They are concatenated in slide-ID order, as in the original
patient bag; the Mamba scans the resulting complete patient sequence. Clinical
labels must agree across those slides. Patient sampling uses the source's
natural ID order and shuffles with PyTorch's actual DataLoader.

## Complete training and selection

The recipe is mean minibatch Cox plus1e-5 L1 over **all** network parameters,
Adam(lr5e-5,weight_decay1e-4), batch16 including short final batches, no gradient
clipping and the original shuffled TRAIN and VAL loaders. Original clinical
Torch times are float32 for native risk sets and native C-index; original
table precision remains available for standardized clinical reporting.

Source logging multiplies each batch loss by its number of graphs and divides
the sum by the number of batches. The native validation loss preserves this
definition and the actual shuffled minibatch risk sets. It is not replaced by
a cohort-wide Cox loss or patient-count-weighted mean. The first and every
later `<=` minimum saves the checkpoint, so ties select the **last** epoch.
ReduceLROnPlateau uses min mode and patience5. Early stopping occurs after
**more than** five unsuccessful epochs. The original `while epoch <=200`
executes up to201 updates of the epoch counter; default CLI epochs is therefore
201, with the inclusive-source argument recorded in the protocol.

Outer TEST is sealed during model selection. Checkpoints contain the complete
model, optimizer, plateau scheduler, RNG, clinical/input/backend contracts and
selection counters for exact epoch-boundary continuation. Source C-index uses
the separately pinned lifelines metric, including observed/censored time ties;
it is reported alongside the common benchmark C-index rather than conflating
different tie conventions.

## Preparation and execution

The category CSV has `slide_id,path` and covers exactly the task WSI inventory.
Each path names an H5 with `coords` and integer `node_types`, aligned exactly
to the primary FM rows. A versioned JSON schema identifies the six-class
vocabulary, actual classifier weights/revision, ensemble/training provenance,
physical graph construction and patch order. The synthetic demo shows the
schema structure; its Gaussian features/modulo labels are permanently marked
synthetic and cannot serve as real tissue predictions.

```bash
milplus setup --method GAT-Mamba
milplus prepare-gatmamba --data /path/to/task --index /path/to/categories.csv \
  --schema /path/to/schema.json --output /path/to/new_graph_assets
```

Register the returned manifest as `encoders.YOUR_FM.gatmamba.manifest`, retain
the ordinary FM weight/preprocessing registration and declare
`spatial.coordinate_order: xy`, `spatial.unit: level0_pixels`, origin, actual
patch pitch and extraction provenance. Feature/coordinate/graph hashes and
the FM identity are frozen; changed assets cannot silently reuse a checkpoint.

```bash
milplus check --data /path/to/task --method GAT-Mamba --deep
milplus run --data /path/to/task --method GAT-Mamba --output /path/to/new_run
milplus evaluate --run /path/to/new_run
```

The default explicit `model.backend: torch_reference` executes the actual
same-version author Mamba slow branch and `selective_scan_ref`. This backend
is identified in every protocol; it is not reported as running a native CUDA
kernel. `model.backend: cuda` requires the declared native package versions,
matching installed scan source and an accessible GPU, records binary hashes
and fails cleanly if unavailable. It never falls back automatically.

These engineering checks use synthetic graphs. They do not establish natural
BRCA/COAD performance or reproduction of published paper metrics.

The unified evaluator exports patient risk/time/event, standardized C-index and source-native C-index/batch loss. The original plotting and source-specific posthoc dynamic-AUC script are not run as part of checkpoint selection or substituted for the common evaluator.

Independent-package native-budget verification can be reproduced with `python scripts/validate_gatmamba_wheel.py --wheel /path/to/package.whl --source-tree /path/to/checkout --output /path/to/new_evidence`. It creates only explicitly synthetic data and uses the CPU reference backend.
