# C²MIL: complete causal graph survival recipe

The execution profile is `c2mil_source`; `C2MIL` resolves to it.
Version0.55 acceptance includes all2361 current CPU cases,210 actual profile runs and two independent-wheel original100epoch/native-width runs. Natural WSI, GPU and paper-score validation remain separate.

The original is [mimic0127/C2MIL](https://github.com/mimic0127/C2MIL), MIT,
commit `fd4d54fc115466e43019c77015cdef01d90c85c3`. Original modules are fetched
separately and read from verified Git objects in a private namespace. The
complete bias-removal MLP/adaptive soft clustering, four-layer residual
PatchGCN, graph transformer, Bernoulli straight-through causal selection,
three-view attention pooling and risk classifier execute from that source.
Unused source modules remain initialized, retaining their parameter cost and
initialization RNG effects. Unused scatter/timm/rotary/plot imports are omitted;
the original disabled rotary path is not activated.

## Inputs and source population

Each patient requires one explicitly selected diagnostic WSI, physical patch
coordinates, full FM patch features and a separately computed whole-slide
thumbnail embedding. The source selects `patientID-01Z-00-DX1`; for another
cohort, explicitly register the representative-slide selection. There is no
silent first-slide selection, slide averaging or replacement of a thumbnail
by the mean of patch features. The demo's first-slide selection and independent
Gaussian thumbnails are explicitly synthetic only.

The original thumbnail preprocessing uses OpenSlide
`get_thumbnail((level0_width//30, level0_height//30))`, followed by the selected
FM's image transform and embedding. Register the actual thumbnail generation,
encoder preprocessing and weight provenance. Patch and thumbnail FM identity
and full dimension must match. The source uses UNI1024; changing the FM and
dimension is an explicit benchmark condition. The thumbnail encoder still
uses the original 512-hidden/1024-output projection. Wider patch features
are never truncated to1024.

The spatial graph uses the original HNSW L2 construction with eight neighbors,
or min(8,N-1) for small bags. It queries actual coordinates in their frozen row
order, without inferred square grids, fabricated edges between patients or
reordering of graph rows relative to features. A single HNSW build thread is
declared for reproducibility. The original unused latent graph is not built.

The reference population contains **TRAIN patients only**, in the supplied
original fold-list order. For each patient the original `randperm(N)[:1500]`
selects patches; a short bag is padded to1500 by its selected-row mean.
This is the original statistical reference tensor, not additional spatial
nodes. The original script seeds Torch98 once **before all five folds**;
later folds therefore inherit a previous-fold RNG stream. Preparation defaults
to an explicitly named fresh single-split seed98 protocol. To reproduce a
later original fold exactly, supply its saved starting CPU generator state
using `--sampling-rng-state`. Both generator states and all sampling indices
are frozen and validated. Reference tensors move with the model as constant
buffers; checkpoints bind their content and split, rather than duplicating
large input populations in every weight file.

## Native training and behavior

The five joint terms are full-graph Cox, causal-graph Cox, 0.1 times source
cosine contrastive loss, 0.1 times the sum of squared causal ratios divided by
the **nominal** batch size, and0.1 times MSE(z_new,z_old). Cox averages over
all patients including censored terms. Original Python clinical-time precision
is retained for risk-set comparisons. The source's `censorship` argument is
actually the observed-event indicator; the adapter uses event1 consistently.

Adam uses lr1e-4 and weight_decay5e-4; the learning rate halves before one-based
epochs41 and81. The default is100 epochs, no early stopping, batch16, no
gradient clipping. The source merges the last full TRAIN batch with the
remainder:33 patients become16+17;31 become31. Its ratio penalty still divides
by16. At least one full TRAIN batch is required, preserving the source's
division domain. Training mutates the same Python patient list with shuffle
each epoch; its current order is retained in the resumable checkpoint.

Validation loops over its declared patient order with no DataLoader seed draw.
Selection starts on the **third** epoch and saves the last tied maximum of
causal-risk lifelines C-index, with initial best0. There is no test-based
selection; the source's every-epoch TEST logging is omitted and TEST remains
sealed until explicit evaluation. Common benchmark and native lifelines
C-index are exported separately where their time-tie conventions differ.

Several nonstandard source behaviors are preserved explicitly:

- Both full and causal risks are sigmoid-bounded before being passed to Cox.
- The adaptive cluster constructor resets global NumPy/Torch seeds to988 after
  initializing its cluster logits; up to400 differentiable iterations run.
- TRAIN uses Bernoulli straight-through masks. VAL uses positive sigmoid values
  converted to bool for edge selection: all positive nodes enter the causal
  edge set while the remaining feature branch is still weighted by1-p.
- The original transformer query/einsum and untrained/unused parameter branches
  are retained; the method is not replaced by a generic PatchGCN.
- Dynamic centers and bias vectors come from the **last TRAIN forward before
  the last Adam update**. They are not recomputed after the update or for VAL.

Those centers, biases, selected K, optimizer, RNG and persistent patient order
must survive selected-weight reload and exact epoch resume. The source's final
weight-only overwrite drops its ordinary Python cache attributes. The adapter
explicitly includes them in state, avoiding an inference-only missing-cache
failure. Missing or malformed cache states fail rather than using test data
to reconstruct them. Cox's unconditional CUDA/global-device placement is
ported to the hazard tensor's actual device without changing its formula.

## Preparation and execution

Prepare a normal survival task bundle with patient mapping, fixed split, real
coordinates and FM provenance. The CSV contains exactly
`slide_id,thumbnail_path`, covering the selected WSI inventory. Each thumbnail
file is an NPY/PT feature vector of shape(D,) or(1,D). A JSON schema declares
`schema_version:1`, `kind:c2mil_graph_thumbnail_reference`, the synthetic marker,
selection/graph provenance, thumbnail origin/FM/preprocessing and exact ordered
global patient IDs for TRAIN/VAL/TEST. The synthetic demo exports a complete
schema example but its synthetic provenance must not be reused for real data.

```bash
python -m pip install '.[topology]'
milplus setup --method C2MIL
milplus prepare-c2mil --data /path/to/task --index /path/to/thumbnails.csv \
  --schema /path/to/c2mil_schema.json --output /path/to/new/c2mil_assets
```

Register the returned manifest under the chosen encoder in `task.yaml`:

```yaml
c2mil:
  manifest: /path/to/new/c2mil_assets/manifest.json
```

```bash
milplus run --data /path/to/task --method C2MIL --output /path/to/new/run
milplus evaluate --run /path/to/new/run
```

This profile requires its thumbnail and physical-graph assets for both training
and prediction. It does not silently fall back to generic features. CUDA,
natural WSI runs and original paper scores remain separate validation stages.
