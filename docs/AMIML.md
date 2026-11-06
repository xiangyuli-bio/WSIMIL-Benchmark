# AMIML: complete four-cluster reference implementation

This candidate is not a formal release. It adds the AMIML record from the
164-entry admission plan; it does not replace it with ABMIL or an ordinary MLP.
All clinical endpoints remain ordinary patient-level labels, not model inputs.

## Reviewed primary sources

- Author code: <https://github.com/Boomwwe/AMIML>, fixed commit
  `7c038261d1a52a4eb20d4b4f444e730c36bf5cd2`, GPL-3.0.
- Author preprint: <https://arxiv.org/abs/2206.00455>, v1, 34 pages;
  methods on PDF pages 4–6, architecture on page 18, caption on page 15.
- Journal identity: DOI `10.1016/j.compmedimag.2023.102189`.
  The accessible preprint is the recipe source; a journal-version difference
  must not be silently treated as checked.

The paper's feature extractor is a fine-tuned Xception outputting 256 channels.
It does **not** describe PCA to 256. Arbitrary full FM widths are an explicitly
named first-convolution input extension, not a claim to reproduce those features.

## Actual public implementation defects

`AMIML/model.py` uses `super(HE2RNA, self)` although that symbol is undefined.
Its `input_dim` argument does not control the first convolution. After repairing
the constructor alone it outputs eight probabilities; the training script uses
two class weights in cross entropy. Those contracts cannot run together.

The original min/max gathering duplicates the index tensor to eight channels.
Changing the latent width to two would break that mechanism and would change
the paper's eight-channel residual representation. We preserve the native
128/64/32/8 encoder, one-channel scorer, 1→8 transposed convolution, alpha=0.1
residual, Q/K width8, R=100, and actual dropout placement. BN4/BN5 are retained
as unused source modules, not newly activated layers.

The paper describes aggregating two-dimensional attended instance outputs but
does not supply a working binary output mapping. The named reference repairs
V from 8→8 to **8→C** and leaves Q/K and the entire eight-channel representation
unchanged. This is a declared reconstruction of the missing output, not evidence
that an unpublished implementation used this exact head.

`fit_model.py` imports missing `AMIML_model` and `dataloader_function` modules,
instantiates undefined `HE2RNA`, and depends on globals not defined in its module.
The launcher also computes class weights from the entire cohort. Its loader,
short-bag sampling, clustering setup, batch-shape behavior and exact original
feature preprocessing are not recoverable from this checkout. We do not invent
hidden padding/sampling rules or ship a fake original dataloader.

## Complete input and training pipeline

The task bundle must declare `training_unit: patient` and
`evaluation_unit: patient`. All slides of one global patient are concatenated in
registered order. Train/val/test patients must remain disjoint under the shared
bundle checks.

Four morphological clusters are fitted with complete **TRAIN features only**.
The fixed implementation is sklearn Lloyd KMeans, n_init=10, max_iter=300,
cluster_seed=2026; these unreported choices are explicitly recorded. Memory
budgets fail before the full fitting allocation; no subset or MiniBatchKMeans is
silently substituted. Frozen centers assign VAL/TEST patches without refitting.
Class weights use TRAIN patient counts only. Binary weights match the source
fractions; C-way inverse-count weights are a named extension.

Each patient must contain >10 and at least R actual patches in every cluster.
Missing/short clusters fail before updates. The pipeline never repeats rows,
silently removes patients or uses made-up coordinates to create a viable bag.
For ragged minibatches, convolution and BN see only actual patch cells, and
min/max selection is performed independently within each patient. Equal-length
minibatches use the exact author layout. Ragged BN is a documented extension
because the author loader is absent.

All **four** complete AMIML models train independently. Each has its own Adam,
ReduceLROnPlateau, stopping state, validation-loss checkpoint and training order.
There is no preliminary cluster choice using TEST. All source encoder/decoder,
ranking, residual and attention parameters participate in their actual paths.
The launcher resets the same seed for each cluster. The adapter preserves
identical initial weights and four independently saved Python/NumPy/Torch/CUDA
RNG streams. Interleaving the four trainers does not couple their sampling or
dropout, and continuation restores every branch stream.

Two profiles distinguish conflicting recipes:

| Profile | Default LR | CE input | Plateau patience | Other defaults |
|---|---:|---|---:|---|
| `amiml_paper_reference` | 0.01 | pre-SoftMax logits, explicit corrected CE reference | 10 | 300 max epochs, batch64, Adam, no decay/clip, R100, alpha0.1 |
| `amiml_source_repaired` | 0.001, source launcher | SoftMax probabilities, source behavior retained | 5 | same complete four-cluster training; binary only |

Early stopping follows the public helper: default patience30, equality resets
the counter. Per-cluster selection starts at zero-based epoch5 and retains the
last equal minimum VAL weighted mean-batch CE. The original arbitrary loss
sentinel10 is replaced by an explicit uninitialized state. AUC is computed over
all validation patients, avoiding undefined per-minibatch AUC in the source.
The cluster is chosen using those four checkpoints' **VAL** macro-AUC, with a
fixed lowest-index tie. The paper does not clearly disclose whether its best
cluster was picked on held-out tests; this implementation explicitly forbids it.

`best.pt` is the latest container of all four independently VAL-selected models,
centers, optimization/RNG/scheduler/stopping/order state and the chosen cluster.
`amiml_validation_generation` is a monotonically increasing checkpoint-generation
index, **not a performance metric**. This avoids accidentally keeping an older
cluster state based on a pooled or maximum-ever AUC. Full prediction metrics and
per-cluster best epochs/loss/AUC remain recorded separately.

## Use

```bash
milplus setup --method AMIML
milplus demo --amiml --patients 20 --output /path/to/new_synthetic_bundle
milplus run --data /path/to/prepared_patient_task --method AMIML --output /path/to/new_run
milplus evaluate --run /path/to/new_run
```

The original Xception, raw WSI/segmentation/normalization, clinical labels and
original TCGA splits must be independently obtained for numerical reproduction.
The method interface accepts prepared full FM features; no gene expression or
mutation assay is added as an inference input. Synthetic tests prove software
properties only. Full default training budgets, fresh whole-library checks,
all-profile runs and independent final-wheel reload remain admission gates.

The current complete candidate passed all27 focused checks, including raw author fullR100/allgradients on256/1024/1536, actual four-branch training, independent RNG streams, original equal-loss early-stopper/plateau controls, C-way patient evaluation, and both eight-epoch bitwise continuations. Full2540-suite/223profiles, four default300max installed-wheel budgets and final packaging remain pending.
