# SI-MIL: complete dual-branch model and training recipe

The author source is [bmi-imaginelab/SI-MIL](https://github.com/bmi-imaginelab/SI-MIL),
fixed to `3ffb218672ffe0e4a183924af50e5920b498483d`.
The primary specification is [the CVPR 2024 paper and appendix](https://arxiv.org/html/2312.15010v2).
The source has no root redistribution license; PULS loads verified external Git
objects and does not bundle the author's implementation. Run `milplus setup
--method SI-MIL` on a connected preparation host; copy the pinned source cache
to an offline worker if needed.

## Inputs

Binary classification, one slide per training bag. Each slide needs deep FM
features **and actual PathExpert descriptors** for the exact same retained patch
rows and physical field of view. The paper uses nuclei-derived morphometric,
graph/social-network and spatial-heterogeneity features (HoVerNet, PanNuke,
ATHENA/FLocK pipelines). Deep features alone cannot supply this branch.
The adapter consumes raw, already filtered descriptors and performs normalization
inside the selected training split. It does not claim to implement nuclei
segmentation or descriptor extraction. Authors' already TRAIN+VAL-normalized
features must not be mislabeled as raw descriptors.

Prepare a CSV with exactly `slide_id,path` for every slide in the task. Paths
are relative to that CSV, or absolute. Each HDF5 has `features` (N × D, NaN
allowed for missing descriptors, infinity forbidden) and `coords` (N × 2).
Coordinates must equal the primary deep stream, including order and raw
precision. Duplicate coordinates, missing slides and shared descriptor files
are rejected. Native BRCA/CRC uses 246 descriptor columns; lung uses 203. A
different FM width is an explicit transfer setting and must keep the same FOV.

The schema JSON must contain:

```json
{
  "schema_version": 1,
  "kind": "raw_filtered_pathexpert",
  "synthetic": false,
  "feature_names": ["actual_ordered_descriptor_name_1", "actual_ordered_descriptor_name_2"],
  "field_of_view_um": 448.0,
  "segmentation_weights_sha256": "REPLACE_WITH_ACTUAL_64_LOWERCASE_HEX_HASH",
  "provenance": "actual source and raw extraction specification",
  "segmentation_model": "actual nuclei segmentation model and version",
  "feature_extraction_revision": "actual code revision",
  "patch_filtering_provenance": "actual filtering rule and revision",
  "aligned_field_of_view_provenance": "actual matched deep/PathExpert patch geometry"
}
```

The names above are placeholders, not a two-column replacement for the native
246 descriptors. Replace the entire list with the true ordered descriptor schema.
Register `field_of_view_um` in the selected encoder as well. Then run:

```bash
milplus prepare-simil --data /data/task --index /data/aux/index.csv \
  --schema /data/aux/schema.json --output /data/task/pathexpert.json
```

Add `simil: {manifest: pathexpert.json}` to `encoders.<FM>` in `task.yaml`.
The manifest binds descriptors, source commit, deep encoder, per-slide primary
features and their hashes. The normal data inventory includes every auxiliary
file, so changing them invalidates resume/evaluation signatures.

## Full model and recipe

Both profiles execute the original projector, gated deep attention and additive
deep classifier; hard top-K, the four-layer PF-Mixer and feature gate; and the
PathExpert additive prediction branch. Short bags repeat indices for the gate
but retain **all original rows** for prediction, matching the source. Per-patch
classifier bias is summed once per selected patch, never moved after pooling.

The objective is BCE(SI,y) + BCE(deep,y) + 20 MSE(SI,stopgrad(deep)) +
0.05 L2(patch attention). Adam uses betas (0.5,0.9), source patch dropout with
coupled row selection, persistent NumPy-shuffled bag order, and epoch cosine
schedule. The sample shell searches weight decay .005/.01 while the parser
defaults to .05; PULS explicitly defaults to .005 and records overrides.
Source `range(1,40)` executes 39 epochs; PULS defaults to 39 **actual** epochs
and T_max=40. No unrecorded clipping or generic bag sampling is applied.

Validation selection is mean(slide AUROC, accuracy using p>0.5), last tied best.
The dataset may additionally report patient-aggregated metrics, but selection
retains the author's slide definition. TEST is evaluated only on request after
training; the source's per-epoch TEST logging is not repeated. Both native
sigmoid probabilities (SI and deep) are exported.

## Explicit fixes and source route

- `simil_source`: literal published perturbed-top-K derivative.
- `simil_full_vjp` (alias `SI-MIL`): identical forward and random draws; the
  backward contracts the **full** Gaussian score-function VJP, including cross
  coordinate derivatives. The original contraction only keeps diagonal terms.
  A two-choice Gaussian closed-form derivative checks this independently;
  matching the original forward is insufficient proof of a correct derivative.

Both routes fit exact ten-quantile bins and missing-aware population statistics
using TRAIN patches only. This is a named leakage repair: published preprocessing
and training normalization include TRAIN+VAL. Missing and out-of-TRAIN-range
values retain the source -1 sentinel, without extrapolation/clipping. Constant
and entirely missing columns also remain -1. Exact column blocks avoid
concatenating the whole cohort's descriptor matrix; the configurable budget is
an estimated numeric workspace, not a process-RSS guarantee. HDF5 slices avoid
loading full descriptor columns for the quantile step. Per-slide alignment
verification still loads one complete slide at a time. No approximate quantiles
or sampled training patches are silently substituted.

Checkpoint state includes bins, moments, exact training order, optimizer,
scheduler, all RNG streams, counters and input contract. Interpretation HDF5
exports original coordinates, patch attention, gate-input/prediction indices,
conditional per-feature additive terms and the per-patch bias. Their sum
reconstructs the actual SI logit/probability. These conditional terms are not
causal effects.

```bash
milplus run --data /data/task --method SI-MIL --output /data/runs/simil \
  --config configs/simil.yaml
```

CPU synthetic validation is separate from
CUDA execution, natural WSI validation, and paper-score reproduction.
