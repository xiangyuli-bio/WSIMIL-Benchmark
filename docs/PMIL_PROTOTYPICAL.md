# Prototypical PMIL (Yu et al., MedIA 2023)

This is the full raw-image pipeline, separate from PMIL (Shapley pseudo-bags)
and ProtoMIL (prototypical parts). Source: https://github.com/zihou98/PMIL,
commit `f9b241a30ef90235da2c57814408da5eedb368e0`.

The source remains a separate checkout. Its LICENSE is GPLv3; its README also
states noncommercial academic use. This package does not redistribute author
files, learned assets or clinical data. Source acquisition and genuine weight
identity are mandatory; no random encoder, frozen-FM substitute or KMeans
fallback is supplied.

## Complete stages

1. Original full ResNet34 `[3,4,6,3]`, initialized with the actual
   `resnet34-333f7ec4.pth` ImageNet weights, then a newly initialized binary head.
   Inputs are real uint8 RGB256 patches. Preprocessing is exactly ToTensor,
   Normalize(mean=.5, std=.1) in each channel, with no implicit resize.
2. Repeated full-TRAIN patch scoring, positive-class top1 per slide, Python
   sample and Torch loader shuffle, task-trained CNN CE, original Adam
   1e-4/wd1e-4, train batch128 and inference batch256. Full budget is 30 epochs,
   scheduler milestones10/20 with gamma.1, VAL every5 epochs, patience3.
3. Selected encoder re-extracts every TRAIN and VAL patch into actual512D
   features. Patch identity/order and each feature file hash are retained.
4. Literal two-level affinity propagation: all patches within each TRAIN
   slide, followed by all local exemplars across TRAIN. Parameters are
   preference0, damping.5 and random_state24. Source similarity is
   `exp(-EuclideanDistance / maxDistance * .25)`. No candidate cap is hidden.
   Undefined distance normalization or failure to converge is reported.
5. Original selected-patch head, top5 by positive probability, shared512->128
   projection, source pairwise Euclidean distance, row division by its maximum,
   mean code, and 128-wide classifier. Despite the source's "similarity"
   name, its distance direction is retained. Both bag CE and max-instance CE
   train together. The original trainer uses eval mode during fitting; it is
   retained, as are Adam1e-4/wd1e-4, milestones10/20, 30 epochs, and latest
   validation-accuracy tie selection. Its registered but unused CNN parameters
   are retained and receive no fabricated gradients.

## Explicit source repairs and boundaries

The native `max-mil.py` has an IndentationError, feature-export row/name mistakes,
an incorrect final inference tuple, and a validation path that calls its
optimizer-updating training function on VAL patients. Its early-stop checkpoint
is also overwritten unconditionally. The model import `from model import
resnet34` points at an otherwise empty package.

This named **patient reference** binds the source's own ResNet definition,
uses no-grad validation, keeps training/validation/test patient partitions
disjoint, selects by validation patch CE, and consistently exports features
from that selected encoder. Final-epoch validation supports explicitly shortened
software checks; it does not alter the original30-epoch divisible-by5 schedule.
TEST pixels and labels are excluded from fitting, feature discovery and model
selection. They enter only explicit evaluation. No historical score reproduction
or identity with the unsafe public main() is claimed.

Both phases checkpoint complete CNN/head weights, BN buffers, Adam moments,
scheduler, selected instances/order, all RNG and full history. Atomic immutable
checkpoint generations retain the last healthy pointer on interruption. Cache
contracts bind patient/slide/patch identities, source, actual learned encoder
and input hashes. Cache reuse restores its precise post-stage RNG.

## Usage

On a networked CPU host:

```bash
milplus setup --method 'PMIL Prototypical'
milplus prepare-pmil-prototypical-assets --output /owned/assets/pmil
```

Prepare a normal audited binary slide task bundle with patient-grouped splits.
Its `raw_rgb` encoder has dimension3 and an index of HDF5 pixel bags, not FM
features. HDF5 contains embedded `rgb: uint8[N,256,256,3]` and unique ordered
`patch_ids: string[N]`, with at least5 actual patches per slide. Optional
measured coordinates must have an explicit aligned coordinate frame.

`pack-pmil-prototypical --table tiles.csv --output /owned/packed` can pack
actual RGB256 patch paths using the shared literal PIL reader. The tool never
infers patients, labels or splits.

The preparation schema declares kind `pmil_prototypical_original_rgb256`,
schema_version1, layout NHWC, dtype uint8, color RGB, patch_pixels256,
normalization `torchvision_ToTensor_then_mean0.5_std0.1_all_channels`, keys and
provenance for pixels/patch IDs, coordinate_frame, and the original
`initial_weights` path. Use `prepare-pmil-prototypical --data TASK --schema
schema.yaml --output manifest.json`, then bind its path under
`encoders.raw_rgb.pmil_prototypical.manifest`.

```bash
milplus run --data /owned/task --method 'PMIL Prototypical' \
  --config configs/pmil_prototypical.yaml --device cuda:0 --output /owned/run
milplus evaluate --run /owned/run --device cuda:0
```

`demo-pmil-prototypical` generates explicitly synthetic software bags. Reduced
epoch software tests are not original-budget GPU gates. The latter use the
full30/30 budgets, native widths and batch settings, real interruptions,
continuous/resumed state comparison and two fresh selected-checkpoint reloads,
both from source and an independently installed wheel. All-library release
acceptance and real BRCA/COAD WSI experiments are separate requirements.
