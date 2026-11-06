# Iteratively coupled ICMIL

Official source is `Dootmaan/ICMIL`, main branch commit
`c2fd2faa4e85cf85527e228573b434a8cc461cba` (MIT,2023). This is separate
from both the source's later `confidence_based` extension and In-context ICMIL.
All three original base-MIL branches are explicit profiles: gated ABMIL,
MaxPool, and two-tier/five-group DTFD.

The complete reference loop starts with actual ImageNet
`resnet50-11ad3fa6.pth` (SHA256
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`),
uses the full original truncated ResNet50 (1024 output), extracts all registered
RGB256 patches, fits the MIL classifier, refines the encoder with the learned
classifier as a teacher, re-extracts every patch with the selected encoder,
and refits the MIL classifier. ABMIL starts a fresh head, while MaxPool and
DTFD warm-start from the previous selected MIL checkpoint, matching their
released step1 scripts. The student reduction/classifier trained during
distillation is retained as recovery evidence but does not silently replace
that previous bag-model checkpoint. More iterations repeat that entire loop.
The default reference explicitly declares one iteration; the source leaves
outer iteration control to manually rerunning its scripts.

The original encoder distillation is three terms:
patch prediction KL +0.5 classifier consistency KL +0.5 feature consistency KL.
It preserves `KLDivLoss(reduction='none').mean()`, not batchmean or a plain
supervised patch loss. Teacher encoder/reduction/classifier stay frozen; the
student encoder/reduction/classifier all train. Gradient clipping5 applies
only to student reduction/classifier, as in the source. Both randomized
flip/flip/RandomResizedCrop256 views are generated even though the third view
does not enter the loss. Original input conversion is ToTensor only, with no
ImageNet mean/std normalization. No image is silently resized or recolored.

MIL phases run200epochs, batch1, Adam/wd1e-4 and separate module clipping5.
ABMIL uses LR2e-4; MaxPool/DTFD use1e-4. The two-tier DTFD performs the first
optimizer step before the second backward, retaining the graph. Encoder
phases run200epochs, Adam1e-5/wd1e-4, batch100 and teacher-probability MAE
selection. Extraction batches448 images. The scripts' unused `epoch_step`
argument does not become a scheduler. Software fixtures use explicitly smaller
epoch budgets; they are not evidence that the full native budgets passed.

## Historical DTFD optimizer behavior

The original README specifically requires PyTorch1.4 for DTFD, with newer
versions for its other branches. Actual PyTorch1.4 author-loop AST runs
confirmed that its Adam mutates both `grad.data` for weight decay and `p.data`
for the parameter step. Replacing it by modern Adam changes retained gradient
state and introduces a parameter-version error at the second backward.

The DTFD condition includes the **unmodified original**
`torch/optim/adam.py`, SHA256
`97067d8382378053cbd13084c84f1da6dac7f584b7fbceff5f61b2070e776091`,
with its original BSD notice and a small host Optimizer bridge preserving the
old `zero_grad` default. The actual original loop in an isolated Python3.8 /
Torch1.4 CPU environment is the independent oracle. All four native-width
steps, gradients, learned weights and Adam moments matched the modern port
within the predefined cross-version tolerance (max observed error8.01e-8).
Same-runtime literal-source checks remain bitwise. No saved-tensor hooks,
deferred first step, or detached pseudo-bags replace this source behavior.

The source MaxPool uses output-size1 `AdaptiveMaxPool1d`. The actual H200 /
Torch2.5 CUDA backward rejects strict deterministic mode because the generic
2D pooling backend is marked nondeterministic. Its forward and tie indices
remain the literal source operation. The adapted CUDA backward writes each
channel's single selected gradient once, without atomic accumulation; CPU uses
the original operator. This explicit numerical implementation port preserves
the source K=1 derivative and does not disable deterministic checks.

## Explicit benchmark reference choices

These profiles are complete method adaptations, not a claim of historical
paper-number reproduction. The original scripts contain absolute paths,
manual iteration state, source-cohort MIL checkpoints, repeated TEST queries
and different patch/bag partitions. The reference makes these choices explicit:

- Bootstrap from the actual original ImageNet weights and train on the new
  task's TRAIN partition, not a Camelyon-trained classifier for a new cancer.
- Use the task's patient-disjoint TRAIN/VAL partitions for all phases. The
  released patch-distillation dataset instead shuffles all training-slide
  patches and splits them90/10, which is not the same patient protocol.
- Select the first finite eligible validation result, then strict improvement.
  The released MIL scripts start at hard-coded AUC0.1 or0.7 and can leave no
  checkpoint. Teacher MAE and all subsequent selection directions are retained.
- Retain the batch dimension for singleton validation tails; do not reproduce
  the released `squeeze(0)` failure. All actual patches remain present.
- Make no TEST query during model selection. Fresh TEST evaluation has a
  declared repeatable feature-permutation seed. Released test queries also
  consume RNG; removing them is an explicit protocol change, not a claim of
  identical historical training RNG.

Every training feature shuffle has a saved full row permutation. Checkpoints
retain teacher/student weights, BN buffers, all optimizer states, training
modes, mutable feature order, selection state and RNG. Immutable payloads are
checksummed before a single atomic generation pointer is committed, so a kill
cannot pair new weights with an old checksum. Completed phase state is restored
before the next phase is initialized. Original failed attempts are retained.

## Prepare and run

Fetch source and weights on a networked CPU host:

```bash
milplus setup --method icmil_coupled
milplus prepare-icmil-coupled-assets --output /path/to/assets
milplus pack-icmil-coupled --table original_patch_paths.csv --output /path/to/pixels
```

The tile CSV contains `slide_id,path`. Packing preserves literal PIL RGB256
pixels and sorts original patch paths within each slide. Provide audited
`slides.csv`, patient `split.csv` and `task.yaml`; the encoder dimension is3
and its index is the packed `pixels.csv`. Labels/partitions are never inferred
from filenames. The schema requires:

```yaml
schema_version: 1
kind: icmil_coupled_original_rgb256
layout: NHWC
dtype: uint8
color: RGB
patch_pixels: 256
normalization: torchvision_ToTensor_only_no_normalization
pixel_key: rgb
patch_ids_key: patch_ids
coordinate_frame: path_identity_only
image_provenance: your actual scanner, tiling and pixel provenance
patch_provenance: your actual ordered patch identity provenance
initial_weights: /path/to/assets/resnet50-11ad3fa6.pth
```

```bash
milplus prepare-icmil-coupled --data /path/to/task --schema rgb_schema.yaml \
  --output /path/to/task/icmil_coupled_manifest.json
```

The schema accepts YAML or JSON with the displayed fields.
Attach `icmil_coupled: {manifest: icmil_coupled_manifest.json}` to the selected
encoder in `task.yaml`, then the normal interface is:

```bash
milplus run --data /path/to/task --method 'ICMIL Coupled' --output /path/to/run
milplus evaluate --run /path/to/run --device cpu
```

Use `icmil_coupled_maxpool_patient_reference` or
`icmil_coupled_dtfd_legacy14_patient_reference` for the other two branches.
`configs/icmil_coupled.yaml` declares original complete phase budgets.
These methods train the encoder and need original RGB images; frozen TRIDENT
FM features alone cannot stand in for the full method. Current synthetic
software checks, actual GPU full-budget gates, independent installed-wheel
checks, new real-WSI experiments and full-library admission are separate states.
