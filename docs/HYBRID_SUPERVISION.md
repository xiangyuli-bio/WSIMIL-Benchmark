# Hybrid Supervision Learning

This candidate implements the full paper-mechanism reference pipeline. It is
under verification and is not a claim to reproduce the unpublished original
gastric trainer, clinical results, or the released CAMELYON17 challenge recipe.

## Source and the two distinct protocols

The original repository is
https://github.com/JarveeLee/HybridSupervisionLearning_Pathology at commit
`1f54478812cd057ad3ddc70543bed503ba39f800`. The paper is
https://arxiv.org/abs/2107.00934. Author code is loaded from a separately
obtained, commit-verified checkout. It is not silently replaced by a generic
attention head or bundled into this distribution.

The paper-reference profile uses the complete ImageNet-initialized DLA34Up
segmentation network and DLA34 classifier. Inputs are actual 512 × 512 RGB
patches at 0.5 micrometres/pixel, with 128-pixel overlap (384-pixel stride).
Observed pixel masks are required for TRAIN patients. Image-only FM features
cannot replace the pixel and annotation inputs.

Each round trains pixel segmentation for 30 epochs, scores every TRAIN patch,
retains patches with maximum probability at least 0.4, and trains an image
classifier for 15 epochs using the Top-16 actual patches. Image probability
is the mean of patch sigmoid probabilities; its BCE is not the BCE of a mean
logit. Initialization is followed by two EM rounds. The segmentation network
starts each subsequent round from the previous learned segmentation weights.
The selected round minimizes its final-epoch TRAIN image-classification loss.

Pseudo pixel probabilities are multiplied by four, clipped at one, and values
below 0.01 are zeroed. Mixed segmentation batches contain two fine-annotated,
one positive pseudo-labelled, and seven hard-negative patches. A missing
retained pseudo pool fails explicitly; no patches or labels are invented.

The released challenge code instead contains a 1024-pixel/768-crop route,
PNG-quantized pseudo labels, a different inclusive-integer sampling rule,
three-epoch phases and a heatmap/morphological-feature/Random-Forest patient
staging route. The corresponding original equations and patient rules have
component oracles. That RF route is not labelled as the paper CNN pipeline,
and no full RF-profile admission is claimed here.

## Disclosed reference choices

The public materials do not fully specify the original private trainer.
The following decisions are explicit reconstruction choices:

- Adam at 0.001 with default betas and zero weight decay, taken from the
  released segmentation optimizer, for both trainable stages.
- Segmentation batch size ten; an epoch has
  `ceil(number of observed TRAIN fine patches / 10)` updates.
- Two balanced image bags per image-classification update, with equally
  weighted image losses; an epoch contains the larger class's bag count.
- A fresh ImageNet-initialized image classifier in each round.
- Joint 90-degree rotations/flips for pixels and masks, followed by the
  released RGB jitter of 0.4. No resize that changes physical field of view.
- Stable patch-index order resolves tied scores. No TEST or VAL labels
  participate in pseudo labels, fitting, or round selection.

The method protocol and request record these choices. CPU and CUDA are
explicit devices; an unavailable device does not trigger a fallback.

## Input and execution

The task bundle contains patient-disjoint slide, split and raw-pixel indices.
Every HDF5 slide has uint8 NHWC RGB, original ordered patch identifiers,
integer coordinates, and TRAIN-only binary observed fine masks with their
patch indices. The schema records physical resolution, stride, provenance,
dataset keys and the original pretrained initializer path. Preparation binds
each asset's bytes and rejects annotation or pixel changes.

```bash
milplus prepare-hybrid-supervision --data TASK --schema SCHEMA.yaml --output MANIFEST.json
milplus run --data TASK --method 'Hybrid Supervision Learning' --output NEW_RUN --device cuda:0
milplus evaluate --run NEW_RUN --device cuda:0
```

Use the `hybrid_manifest` field demonstrated by
`milplus demo-hybrid-supervision --output DEMO --weights DLA_WEIGHTS` when
binding a prepared manifest in `task.yaml`. The demo contains synthetic pixel
fixtures, not pathology patients.

The original `dla34-ba72cf86.pth` initializer is validated against its complete
SHA384 and 63,228,658-byte size published by Intel Open Model Zoo, in addition
to SHA256. Its SHA256 is
`ba72cf86426e6333d9e8c6c7a8cae5549879212e48ae3aaad4bc8a7d009c34e1`.
Weights are not downloaded during GPU training or included in the wheel.

## Recovery and evidence

Immutable, checksummed checkpoints retain all model parameters and BN buffers,
gradients, Adam state, RNG state, ordered history and phase progress. Pseudo
probability caches are bound to input bytes and learned segmentation weights.
Completed phases are reused with their original state on `--resume`; changed
software, inputs, device, budget or masks are rejected. Held-out evaluation
requires completed training and verifies the selected pair of checkpoints.

Component checks compare the complete original DLA34Up forward, loss,
gradients and Adam step against the literal released training body. Public
CLI checks cover three actual interruptions and two independent selected
reloads. Full default-budget source and independent-wheel native runs,
whole-library checks and release acceptance remain separate requirements.
Short software-check budgets do not stand in for the 30/15/2 native recipe.
