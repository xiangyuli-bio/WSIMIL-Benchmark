# ProtoMIL (MICCAI2025 concept version), component work

Original source `ss-sun/ProtoMIL`, commit
`3e3745d7e4fecccb24aa486bd2af8b0dc6493711`; this is the concept/SAE method,
separate from the earlier prototypical-parts method with the same short name.
No LICENSE file is present in the pinned author tree: private verified source
loading, no third-party source or weight redistribution.

The author references Discover-then-Name and sparse_autoencoder0.0.0 without a
Git revision. Reviewed dependency:
`neuroexplicit-saar/Discover-then-Name` at
`03484baef77747596712cf971e402f77b9da5a9a`; its actual modified sparse_autoencoder
subdirectory supplies the resample_epoch_freq interface and has an MIT license.
This is an explicitly reviewed dependency, not a historical dependency-pin claim.
Unused root aggregation of NLP datasets and sweep automation is isolated; actual
SAE modules, optimizer, loss, resampler, metrics and ProtoMIL pipeline are retained.
Author-declared numerical dependencies are isolated: NumPy1.26.3,
Pydantic2.5.3/core2.14.6, jaxtyping0.2.25, einops0.7.0 and protobuf4.25.1.

Complete method requirements, still pending implementation/acceptance:

- Actual normalized512D CONCH features, explicit encoder/checkpoint identity and
  ordered patient-disjoint task data; TRAIN-only SAE fit population and separate
  pretraining validation. Original preprocessing points to complete Camelyon/PANDA
  folders; any change to exposure must be documented, not silently called original.
- Actual512->2048 tied-bias SAE with ReLU encoder, unit-norm decoder, its gradient
  projection hook and post-step normalization; source AdamWithReset, reconstruction
  plus3e-4 L1 activation loss. Original script uses200epochs, paper states60:
  these must be separately named. Native batch4096; incomplete tail updates and
  original reported activation-counter rounding must remain explicit.
- Original piece ordering, TRAIN/VAL chunk split and actual resampler/dead-neuron
  optimizer-state resets, all counters/metrics/RNG in checkpoint. Resampling uses
  source500000piece-epoch guard and may be skipped under its literal condition;
  do not silently replace it with an activation-count schedule.
- TRAIN probe set with10000normal+10000tumor patches and original mask/patch
  provenance; actual top10 active patches per neuron and inspectable artifacts.
  Pathologist concept names and removal choices are user-supplied annotations,
  never fabricated. Hardcoded author neuron IDs are tied to a specific SAE and
  must not be blindly applied to a newly fitted SAE.
- Actual concept activation cache and frozen SAE identity, original2048-concept
  gated attention and linear explanation, CE plus **mean** absolute classifier
  weights, Adam1e-4 and explicit selection/last-model semantics. Original MIL
  CLI defaults one epoch, comments say200; specify budget explicitly. It calls
  TEST reporting each epoch; benchmark TEST is deferred and never selects models.
- Fresh public CLI, complete both-stage interruptions and exact restore, installed
  wheel, original-budget CPU/GPU and global/release gates.

The first component run stopped on a canonical-source-root check across symlinks;
resolved roots corrected that port. A later oracle used deepcopy for the reference
SAE, which drops PyTorch Parameter backward hooks. A separate diagnostic proved
hook counts1/0/1 for original/deepcopy/fresh-constructor+state-load, and exact original
versus reconstructed gradients. The reference now invokes the full original
constructor and loads state. These are adapter/test issues, not evidence against
author experimental results. Failed logs remain preserved. No runnable profile or
method-count increase is claimed for this component stage.
