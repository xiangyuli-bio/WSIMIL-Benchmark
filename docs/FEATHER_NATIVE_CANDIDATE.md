# FEATHER24K native adapter candidate

FEATHER now exposes nine original-pretrained code profiles, marked
`adapted_under_verification`. See [public usage](LAST_THREE_METHODS.md).
Registration does not establish actual pretrained or full-budget validation.

The original MIL-Lab source is pinned to
`dfc3c17ca6876a8f9bd33a9162c40f2c22631d9d` and loaded privately from verified
Git objects. Author source and weights are not redistributed in this package.
The complete released base network has a 512-dimensional patch projection,
384-dimensional gated-attention branches, branch dropout 0.25 and the original
classifier initialization/reset. The native CONCHv1.5, UNI and UNIv2 inputs
have dimensions 768, 1024 and 1536 respectively.

The asset interface checks user-supplied local checkpoint bytes against the
public Git/LFS identities for each original release. It performs no restricted
download or access-condition acceptance:

```bash
milplus prepare-feather-assets \
  --encoder uni \
  --checkpoint /absolute/path/to/original/model.safetensors \
  --output /absolute/path/to/feather_uni_assets.json
```

The candidate full-model builder uses the actual original Hugging Face wrapper
and local-checkpoint loading helper. It requires every non-classifier tensor,
compares all loaded backbone values with the original checkpoint, and preserves
the source's downstream classifier reset. This positive loading path has not
been executed because the original checkpoints are unavailable.

The public 24K release and the earlier manuscript-scale experiments are distinct
assets/protocols. Rounded release labels do not establish exact cohort counts.
Standalone CONCHv1.5 and the CONCHv1.5 file distributed with TITAN have different
published file hashes; numerical feature equivalence has not been established.
Native feature provenance must therefore be bound explicitly before training.

The paper's downstream recipe uses AdamW, cosine scheduling, mixed precision,
class-weighted sampling and cross-entropy. It specifies 20 maximum epochs,
10 minimum epochs and patience five with validation, plus feature dropout 0.1.
The released base network has no dropout after its single patch projection,
whereas the paper describes dropout at feedforward layers. The candidate makes
selection, sampling, AMP dtype and scheduler conventions explicit reference
choices because the released repository contains no complete trainer.
See the [original paper](https://arxiv.org/html/2506.09022v1#A2.SS2) and
[author library](https://github.com/mahmoodlab/MIL-Lab).

Completed checks exercise the full original architecture and gradients in all
three native dimensions, unchanged backbone tensors after classifier reset,
original-source identity, and rejection of missing or unrelated checkpoint
files. These are component/input checks. They do not establish original
pretrained loading, complete downstream training, source/wheel equivalence,
native-budget acceptance, real-WSI results or paper-score reproduction.

## Native feature and patient binding

After `prepare-feather-assets` verifies your original local checkpoint, write
an extraction metadata JSON with `schema_version: 1`, `encoder_id` equal to
`conch_v15`, `uni`, or `uni_v2`, the native `feature_dim`, original visual
`model_id`, pinned `revision`, `weights_sha256`, and a nonempty
`extraction_provenance`. The visual weight identities are in
`src/milplus/resources/feather24k.lock.json`. These are explicit current
reference revisions; the paper did not identify those exact visual revisions.
The manifest records the producer's provenance assertion, not independently
observed extraction. Equal-dimensional features from another encoder are
rejected, including the different CONCHv1.5 file shipped with TITAN.

The input schema is:

```yaml
schema_version: 1
pretrained_assets: feather_uni_assets.json
visual_extraction_manifest: uni_extraction.json
provenance: Describe the actual extraction and task curation here
```

```bash
milplus prepare-feather --data /absolute/path/to/task \
  --schema /absolute/path/to/input_schema.yaml \
  --output /absolute/path/to/feather_inputs.json
```

Then add `feather: {manifest: /absolute/path/to/feather_inputs.json}` under
the matching encoder in `task.yaml`. The binding hashes every unchanged full
bag, labels, global patient memberships and extraction/checkpoint manifests.
It rejects repeated patient identities or byte-identical bags across splits.
The complete task bundle remains TRAIN/VAL/TEST. In the no-validation training
reference, VAL is held out and is not consumed by optimization or selection.

## Downstream candidate conditions

The nine public code profile IDs combine each native encoder with
`finetune_val20`, `finetune_no_val10`, or `frozen_knn20`, for example
`feather_uni_finetune_val20_reference`. The common engine and isolated runtime
use these profiles. The family name selects the prepared encoder and the
explicit with-validation finetuning reference. Actual original-pretrained
loading and complete execution gates remain separate and pending.

The complete finetuning reference retains the original model, full bags,
feature dropout and paper optimizer/budget constants. All model parameters
are updated; only the original source downstream classifier is reset.
Explicit additional choices are inverse TRAIN-slide-class frequencies with
replacement and N draws per epoch; a separate seeded sampler; AdamW default
betas/epsilon; end-of-epoch cosine steps; strict minimum VAL cross-entropy
selection; and consecutive nonimprovements counting from epoch one with no
stop before ten epochs. The CUDA condition uses native FP16 autocast and
GradScaler. CPU FP32 is separately recorded. The source patch projection
retains its actual absence of dropout. Neither condition is claimed to recover
the author's unprovided trainer.

Checkpoint recovery preserves the entire graph, Adam moments, scheduler,
GradScaler, sampler state and current draw order/cursor, gradients, Python,
NumPy, Torch and CUDA random states, history and selected epoch. The interruption
boundary changes no declared native budget. No-validation training uses the
last checkpoint; validation training uses its best validation checkpoint.

Frozen evaluation uses the actual 512-dimensional preclassifier embedding
and only TRAIN slides as the KNN bank. All pretrained tensors stay unchanged.
K remains 20; fewer TRAIN slides fail rather than silently reducing K. Because
the paper does not give a distance/voting implementation, the explicit
reference uses float64 squared Euclidean distance, no normalization, uniform
votes, stable TRAIN slide-ID tie order, and lowest-index class vote ties.
VAL and TEST cannot enter the bank. Patient-level metrics use the common
bundle's declared evaluation rule after slide predictions.

The source dependency requirements fix Transformers 4.53.1 and NumPy <2.
The isolated reference additionally pins tokenizers0.21.4, hub0.30.2,
safetensors0.5.3 and loguru0.7.3; Torch2.5.1 is a declared common-runtime port.
Original weights are neither downloaded nor access conditions accepted by
this adapter. Set `MILPLUS_FEATHER_PYTHON` and, for a separate dependency
prefix, `MILPLUS_FEATHER_DEPENDENCIES`; the subprocess rejects mismatched
runtime versions or different adapter code.

Downstream control checks cover the full native architecture, ten-epoch
literal-loop parity in all three input dimensions, twenty-epoch validation
selection, ten-epoch minimum early stopping, weighted-order recovery at six
different batch boundaries, KNN ties/votes, and native feature-space/patient
rejection. These use explicitly unpretrained architecture components and
synthetic arrays, never a substitute FEATHER model.

The0.114 candidate additionally binds the actual numerical environment and
effective backend flags. Runtime identity includes the Torch build, CPU ISA,
thread counts, MKL/OMP/CUBLAS settings and precision/determinism settings;
schema2 checkpoints record effective post-initialization flags. Restoration
rejects a changed backend before loading model parameters. Old candidate
checkpoints remain associated with their original frozen code.

An initial default-CPU cross-process mismatch remains preserved and unresolved.
Four fresh allocator layouts subsequently produced exact full checkpoints
within each of the default and MKL-COMPATIBLE branches; those two branches
produce different numerical trajectories. Instrumented/default successes and
the four-condition COMPATIBLE source/wheel checks do not establish the initial
mismatch's cause or turn it into a passed full FEATHER gate. The added backend
binding prevents a proven different numerical condition from being labelled
the same runtime merely because its package versions match.

## BRCA original UNI asset access recheck (2026-11-06)

At2026-11-05T22:28Z, bounded official HEAD requests for the pinned UNI
`model.safetensors` returned401/GatedRepo anonymously and403/GatedRepo with
the existing locally cached authentication. Redirects were not followed and
no authentication material was logged. No access request, term acceptance,
restricted-file download or random-weight replacement was performed.
This newer observation does not overwrite the historical2026-10-26 access
report or imply that either other encoder variant was newly checked.

The required UNI checkpoint remains3677668bytes with SHA256
`19b7edb4582ea534f63c74b8b83735c0529388d1b8f26945786cce9d0b4aea19`.
The current input interface separately requires actual native visual extraction
provenance. The historical BRCA UNI weight/revision is not known; the explicit
current reference in the lock must not be asserted as that historical identity.
No training or positive original-pretrained loading gate was completed.
Shared evidence:`feather_uni_official_access_recheck_r1.json`.
