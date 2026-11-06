# Attention2majority: original Keras two-stage reference integration

The implementation uses the original Keras ResNet50, the original binary
attention model, and the complete three-head model restored from the author's
commented class. This is an end-to-end image-training method: a folder of frozen
FM features is insufficient. Keep it in the encoder-training comparison track.

Development status: component and input-contract checks have passed. Complete
two-stage lifecycle, source-budget, independent-wheel and public-CLI admission
checks remain separate. No real WSI experiment or paper-score reproduction has
been completed. Pending profiles must not be advertised as accepted methods.

## Original source and explicit repairs

- Author repository: <https://github.com/JoeSu666/Attention2majority>.
- Fixed commit: `17c9614e2192ddf899a7a449081256bc9542b5d0`.
- Paper: <https://doi.org/10.1016/j.media.2022.102462> (2022).
- The author source has AGPL-3.0 terms. It is loaded privately from the pinned
  Git source; the package does not redistribute a copied author source tree.
- `discriminator.py` has a syntax indentation error and undefined training,
  validation and encoder variables. Orchestration is explicitly reconstructed.
- The source binary discriminator converts RGB pixels to signed `int8`, applies
  Caffe preprocessing, and then divides again by 255. Its instantiated Adam
  learning rate is 0.001; an unused schedule mentions 0.0002. These conditions
  are retained separately from the named uint8/Caffe, Adam0.0002 reference.
- The source negative-tail selector retains every negative patch when
  `round(0.3 * negative_count)` is zero (`[-0:]`). Only the explicitly named
  reference condition repairs that behavior.
- The complete commented three-head network is restored literally. Its three
  independent 1024→512 gated heads concatenate to 3072 dimensions before the
  three-class classifier. It is not a generic single-head attention substitute.
- Patient TRAIN/VAL/TEST boundaries are retained. TEST is never used for
  discriminator updates or MIL checkpoint selection. Incomplete author
  orchestration must not be interpreted as permission to leak TEST labels.

## Input task bundle

Use the common `slides.csv`, patient-disjoint `split.csv`, `images.csv` and
`task.yaml`. `images.csv` maps each `slide_id` to an HDF5 file containing:

- `pixels`: original uint8 `[N,224,224,3]` for binary Camelyon or
  `[N,128,128,3]` for the three-grade kidney condition;
- `patch_ids`: N unique UTF-8 identities in exactly the same row order.

Declare `encoder.dimension: 3`, and put the raw schema under
`encoder.attention2majority_raw`:

```yaml
schema_version: 1
kind: attention2majority_uint8_rgb_hdf5
mode: camelyon_binary
image_provenance: actual WSI reader, resolution and tissue-QC version
patch_provenance: original patch coordinate/identity registration
weights: /absolute/path/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
discriminator_train_pool: [normal_train_slide, reviewed_positive_train_slide]
discriminator_pool_evidence: pool_review.json
```

The required original Keras no-top ImageNet weights have MD5
`4d473c1dd8becc155b73f8504c6f6626`. Torchvision ResNet50 is not interchangeable
with this graph. Input checking retains original file SHA256, timestamps, shape,
ordered patch IDs, patient mapping, pool evidence and weight identity.

The binary pool evidence binds exactly the selected TRAIN slide IDs and explains
whether selection came from the author's published list, reviewed large-tumor
TRAIN slides, or the explicit Appendix-I ICA reconstruction. It declares
`heldout_labels_used: false` and a nonempty `review_provenance` field. Kidney
training uses all TRAIN slides and explicit three-class labels.

## Optional Appendix-I pool preparation

The paper does not release its ICA code, whitening convention or normal slide
identity. They must be disclosed rather than guessed. This command extracts
the actual original pretrained 1024D features from **every TRAIN patch**, fits
two-component ICA on the named TRAIN-positive template, uses the named
TRAIN-normal reference to label components, and records complete components,
per-slide tumor fractions, threshold0.1 and a maximum of12 positive slides.

```bash
milplus prepare-attention2majority \
  --data /data/raw_task \
  --template-slide tumor_110 \
  --normal-reference YOUR_REVIEWED_TRAIN_NORMAL \
  --component-space unmixing_rows \
  --output /data/prepared_attention2majority \
  --device cpu
```

The new bundle refers to the original raw files; it does not copy WSIs. This
stage does not update the ImageNet encoder. Task-specific CNN training happens
inside the subsequent complete method. ICA convergence failures and empty
eligible pools fail explicitly; no fabricated score/selection fallback exists.

## Native environment and full recipe

Use a separate TensorFlow2.15.1/Keras2.15/NumPy1.26.4 environment. Set
`MILPLUS_TF_PYTHON` to its Python executable. GPU validation uses the original
`tensorflow[and-cuda]==2.15.1` CUDA12.2/cuDNN8.9 packages, installed in its own
environment without changing the host driver or other methods' environments.
Requested unavailable CUDA devices fail explicitly.

The full source/reference conditions retain:

1. Actual ImageNet ResNet50 truncated at `conv4_block6_out`, GAP1024,
   Dropout0.25 and a learned binary/three-class classifier. Both backbone and
   classifier train on weak TRAIN patch labels.
2. Up to2000 randomly sampled patches per binary TRAIN slide or200 per kidney
   slide; finite class-balanced shuffled streams; native batch104 including
   the final incomplete batch.
3. Source discriminator Adam0.001 for150 epochs; explicit paper-reference
   Adam0.0002 with150 maximum and patience10 on TRAIN accuracy.
4. Actual task-trained inference for every original patch, stored1024D
   features and probabilities, original selected indices and patch IDs.
   Camelyon retains top30% positive and bottom30% negative instances; kidney
   uses the majority/confidence queue with K2000.
5. Original binary or three-head MIL network, class-weighted BCE or categorical
   CE plus actual source L2, bag batch1, Adam0.0002 with inverse-time decay,
   maximum150 epochs and TRAIN-accuracy patience10. The final completed epoch
   is used (`restore_best_weights=False`); VAL metrics are logged separately.
6. Full TensorFlow model, nontrainable BN state, Adam slots/iterations,
   deterministic per-update seeds, actual orders, both stage recovery,
   immutable input/runtime bindings and fresh selected-checkpoint evaluation.

Two-epoch software tests are labeled as such and do not establish completion
of this native150-epoch/stopping budget. Final method admission additionally
requires the complete source budget, public CLI and independent installed-wheel
checks on the same frozen implementation.
