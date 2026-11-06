# RMDL complete-method reconstruction: development, not yet admitted

The pinned original source is `emma-sjwang/RMDL` at
`8f13238285b92e772e04d2abe25cc24bb8099829`. No license was found in that
tree. Author files are loaded privately from the pinned Git objects and are not
bundled. The repository contains inference graphs, not the original training,
annotation sampling, hard-negative mining or NMS implementation. The private
WSGI-finetuned `Res_i_v2.h5`/`RMDL.h5` assets were not obtained. No random
encoder or generic frozen-FM MIL head substitutes for them.

The actual full ImageNet initialization is the official Keras InceptionResNetV2
no-top HDF5, SHA256
`937fb087127afa1d6d09f5eed7e560fc9ef20d3c6583b54653141c922a6dee0f`.
The released backbone `weights` argument does not load those values. The port
therefore loads the complete official no-top graph and checks every copied
weighted backbone layer. The FConv and task head are learned on TRAIN pixels.

## Preserved method

The actual author InceptionResNetV2, including all 10/20/10 residual blocks,
8x8 average pool, pre-ReLU1024 FConv and three-class CConv, is trained with
pixel annotations at0.5034micrometers/pixel. TRAIN470pixel crops undergo
0.9--1.1 scaling, arbitrary rotation and random flips before center299 cropping.
The source configuration's1.0068MPP differs from the paper; it is not silently
substituted for the explicitly selected paper physical scale.

The full FCN receives1899pixel blocks, emits51x51 probability maps, and moves
by1632pixels. Three spatial NMS passes use an area-overlap threshold0.68 and
retain100 actual299pixel regions per localization class. Repeated coordinates
across different classes are allowed by the three separate passes. The selected
299pixel images are separately encoded by the learned FConv. Contextual features
from the large FCN block are not substituted for those patch representations.

The original three-layer128/64/64 local/global recalibration graph privately
executes the actual `get_chowder` function body. That source class name does not
make this the CHOWDER method. Actual keras-contrib InstanceNormalization uses
per-channel bag standard deviation plus0.001. Dropout rates remain0.5/0.5/0.5/0.1.
The original per-bag all-element min/max scaling and instance permutation remain.
The source-SUM and explicitly named paper-MEAN pooling references are separate.

The aggregation defaults are7000 optimizer updates, batch256, Adam1e-3,
inverse-time decay0.001 and a0.8 learning-rate multiplier every140updates.
Dense-kernel L2 is0.005. Probability-clipped categorical crossentropy follows
Keras2.2.1; it is not replaced by a modern logits shortcut. A non-fused dense
Adam port preserves the original expression order: TF2.15 fused Adam changes
the rounding of its moments. Explicit dropout counters and stateless random
draws retain the rates/shapes while making cross-process recovery observable;
the historical TF1 random stream is not claimed.

## Explicit reconstruction choices

The missing localization total budget defaults to10000 declared reference
updates, with a single TRAIN-only hard-negative pass after5000updates. These
numbers are not presented as published paper totals. Localization uses full
batch80, Adam1e-4 and a0.9 multiplier every2000updates. Since80 is indivisible
by3, the27/27/26 class counts rotate over three updates. The current isolated
single-device implementation retains the global batch80; it is not a claim to
reproduce the paper's unpublished four-GPU batch-normalization distribution.

The sampler uses actual pixel labels at224-stride abnormal centers and a
reproducible finite pool of randomly drawn annotated normal centers. Center
label assignment and sampling with replacement are stated reference conventions.
Hard-negative mining adds all unique normal-annotated224-stride centers whose
actual current localization prediction is abnormal. The NMS overlap convention
is intersection divided by equal299-square area, not IoU. Ties are row-major.
No original unpublished implementation is claimed for any of these choices.
Final declared weights are selected; held-out labels never select a checkpoint.

## Data and isolated runtime

`task.yaml` uses classes `[normal,dysplasia,cancer]`, slide-level training and
evaluation, dimension1024 and an `encoder.rmdl_raw` mapping:

```yaml
schema_version: 1
kind: rmdl_original_scale_WSI_and_TRAIN_pixel_masks
records: raw_records.jsonl
imagenet_weights: /actual/official/inception_resnet_v2_keras_notop.h5
image_provenance: Actual scanner/scale/materialization provenance
annotation_provenance: Actual TRAIN pixel annotation source and annotator
```

Each JSONL row has exactly `slide_id`, `kind`, `working_mpp`, `annotation`,
and `provenance`. The feature-index `path` points to a complete OpenSlide WSI
or a spatial working-scale HDF5, not an array of frozen features. A WSI needs
actual native MPP metadata. A working-scale HDF5 contains `rgb[H,W,3]` uint8
and an attribute `working_mpp:0.5034`. TRAIN annotations are materialized
HDF5 `annotation[H,W]` uint8 with0normal/1dysplasia/2cancer/255ignored and the
same physical-scale attribute. They may share the TRAIN image container.
Validation/test annotation paths and held-out image-container masks are refused.
External links, soft aliases, virtual datasets and external raw storage are
refused. Actual bytes, patients, labels, splits, physical geometry and pretrained
weights are bound. Other MIL methods cannot treat this contract as frozen FM data.

Use an isolated TensorFlow2.15.1/Keras2.15 runtime with Pillow, OpenSlide Python
and OpenSlide native libraries. `MILPLUS_TF_PYTHON` selects its executable.
An optional `MILPLUS_RMDL_PIXEL_DEPENDENCIES` points to a separate dependency
prefix; the existing TF environment is not mutated. Original modern-runtime
differences and full runtime versions are recorded with each request.

`milplus demo-rmdl --output NEW_DIRECTORY --weights ACTUAL_OFFICIAL_H5` creates
explicitly synthetic nine1899pixel slides with TRAIN-only pixel masks for
software validation. No synthetic result is clinical performance evidence.
The two full execution profiles are available only in this development candidate:
`rmdl_source_sum_full_reference` and `rmdl_paper_mean_full_reference`. The alias
`RMDL` selects the explicitly named paper-MEAN reference. Public source/wheel
recovery and native-budget gates remain; this is not a stable release admission.

## Current verification boundaries

Actual full original localization graph/FConv/FCN1899 forward and full-model
gradient checks passed. Native256x300x1024 aggregation was interrupted in one
process and recovered in a fresh one for both SUM and MEAN: model variables,
Adam moments, dropout counters, predictions and loss history matched bitwise.
Literal old Adam arithmetic passed142updates including its learning-rate boundary.
These component checks do not establish completion of the full two-stage trainer,
source/wheel parity, full reference budgets, real WSI experiments or paper scores.

The initial full two-stage source software run completed localization, actual
TRAIN-only hard-negative mining, FCN/NMS, learned patch extraction, native-shape
aggregation and independent validation evaluation. It used declared2+2update
software budgets; it does not establish the default10000+7000reference budgets.

A strict public recovery check found only the serialized TrackableObjectGraph
differed: all1890 localization numeric tensors and58 aggregation numeric
tensors matched. Recovery had created Adam hypers/iteration/slots in a different
lazy-creation order and name scope. The recovery path now mirrors the actual
uninterrupted optimizer construction order/scope. Raw graph bytes remain part
of the strict comparison; no tensor, graph metadata or equality check is removed.
