# MIL with Center Embeddings (MICCAI 2020)

The full author source is `PhilipChicco/MICCAI2020mil`, fixed at
`871b1ecad3f0480c526536b11fafe4d759273a25` (GPL-3.0, fetched privately).
This is an end-to-end original-image method. Cached FM embeddings do not
provide the internal spatial feature maps needed by its source bag network.

The two registered routes are:

| Execution ID | Selected inference |
| --- | --- |
| `center_embeddings_bag_classifier_repaired` | Original default bag classifier |
| `center_embeddings_soft_assignment_repaired` | Author-documented Student-t assignment to learned centers |

`Center Embeddings`, `MIL with Center Embeddings`, and `ChikonMIL` select the
default bag classifier. Both retain the same complete source training process.

## Full model and source repair

The native model is the actual author's ResNet34v2, initialized with the
original ImageNet `resnet34-333f7ec4.pth` (SHA256
`333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef`).
It retains the p2/p3/p4 spatial pyramid, 128-channel intermediate maps,
32-channel bag CNN, shared512D embedder, instance classifier, bag classifier and
two learned512D class centers. There are21,917,897 total trainable parameters.
No backbone, FPN, classifier or center objective is omitted.

The original trainer receives `bag_embed` from the network but invokes center
loss on undefined `bag_feat`. The named repair passes that exact normalized
third network output (`bag_embed`). The three source losses remain
`instance CE + center loss + bag CE`, with coefficients1/1/1. The center-loss
device constant follows the actual center tensor when moving to GPU.

The source is binary, with fixed K32 and256×256 original RGB input. Center count
and inference are hardcoded for two classes; this implementation does not
silently substitute a new multiclass method. Its bag CNN treats the32 selected
patch feature maps as channels; it does not construct a fake physical WSI grid.

## Original data and sampling

Every epoch first scores **all TRAIN candidate images** with the instance
classifier in evaluation mode. That scan uses the author's TRAIN preprocessing,
including RandomHorizontalFlip. It then selects the original positive-score
top32 for each slide, retaining ascending score/tie order from
`group_argtopk`. This order matters to the bag CNN. Bag order is shuffled with
the source Python `random.sample`. The selected images are transformed afresh
for joint training. VAL/TEST separately scan all their candidates, select their
own top32, and do not use TRAIN augmentation.

Original torchvision0.7 code supplies Resize256, TRAIN horizontal flip.5,
ToTensor and Normalize(mean.5/std.1). Genuine raw256 RGB and measured level0
center coordinates are bound to ordered patch IDs and immutable pixel hashes.
The default reader uses the exact source `MILdataset.norm_coord`, OpenSlide
level2, multiplier1, and source nearest resizing only when multiplier differs
from1. An exact cache of those reads can be prepared once and reused; the
network is trained from actual pixels on every run.

The public loader silently removes TRAIN/VAL slides with fewer than32 patches.
Here an incompatible registered cohort is rejected, allowing explicit cohort
preparation instead of silently changing the evaluation population. TEST bags
can be shorter: the original model pads missing **feature-map channels**, not
RGB images, after running the real backbone. Every registered slide is retained.

The author's local config takes the first10slides/class from supplied libraries.
The unified task bundle explicitly registers the chosen cohort and patient
splits; it does not impose an unreported first10 truncation on a larger dataset.
The source-default native checks use20TRAIN slides (10/class),20VAL and20TEST.

## Complete recipe and recovery

The default is20epochs, TRAIN bag batch4, VAL/TEST bag batch1, with no generic
patch cap, clipping or early stopping. Worker0 is an explicit recoverable I/O
port of the source worker6 loader, preserving ordered image/augmentation draws.
Software tests can explicitly set bag batch1; the independent native gate
requires the original default batch4, all32 images per selected bag and256RGB.

Two genuine Adam optimizers are retained. The network has pretrained feature
lr1e-4, new-module lr.01 and wd1e-4. The center optimizer has glr1e-4 and no
weight decay, including the original center-gradient coefficient compensation.
At epochs10/15/17 the source resets to bases.001/.01 with factors.1/.01/.001.
The first base differs from the initial feature lr; replacing this with a generic
MultiStepLR would change the recipe.

Selection maximizes VAL balanced accuracy, starts at0 and replaces on `>=`, so
the last tied checkpoint wins. The positive decision includes p1==.5, exactly
as in the source. Student-t inference uses the actual source normalized
inverse-distance assignment, without an extra softmax.

Recovery retains all backbone/FPN/shared heads, BatchNorm buffers, centers,
both complete Adam moment/step states, literal LR phase, every selected patch
index, exhaustive scan counts, bag order and all global RNG states. Recovery
occurs at completed original epoch boundaries.

## Unified use

```bash
milplus setup --method 'Center Embeddings'
milplus pack-center-embeddings --table actual_wsi_and_centers.csv --output /data/center_rgb_v1 --level 2 --mult 1
milplus prepare-center-embeddings --data /data/task --schema /data/task/center_schema.json --output /data/task/center_embeddings_v1/manifest.json
milplus run --data /data/task --method 'Center Embeddings' --output /results/center_run
```

The pack table contains unique `slide_id`, actual `wsi` path and `coordinates`
path (NPY or H5 `coords` dataset of actual level0 center points). Extraction
records WSI and coordinate file hashes and original read parameters. Do not
reinterpret TRIDENT top-left coordinates as centers without an explicit,
physically justified conversion.

Set encoder dimension3 and index the original RGB HDF5 files. The schema kind
is `center_embeddings_original_rgb_bags`, layout NHWC/uint8/RGB, patch_pixels256,
normalization `torchvision070_resize256_RHF05_Tensor_norm05_01`, coordinate_frame
`level0_centers`, positive_class equal to the second ordered class, and explicit
image/coordinate provenance, dataset keys and historical asset_root. Bind its
prepared manifest in `encoders.<fm>.center_embeddings.manifest`.

```bash
milplus demo --center-embeddings --patients 6 --output /tmp/center_demo
milplus run --data /tmp/center_demo --method 'Center Embeddings' --config configs/center_embeddings.yaml --output /tmp/center_run
```

Full original source-epoch equality, input checks, real interrupted recovery,
all current regression/profile executions, full native20epoch/batch4 runs and
final installed-package acceptance are separate evidence. Synthetic checks use
the actual pretrained weights and full source network. They do not reproduce
clinical WSI data or the historical paper's metrics.
