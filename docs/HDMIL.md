# HDMIL

This adapter executes the complete released DMIN → LIPN pipeline from
`JiuyangDong/HDMIL@ef3d4681054970208e94f8205bd96b3180c00926`.
The four source task recipes preserve their native 512/256 widths, degree12/16
Chebyshev classifiers, instance SVM, Gumbel self-distillation, RGB MobileNetV4
and both training stages. Original source is fetched separately by `milplus setup`.

| Profile | Classes | DMIN degree / LR / retention | Instance k / subtyping |
|---|---:|---|---|
| `hdmil_camelyon16_source` | 2 | 12 / 3e-4 / .6 | 32 / false |
| `hdmil_nsclc_source` | 2 | 16 / 3e-5 / .7 | 32 / true |
| `hdmil_brca_source` | 2 | 12 / 3e-4 / .7 | 32 / true |
| `hdmil_rcc_source` | 3 | 12 / 3e-4 / .6 | 8 / true |

`HDMIL` resolves to the BRCA recipe. A recipe name selects hyperparameters, not
data labels: the task package remains the source of patient labels and splits.
Different frozen feature dimensions are supported and recorded. Other FM
weights constitute explicit benchmark conditions, not the original ResNet50
paper condition. Extra classes require a separately reviewed experiment.

## Inputs

The task bundle needs frozen features and real level0 patch coordinates. Each
feature row additionally needs its matching **16×16 RGB uint8 patch at 1.25×**,
covering the same physical region as the **256×256 patch at 20×**. These input
conditions follow [the original method description](https://arxiv.org/html/2502.21130v1).

Create a CSV with `slide_id,path`. Each path is an H5 file containing `rgb`
(`[N,16,16,3]`, uint8) and `coords` (`[N,2]`, integer level0 top-left coordinates).
H5 attributes `level0_mpp`, `high_mpp`, and `low_mpp` must record each slide's
measured resolution; `low_mpp == 16 * high_mpp`. Ordering must exactly match the
primary feature file. No synthetic coordinate or image fallback exists.

Schema JSON:

```json
{
  "schema_version": 1,
  "kind": "hdmil_matched_low_resolution_rgb",
  "coordinate_frame": "level0_top_left_pixels",
  "rgb_dtype": "uint8",
  "rgb_layout": "NHWC",
  "low_patch_pixels": 16,
  "high_patch_pixels": 256,
  "low_magnification": 1.25,
  "high_magnification": 20,
  "rgb_key": "rgb",
  "coords_key": "coords",
  "coordinate_provenance": "YOUR measured extraction record",
  "image_provenance": "YOUR WSI reader, level selection and resampling record",
  "weights_provenance": "YOUR frozen encoder and exact weight version"
}
```

```bash
milplus setup --method hdmil
milplus prepare-hdmil --data /data/task --index rgb_index.csv \
  --schema /data/task/rgb_schema.json --output hdmil/manifest.json
```

Then add to the task's selected encoder entry:

```yaml
hdmil:
  manifest: hdmil/manifest.json
```

```bash
milplus run --data /data/task --method hdmil_brca_source --output /results/hdmil
```

The manifest binds RGB/features, extraction metadata, patient identities, labels,
splits and row order. Re-preparation uses a new output path; changed inputs are
rejected rather than silently reused. At least the native instance k must exist
in every slide. Missing low-resolution RGB requires actual extraction from WSI;
feature tensors cannot be converted into substitute images.

## Training and evidence

Each stage has a 200-epoch maximum. DMIN selects the first strictly lowest mean
validation slide CE starting at one-based epoch50, patience20. LIPN restores this
selected teacher, resets the same seed as the author's separate script, freezes
teacher parameters and trains its native RGB processor with Adam1e-4, patience5.
The frozen teacher deliberately remains in training mode during distillation,
including its dropout. Both stages use class-balanced replacement sampling.

`last.pt` includes both Adam states, stage counters, selected teacher tensors and
hash, complete validation history and RNG. `best.pt` selects **LIPN-stage**
validation performance only. Test evaluation is separate and cannot choose the
teacher or processor. Deliberately shortened software checks are explicitly
configured; they are separate from the complete native-budget gate.

`demo --hdmil` creates synthetic software inputs. No real TCGA/WSI result or
paper metric is claimed by these checks. The released implementation loads
cached high-resolution features. This route cannot substantiate the paper's
online cropping/FM extraction speedups; that cost needs an online WSI pipeline.

See [HDMIL_AUDIT.md](HDMIL_AUDIT.md) for exact source quirks and validation scope.
