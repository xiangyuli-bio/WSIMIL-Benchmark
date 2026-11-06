# AHDMIL

The adapter follows `JiuyangDong/AHDMIL@2f38354386165167815eb2998424ad63792a65bf`
and first performs the required DMIN pretraining from the fixed HDMIL source.
It runs the complete pretrained-DMIN → asymmetric joint-training pipeline;
neither the high-resolution classifier nor the two RGB experts is replaced.

| Profile suffix | Classes | DMIN degree / LR | Joint attention-branch probability | Cross-expert EMA |
|---|---:|---|---:|---:|
| `camelyon16_source_repaired` | 2 | 12 / 3e-4 | .4 | .2 |
| `nsclc_source_repaired` | 2 | 16 / 3e-5 | .1 | 0 |
| `brca_source_repaired` | 2 | 12 / 3e-4 | 0 | 0 |
| `rcc_source_repaired` | 3 | 12 / 3e-4 | 0 | 0 |

Prefix each profile with `ahdmil_`. `AHDMIL` defaults to the BRCA recipe.
These names choose original hyperparameters and class counts; the actual
labels, patients and held-out partition are always read from the task bundle.

## Prepare and run

Use the measured paired RGB16/256px high-feature asset format documented in
[HDMIL](HDMIL.md). Both methods can use the same registered assets. Feature
coordinates and RGB rows must match exactly; one cannot derive the required
RGB from frozen feature vectors or invent missing spatial coordinates.

```bash
milplus setup --method AHDMIL
milplus prepare-hdmil --data /data/task --index /data/task/rgb_index.csv \
  --schema /data/task/rgb_schema.json --output hdmil/manifest.json
milplus run --data /data/task --method ahdmil_brca_source_repaired \
  --output /results/ahdmil_brca --device cuda:0
```

Register the generated manifest as `encoders.<FM>.hdmil.manifest` in the task
YAML. Setup obtains both separately licensed original source repositories.
Original code is loaded from verified Git objects rather than redistributed.

The default first stage uses all DMIN loss terms and the original 200-epoch
maximum, strict validation CE selection starting at one-based epoch50, and
patience20. The selected weights, not the final pretraining epoch, initialize
the joint stage. The same seed is reset to reproduce the authors' separate
training program initialization.

The joint stage uses 200 epochs maximum and patience5, with two native
MobileNetV4 experts, 512/256 gated attention, and degree12/16 Chebyshev heads.
Each bag first updates LIPN using hard-mask or raw-attention distillation and
retention matching, then updates DMIN using the selected high features.
Both joint Adams use LR1e-5 and weight decay1e-5. All training stages use the
source class-balanced replacement sampler and complete bags.

Validation selects only strict CE improvements with mean retained fraction
between .3 and1. If no eligible checkpoint has appeared after five attempts,
both joint networks and Adams are reset from the original selected DMIN.
The RCC lower bound decreases by .05 on such resets after joint epoch50.
If the declared budget ends without an eligible checkpoint, the run fails
explicitly; it does not force an inadmissible checkpoint onto the leaderboard.

Checkpoints retain three optimizers, pretrained tensor identity, branch and
sampling history, cross-EMA count, reset generations, selection trace, and RNG.
Cross-EMA updates parameters simultaneously while preserving BatchNorm buffers.
The second per-bag expert forward updates those buffers in train mode, exactly
as in the source even though it runs inside `no_grad`.

The `source_repaired` name documents two concrete source faults: the entry
imports `AHDMIL` but defines `HADMIL`; and the RCC reset branch writes
`args.lower_ratio` although the initialized selection field is `self.lower_ratio`.
See the [source audit](AHDMIL_AUDIT.md).

All current fixtures are synthetic engineering checks. Other FM dimensions
are explicit benchmark conditions, not the paper's original ResNet50 setting.
No natural WSI results, historical paper metrics or online FM inference speed
claim has been established. Full native budgets, package checks and whole-library
regression remain separate from short mechanism and continuation tests.

## Actual BRCA IDC/ILC scope check (2026-11-06)

The actual856-patient binary bundle matches the original BRCA recipe's two
classes. Registered bag counts have a minimum128 instances versus native
`k_sample=32`; this is a metadata capacity check, not a new full H5 audit or
proof of prepared measured inputs. The real bundle has no registered native
paired RGB/high-feature contract, and the unchanged constructor rejects it
before network construction. The mixed wording of that exception must not be
misreported as evidence of a class-count mismatch.

Continue the original physical RGB16/high-feature preparation and complete
200+200 stage budgets. Do not exclude this method for being nonbinary, derive
missing image scales from feature vectors, or claim existing10x cached RGB
already meets the original low/high physical field of view. Shared evidence:
`brca_hierarchy_and_ahdmil_actual_scope_r1.json`.
