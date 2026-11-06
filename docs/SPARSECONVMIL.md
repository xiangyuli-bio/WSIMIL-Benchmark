# SparseConvMIL: complete released-source training condition

Original project: https://github.com/MarvinLer/SparseConvMIL,
commit `d7dd705bc3dd4ebfb6a36f0589698400c9fa4c15`, AGPL-3.0.
Dependency: original Facebook SparseConvNet,
commit `cf251d058959a9dbaccb25fc919dc4f4548be232`, BSD-3-Clause.
Author code is fetched separately and executed from verified Git objects.

The two explicit profiles, `sparseconvmil_raw_source_c32` and
`sparseconvmil_raw_source_c128`, use the full trained ResNet18, two original
3×3 submanifold sparse convolutions, ReLU, active-site-only global average
pooling, and the original 32-hidden MLP. There is no frozen feature substitute
or dense-grid approximation. Duplicate coordinates are averaged by the actual
compiled `InputLayer(mode=4)` backend. Sampling uses Python `random.choices`
with replacement. Native public defaults are five tiles per slide, slide batch
two, Adam lr0.002/weight_decay1e-6, 200 epochs, no scheduler, clipping or early
stopping. Final weights are selected independently of validation scores.

The public source and paper are different conditions. The paper's TCGA appendix
specifies a pretrained ResNet34, 200 cropped 128px tiles/WSI, slide batch10,
Adam1e-4/weight_decay1e-4, 200 epochs, inverse-class oversampling and additional
pixel augmentation. The released script defaults to random ResNet18,
ToTensor-only input and the recipe above. These profiles implement the complete
released script, and do not claim the paper's TCGA protocol or its scores.
See https://proceedings.mlr.press/v156/lerousseau21a.html and its supplementary PDF.

Source coordinate augmentation is preserved, including `uniform(-180,180)`
passed directly to sin/cos, per-axis resize/flip, random translation before
truncation and final offset. The source executes these operations even during
eval. Public `main()` has no validation/evaluation implementation. The
benchmark adds fixed-per-slide seeded sampling and these original transforms
for inference, restores all global training RNG afterward and never uses VAL
scores for model selection. This extension is explicitly recorded in each run.

Other ports: explicit network/device placement fixes the source main's missing
model placement; modern torchvision uses a keyword-only weights argument;
the loader uses workers0 for complete epoch-boundary recovery. The full C++
backend is compiled unchanged. Its source/extension identities enter the model
contract. Network, BN buffers, all Adam states, per-epoch sampled indices/order
and RNG are retained. Changing actual pixels, coordinates, patients, labels,
split, backend or sampling settings invalidates the run.

## Setup

```bash
python -m milplus setup --method SparseConvMIL
python scripts/build_sparseconvnet_backend.py --output /your/own/scn-build --jobs 1
export PYTHONPATH=/your/own/scn-build/overlay:$PYTHONPATH
```

Run the build on the host where the method will execute. The source setup
selects its CPU or CUDA extension using actual CUDA availability; a CPU-only
binary cannot be presented as a CUDA backend. Build logs and manifest are kept
in the requested directory. Existing Python environments are not modified.

## Actual images

Prepare a CSV with `slide_id,path`; each path points to an original author-style
tile folder whose image names encode `x_y`, such as `512_1024.png`. Pixels must
be square and retain their original dimensions. Packing preserves original
PIL RGB conversion, actual folder enumeration order, coordinates and per-file
hashes. It never resizes, normalizes or reconstructs missing coordinates.

```bash
python -m milplus pack-sparseconvmil --table folders.csv --output /own/packed
python -m milplus prepare-sparseconvmil --data /own/task \
  --schema /own/schema.json --output /own/task/sparseconvmil/manifest.json
python -m milplus run --data /own/task --method SparseConvMIL \
  --config configs/sparseconvmil.yaml --device cuda:0 --output /own/run
python -m milplus evaluate --run /own/run --subset test --device cuda:0
```

Bind the resulting manifest in `encoders.<raw_encoder>.sparseconvmil.manifest`.
The ordinary task bundle still needs real labels, patient identities and a
patient-disjoint split. Declare the actual coordinate units in schema as
`level0_top_left_pixels` or `tile_grid_indices`; spatial downsampling operates
in that declared frame. Schema fields are demonstrated by the explicit
synthetic software fixture:

```bash
python -m milplus demo --sparseconvmil --patients 20 --output /own/scn-demo
```

Synthetic fixtures and software tests do not constitute real WSI training,
clinical validation, FM extraction or reproduction of reported paper metrics.

## BRCA native runtime preparation, 2026-11-06

The retained legacy extension was inspected on the actual H200 runtime and
returned `is_cuda_build() == False`; it is not a usable CUDA backend. The host
default compiler is CUDA13.2, whereas the unchanged training runtime is
Torch2.5.1+cu124. Seven original NVIDIA CUDA12.4 packages were downloaded with
their repository-listed sizes/SHA256 verified, then extracted into a separate
owned prefix without apt installation or any active Python environment edits.
The compiler is12.4.131; native H200 code is built for SM90.

The entire pinned SparseConvNet source was compiled unchanged in
`/root/milplus_brca_uni_20261105/sparseconvnet_cuda_backend_20261106_r1`.
The normal build script verifies every source file after compilation and
records all extension/Python file hashes. No dimensions, kernels or operators
were removed. The existing CPU extension and all active training paths remain
unchanged; the CUDA overlay is not injected into already running jobs.

Four actual H200 software checks passed: c32/c128 original CUDA submanifold
outputs, duplicate-site averaging and input/weight/bias gradients against an
independent active-site formula, plus c32/c128 complete RGB256 ResNet18,
two sparse layers and classifier backward/Adam updates. A complete real
16848-patch bag separately passed the unchanged RGB/coordinate/identity input
schema. These are dependency and input-interface checks, not BRCA training.
The full856 RGB cohort and immutable per-method manifest are still required.

The eligible source routes remain the explicitly documented released script
condition (random ResNet18,5tiles/slide, batch2,200epochs). They must not be
called the different pretrained ResNet34/128px TCGA paper condition.
Shared evidence: `cuda124_toolchain_preparation.json`,
`native_sparseconvnet_cuda_build.json`,
`native_sparseconvnet_cuda_validation.json`,
`sparseconvnet_original_cuda_tests_r1.xml`, and
`sparseconvmil_real_rgb_pilot_contract.json` in the BRCA working share.

### Full-cohort preparation and queue gate

The preparer is now live (PID566971, starttime832452850), with frozen source
`sparseconvmil_preparer_source_20261106_r1/prepare_native_sparseconvmil.py`.
Before launch it checked316 actually delivered archives against their real
receiver receipts, original coordinate SHA, row identities and current file
fingerprints. This check explicitly does not claim a complete cohort.
The receiver's coordinate digest is nested at
`source.coordinates_int64_le_sha256`, not a hypothetical top-level field.

After the real receiver completes all856 bags, the preparer will call the
unchanged native data preparer, hash every complete archive, preserve the
original742/69/45 split and IDC0/ILC1 labels, bind the actual compiled backend,
and create a new immutable RGB task/manifest. No original image is copied or
deleted, no classifier fit is initiated by the preparer, and no source-native
sampling or200-epoch budget is reduced. Only then may it submit one intake
request to the existing controller. A rejected request preserves its failure;
there is no silent alternative method, backend or dataset.

The V4 controller (PID566749, starttime832444855) extends V3 admission only for
this verified source route. It retains existing workers, reservations, original
recovery/numeric-repair paths and the73-task ceiling. It requires the full RGB
receiver completion, original task/manifest SHA, source recipe, unchanged CUDA
binary/Python files and four actual passing GPU test cases. CPU-only overlays,
changed inputs/configuration, incomplete cohorts, overwritten runs and excluded
methods are rejected. All38 new/existing intake-control tests passed before
handoff. The five then-live repair fits kept their actual PIDs/start clocks;
no training process or process group was signaled.

Evidence: `scheduler_sparseconvmil_controls_r1.xml`,
`scheduler_v4_handoff_launch.json`, `native_sparseconvmil_preparer_launch.json`,
`sparseconvmil_preparer_available_receipts_check.json`; live preparation state:
`native_sparseconvmil_preparation.json`. Waiting for input or successful
admission must not be counted as completed model training.
