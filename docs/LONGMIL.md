# LongMIL execution conditions

The pinned [public author repository](https://github.com/invoker-LL/Long-MIL),
commit `457b1cf6e33a05b3c8359e1fe98114cd8241f772`, releases **v1**.
The [NeurIPS2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/b7eecb72574b043ad0c69ea296212450-Paper-Conference.pdf)
also describes a different local/chunked v2. Its code and full trainer remain
unpublished. These conditions have separate names and evidence.

| Profile | Actual implementation |
|---|---|
| `longmil_v1_source_classification_reference` | Original three full-width384, four-head blocks; full90000x90000 int8 ALiBi; original joint-head minimum local mask and eval dropout0.05. Kernel failure now raises instead of silently substituting identity. |
| `longmil_v1_eval_repaired_classification_reference` | Same original v1, explicitly switching attention dropout off during evaluation. |
| `longmil_v2_paper_classification_reference` | Declared paper reconstruction: two exact radius10 two-dimensional sparse local blocks, real2x2 pooling, then full global ALiBi2D attention. Native width384, FF1536, one head, dropout0. |
| `longmil_v2_paper_survival_reference` | Same full paper reconstruction, patient risk head and explicit full-TRAIN-cohort Cox/Breslow reference training. |

`LongMIL` selects the paper classification or survival condition from the task.
All four routes require real CUDA xFormers. CPU dense equations are independent
small-input tests, never runnable method substitutes. Unsupported kernels fail.

## Dataset contract and commands

Prepare the ordinary task bundle with frozen patch features, real XY coordinates,
patient labels and patient-disjoint train/val/test splits. Under each chosen
encoder in `task.yaml`, declare the measured physical grid:

```yaml
encoders:
  UNI:
    dimension: 1024
    index: features_UNI.csv
    longmil:
      coordinate_frame: level0_xy
      patch_step_level0: 448
      grid_origin_xy: [0, 0]
```

Here448 is an example only. Use the actual level-0 tile step and origin from
TRIDENT/your feature extraction metadata, retaining row-aligned coordinates.
Misaligned or duplicate patch coordinates fail; no sequence pseudo-grid is made.
Holes in foreground tissue remain holes. The paper reference extrapolates beyond
the original public v1's300x300 position grid.

```bash
milplus run --data /path/to/task --method LongMIL --device cuda:0 --output /path/to/run
milplus evaluate --run /path/to/run --device cuda:0
```

The public v1 conditions additionally need `v1_alibi_path` and
`v1_alibi_sha256` under `encoder.longmil`. The entire native90000x90000 int8
asset must be present. Both the serialized file checksum and **every tensor byte**
are verified. Its values have fixed SHA256
`137574db94efa7d3336d91052f34390903b5f62252e095f8e2cb2d66324bdb8a`.
It is registered as a non-persistent device buffer, retaining the complete native
table while avoiding an extra8.1GB copy in every checkpoint. No smaller table is
substituted. Source-v1 radius truncation, slopes, padding and minimum-mask behavior
remain unchanged.

## Reconstruction choices and training

The paper specifies two local blocks, one global block, one head, dropout0,
radius10 (Appendix A.5.3), 2x2 pooling (Section3.3), slide batch1, LR1e-4 and
weight decay1e-2 (Section4). Sparse local neighbourhoods use actual2D Euclidean
distance and exact radius queries; chunked xFormers skips disallowed pairs rather
than allocating a full N-squared local matrix. The global stage retains every
pooled token and pair, with query chunks to bound its attention-bias memory.

Unpublished details are explicit reference choices: public-v1 width384 and
FF1536; observed-tissue mean pooling and coordinate centroids for partially
occupied2x2 cells; public-v1 ALiBi slope convention with true float Euclidean
distances for v2; Adam; max200epochs/patience20; validation macroAUROC or patient
C-index selection; and Cox/Breslow for survival. These are not claimed as
recovered original v2 training code or reproduced paper scores.

Independent slides from one patient retain their individual coordinate frames.
The reference averages slide logits/risks for patient training/inference. Cox
training uses the complete TRAIN patient risk set through exact two-pass
backpropagation; no test/validation patient enters it. Complete bags are mandatory.
Arbitrary width/depth changes and silent sampling are rejected.

The original v1's stochastic eval uses an explicitly recorded per-unit seed,
forked from training RNG so inference does not alter subsequent updates. A
separately named condition disables eval dropout. Run requests bind actual
feature bytes, coordinates, data configuration, source and environment. Model,
optimizer moments, RNG, selected weights and history use the shared exact
checkpoint lifecycle. Native budgets, fresh-process recovery/reload, independent
wheel identity, whole-library and release checks are separate acceptance gates.

Current development is synthetic engineering validation. New raw WSI experiments
and paper-score reproduction have not been performed.
