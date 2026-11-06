# cDP-MIL full-covariance precision condition

The literal default remains `aggregation_dtype='float32'`. The new opt-in
`aggregation_dtype='float64'` changes only the Bayesian mixture's covariance,
fit and assignment arithmetic. It retains full covariance, components10,
random_state0, iterations30, concentration0.1 and sklearn's unchanged default
reg_covar1e-6/tol0.001. Hard-assignment feature means and classifier tensors
remain original float32. No rank reduction, extra regularization, component
reduction, patch omission, label-driven fitting or changed training budget.

This is an explicit numerical repair, not a claim of bitwise equivalence to
the float32 path. The actual precision is recorded in the model config and
protocol. Cache identity includes precision. `cache_bags=856` in the new BRCA
attempt only avoids refitting the same content-addressed deterministic per-bag
mixtures; centroids occupy roughly tens of MiB. It does not share a learned
cross-slide mixture or use held-out labels.

Real evidence: original TRAIN slide TCGA-AN-A0XW-01Z-00-DX1, all153x1024 original
feature rows, source HDF5 SHA256
`541cafb9d45d6f5ab90f683ab4d93639f40951d008b21e499496a8db73742c83`.
Float32 reproduces the ill-defined empirical covariance failure. Float64,
with identical mixture parameters, converges in2 iterations with10 observed
components and finite parameters. The full source adapter produces10x1024
finite float32 centroids on the same actual bag in both the development CPU
environment and the existing GPU allocation's actual runtime (CPU sklearn
preprocessing). This is not full856 or classifier training validation.

Verification:39 numerical and existing original-source preprocessing/training,
cache/RNG, recovery and evaluation checks passed. Evidence under the BRCA
shared experiment directory: `cdpmil_real_numeric_diagnostic.json`,
`cdpmil_precision_cpu_tests.xml`,
`cdpmil_numeric_candidate/gpu_runtime_probe.json`.

The active original GPU source is unchanged. A separate frozen copy
`/root/milplus_brca_uni_20261105/code_cdpmil_numeric64_20261105` differs only in
`src/milplus/cdpmil.py`; full copied source SHA values are recorded in
`cdpmil_numeric_candidate/source_receipt.json`. The fresh, explicitly named
`cdpmil_numeric64_repair2` attempt is accepted into the existing resource-aware
queue. Original failed runs remain intact. Completion requires its actual
full fitting and locked TEST result; the diagnostic is not counted as success.

## Subsequent actual configured BRCA completion (2026-11-06)

The separate frozen0.131.0 `cdpmil_numeric64_repair2` run completed its configured
benchmark fit:17 recorded epochs, maximum50, validation patience10. The first
maximum validation AUC is at epoch6 (zero-based). This is validation-selected
early stopping, not a claim that all50 epochs or the full paper protocol ran.
All856 original patients and2,422,295 UNI rows remain in the deep input inventory,
with742/69/45 splits and `max_instances=null`. Components10, full covariance,
concentration.1 and mixture iteration limit30 remain unchanged. The explicit
float64 arithmetic condition above remains part of the result identity.

The subsequent locked TEST receipt completed at2026-11-06T00:02:11Z, covering
all45 original TEST patients. Independent recomputation gives AUC.9892241379,
accuracy.9333333333, balanced accuracy.9342672414 and macroF1.9282296651.
The complete59-method TEST identity/metric audit has zero errors. This adds one
completed configured BRCA result, not final all-method or paper-reproduction
acceptance. The fitting status's `test_evaluated=false` is preserved: it describes
the fit phase; the separate evaluation receipt proves the later TEST evaluation.

Request, input inventory, protocol, environment, status and all17 history rows
are copied byte-exactly into shared `results/cdpmil/source_run_evidence_r1/`.
Its `audit_receipt.json` binds the actual selected checkpoint, exported request,
prediction/metric hashes and independent TEST audit. No inference/training was
rerun, no source or result was changed, and no instrumented per-update visit
trace is asserted. Checkpoint SHA256:
`a49bdeb396a936d1e8bf5e3264139b499b3a573e43df06685b8c27df0b2906f8`.
