# TRIAGE-MIL: full MASS selection and semantic hypergraph survival training

The candidate execution profile is `triagemil_source`; `TRIAGE-MIL` resolves to
it. Full current-version acceptance is pending. Original source pipeline probes
at1024/1536 demonstrate executable author code, not admission or paper results.

Original source: [barathi-1993/TRIAGE-MIL](https://github.com/barathi-1993/TRIAGE-MIL),
commit `e97e52a32f1358974b0145532fdec5f5d75c6864`. The checked tree contains no
license file. Author source remains externally fetched and is read from
verified Git objects; the package contains input/training adapters and source
metadata. No unpinned substitute aggregator is used.

## Real WSI inputs and preprocessing

Supply a standard survival task bundle with patient IDs, clinical event/time
labels, fixed patient splits, full FM feature provenance and true row-aligned
patch coordinates. The native reader selects one diagnostic WSI when several
files match a patient prefix. This adapter requires that representative WSI to
be explicitly selected beforehand. It does not silently choose the first
filename or combine unrelated coordinate frames across slides.

The full original MASS preprocessing executes per WSI, without clinical labels
or pooled TRAIN/VAL/TEST fitting. Default selection requests4096 patches after:

- Magnitude percentile filtering P2/P98.
- Maximum absolute feature z-score <=3.5.
- Absolute-activation entropy >=P3.
- Source composite quality threshold0.30, with its explicit top-quality safety
  fallback when all patches would otherwise be removed.

Four sequential, mutually exclusive axes select MH/TR/MI/TD in fractions
0.35/0.35/0.15/0.15. MH uses feature variance; TR uses magnitude/variance; MI
uses local variance among up to17 actual spatial neighbors; TD uses original
feature-space farthest-point selection. Integer rounding, remaining-patch
selection and quality-based filling remain the original definitions.

The four labels are feature-derived selection groups, not annotated biological
tissue types. They require no external pathology classifier or additional
patient labels. Different FMs can produce different selected patches from the
same available patch pool; this is intrinsic to the complete method and must
be disclosed in controlled FM comparisons.

The actual budget is min(requestedK,clean_count). Preparation never pads fake
patch nodes to4096, fabricates a square coordinate grid or uses the author's
missing-coordinate feature-neighbor fallback without declaring another
experimental condition. All raw indices, group labels, proxy scores, metadata,
real coordinates and FM identities are content-bound.

Original semantic hypergraph construction groups selected features using
KMeans(n_init10,random_state35), approximately12 tiles per super-node. Its
containment, intra-group neighbor10 and cross-group neighbor5 rules generate
a sparse binary tile-incidence matrix. The adapter stores the actual source
matrix and its metadata. It rejects a missing/empty hypergraph or H/X mismatch
instead of bypassing propagation. An optional source-rebuild check verifies
stored arrays against the complete original selector and graph builder.

## Full model and training recipe

Native defaults are four residual hypergraph propagation layers with hidden512
and full-dimensional residual features, six independent gated-attention heads,
dropout0.22, six learnable temperatures clamped to[0.1,5], concatenation/fusion
back to the full FM dimension, layer normalization and a scalar unbounded
risk head. Wider FM channels are retained. No reduced-width model is reported
as a native-default reference run.

The adapter invokes the original complete training and evaluation functions.
Training uses AdamW(lr2e-4,weight_decay1e-4), shuffled DataLoader batch4,
configured batch accumulation(default1), up to200 epochs and no scheduler or
gradient clipping. Source Cox sorts native float32 clinical times decreasing
and applies logcumsumexp, averaging only over observed events. Its handling of
tied times is retained and is not relabeled as Breslow. A batch without an
observed event performs no optimizer update; actual successful steps are
counted without modifying the source optimizer.

On CUDA the original per-epoch AMP scaler and autocast path are preserved,
with float32 Cox. The scaler is freshly initialized each epoch as in the
source. CPU execution disables source AMP. No CUDA validation is implied by
CPU tests.

Validation uses the original nonshuffled DataLoader and native lifelines
C-index. Only a strict increase saves the selected checkpoint, so ties retain
the first best epoch. Early stopping requires bad_epochs>=40 and at least60
completed epochs. The common standardized patient metric retains original
clinical table precision; source float32 times and native C-index are also
exported, avoiding confusion over time ties or precision.

The source returns an undefined native C-index for a constant-risk validation
set or no observed events. The adapter exports that value as null and forbids
selecting it. The internal selection-only value-1 is outside the C-index domain
and is explicitly named `triagemil_selection`; it is not a fabricated score.
The original bad-epoch counter still advances. If training finishes without
any valid selected checkpoint, it fails explicitly as the source does.

Complete selected weights, optimizer, RNG and early-stopping counters are
saved for exact epoch-boundary continuation. Final evaluation actually reloads
the validation-selected model. TEST remains sealed until explicit evaluation;
no TEST metric is used in selection or stopping.

## Preparation and execution

The JSON schema declares `schema_version:1`,
`kind:triagemil_mass_hypergraph`, the actual synthetic marker, WSI selection and
graph provenance, and the exact ordered global patient IDs of TRAIN/VAL/TEST.
Optional `mass`, `selection_budget` and `hypergraph` objects configure only
declared original preprocessing options. The synthetic demo exports the full
schema layout but its Gaussian features and diagnostic-like selection are
software fixtures, not actual WSI evidence.

```bash
milplus setup --method TRIAGE-MIL
milplus prepare-triagemil --data /path/to/task \
  --schema /path/to/triagemil_schema.json --output /path/to/new/assets
```

Register the returned manifest under the selected FM in `task.yaml`:

```yaml
triagemil:
  manifest: /path/to/new/assets/manifest.json
```

```bash
milplus run --data /path/to/task --method TRIAGE-MIL --output /path/to/new/run
milplus evaluate --run /path/to/new/run
```

This is a complete source-method adapter with declared data-domain constraints.
Current full-library admission, CUDA runs, natural WSI experiments and original
paper numbers are separate evidence gates.
