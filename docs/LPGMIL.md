# LPGMIL complete reference adapter

This adapter retains the full published512D model: two Mamba1.1.1 blocks
(state16, convolution4, expansion2), six learnable prototypes, eight-head
masked attention per branch, native top1000 instance selection, the separate
slide reaggregation head, and all losses. It is a named reference reconstruction
of a released implementation with execution faults, not a paper-score reproduction.

## Sources and explicit repairs

The author repository is `FPXMU/LPGMIL`, commit
`6b14bb01dff042a09bd34ebe956d696e665a7cd0`. The environment pins
`mamba-ssm==1.1.1`; its official source is pinned separately at
`fb7b5310fa865dbd62aa059b1e26f2b431363e2a`. Original definitions are privately
loaded from verified clean Git objects; their code is not bundled in this wheel.

The released model accepts `num_tokens`/`num_masked_patches`/`prototypes`, while
the trainer uses incompatible older keyword names. The trainer unpacks seven
outputs, while the released network returns six. The source also sends raw
1024D learnable prototypes directly into512D attention. The reference applies
the already published prototype projection to these learnable queries, registers
the fixed selector as a device-aware checkpoint buffer, and removes only the
single-bag dimension from the six-branch logits before cross entropy.

The original first-head, mean-over-six-queries top1000 selection is retained.
It is hard selection: selector-only modules do not receive task-loss gradients.
They still receive the author's all-parameter L1 regularizer. The source Xavier
initializer, including its overwriting of special Mamba linear biases, remains
unchanged. There is no missing-SSM or random-prototype fallback.

## Required prepared data

The task uses ordinary frozen FM feature bags with original ordered patch IDs.
H5 bags can include `features` and `patch_ids`; other formats require a
`patch_ids_path` index column. All patients must be separated across splits.

The original `proto_vector.pt` and its construction code are absent from the
released repository. This adapter therefore supplies an explicit reference
preparer. It accepts actual nucleus counts computed beforehand using a documented
detector, selects lymphocyte-dense TRAIN patches using declared thresholds, and
clusters all candidates using Ward hierarchical linkage into exactly six groups.
Each initial prototype is the float32 mean of its group's actual FM features.
Thresholds and Ward linkage are reconstruction choices, not recovered author
defaults. No candidate is secretly subsampled.

Counts CSV columns:

```text
slide_id,patch_id,n_lymphocytes,n_nuclei
```

It must contain every patch from every TRAIN slide and no VAL/TEST slide.
Duplicate IDs, missing rows, fractional or negative counts, and lymphocyte counts
exceeding total nuclei are rejected. A detector JSON must describe `detector`,
`version`, actual `weights_sha256`, `patch_geometry`, and `nucleus_type_mapping`.
These are provenance declarations for previously computed counts, not proof that
the preparer itself executed a nucleus detector. Counts and prototypes are bound
to the actual feature bytes, ordered patch identities, labels, split and patients.

```bash
milplus setup --method lpgmil
milplus prepare-lpgmil --data /data/task \
  --counts /data/task/train_nucleus_counts.csv \
  --detector-provenance /data/task/hovernet_provenance.json \
  --min-fraction 0.75 --min-nuclei 10 \
  --output /data/task/lpgmil_assets
```

Then set the chosen encoder's metadata in `task.yaml`:

```yaml
lpgmil:
  manifest: lpgmil_assets/manifest.json
```

Normal execution uses the same unified entry point:

```bash
milplus run --data /data/task --method LPGMIL --device cuda
```

`lpgmil_tls1024_reference` preserves the original1024D, binary TLS task contract.
`lpgmil_fm_width_reference` explicitly ports the feature/prototype input widths
and output class count while retaining the complete512D architecture. A new FM
requires constructing prototypes in that actual FM feature space.

## Training and selection

The native budget is50epochs, one original WSI per step, Adam(lr1e-4,
weight_decay1e-5). Loss is bag CE + mean six-branch CE +1e-5 times the sum of
absolute values of all model parameters. No gradient clipping or scheduler is
introduced. The source's unused `accumulation_steps=4` variable does not produce
accumulation; its real optimizer step is per bag, which is preserved.

Source default early stopping is disabled and its final epoch is evaluated.
The wrapper therefore selects the final epoch; validation remains recorded.
Thresholds are never fitted using test labels. Predictions use source softmax
probabilities, with optional patient averaging from the common evaluation layer.

The explicit `torch` backend executes the complete original Mamba graph and
selective recurrence with state-preserving checkpoint chunks. Chunking does not
truncate the instance sequence or state. An optional `cuda` backend requires
the exact pinned Mamba1.1.1/causal-conv1d1.1.3.post1 operator environment and
fails if those binaries are absent. It never silently changes backend.

All original weights, fixed and learnable prototypes, optimizer moments, sampling
history, RNG and immutable input identity participate in checkpoint recovery.

## Validation state

Eighteen native-architecture checks and eleven common interface checks passed, including the independent published
Mamba reference equations and gradients, the literal unrepaired source failure,
all losses/gradients/Adam states against the repaired original graph, complete
epoch recovery, actual1025-patch top1000 selection, TRAIN-only centroids, invalid
counts, feature tampering and the full-width multi-class port.

These are software checks on synthetic arrays. Complete public CLI/source/wheel
process recovery, native50epoch budgets, whole-library and final release gates
remain required. No clinical WSI experiment or paper metric is claimed.
