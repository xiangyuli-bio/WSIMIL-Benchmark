# ProtoSurv: tissue-aware graph survival and its complete training recipe

This is the heterogeneous graph/prototype **ProtoSurv**, not another similarly
named survival method. Both `protosurv_source` and `protosurv_repaired` execute
the complete released network and training recipe. `ProtoSurv` selects the
explicitly repaired profile. Version0.53 acceptance includes the current complete2320-case CPU suite, all207profile runs and four independent-wheel full-native-width training runs. GPU and natural WSI validation remain separate.

The primary source is [wjx-error/ProtoSurv](https://github.com/wjx-error/ProtoSurv),
commit `551eaa3c0a2844e2c3557ef412d8a1fd2fec6e20`, associated with the
[NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/760341adc5632de3f1cf2e8d22215a93-Abstract-Conference.html).
No root code license was found at that commit. Source definitions are loaded
from verified external Git objects, not redistributed as locally licensed code.
Any use or redistribution still requires respecting the upstream rights.

## Full network, loss and training

Native defaults retain four GENConv/DeepGCN layers, hidden128, five declared
tissue types, eight learned prototypes per type, prototype width768, two
prototype attention iterations, four cross-view fusion layers and a scalar
risk head. Feature input width comes from the registered FM. A forward-only
graph model with a replacement Cox trainer is not the implemented method.

All three original objectives remain active: source sorted epsilon Cox loss,
0.01 compatibility loss and0.1 input-space orthogonality. The compatibility
term is the released global positive-distance/logsumexp-negative-distance
formula divided by the number of nodes and tissue types. It is not replaced
by a conventional cross-entropy prototype classifier.

The source recipe uses Adam(lr5e-5,weight_decay1e-5), six patients per batch,
two-batch gradient accumulation, the author's event/censor sampler and maximum
100 epochs. Its cosine scheduler has **T_max5**, eta_min1e-6, even over the
100-epoch maximum. Training uses original feature noise, zero-node masking and
spatial edge dropout; TRAIN bags exceeding25000 nodes use the original random
selection. Evaluation uses the full graph.

The released graph also has an unused latent edge set. Its edge-drop
augmentation consumes Python/NumPy random draws. We replay those exact draws
using the original edge count without constructing fake latent edges. The
released latent-graph builder queries a feature-space index with coordinates;
that defective, unused builder is not substituted for the actual spatial graph.

This port explicitly uses FP32 and zero DataLoader workers. It does not claim
the original four-worker/CUDA environment was reproduced unchanged. Patient
times must be integers in the source route because the released dataset casts
them to integer. The repaired route preserves fractional clinical times.

Validation uses the exact source **1-minus-lifelines C-index**, including
observed/censored equal-time pairs. That differs from the common engine's
generic reporting metric for some ties. The first strict improvement selects
the checkpoint. Early stopping uses warmup0, patience15 and only stops after
one-based epoch20. Outer TEST is sealed during training and checkpoint selection.
The independent metric reference is lifelines0.30.0 Git commit
`47afb1c1a272b0f03e0c8ca00e63df27eb2a0560` (MIT).

## Source behavior versus explicit repairs

The source profile retains global `manual_seed` in graph padding, fixed1/gc
loss scaling, the batch-index accumulation boundary and unconditional tail
optimizer call. It also retains the compatibility function's inferred one-hot
width. A training graph lacking the highest tissue category therefore lies
outside its valid compatibility-loss domain and is rejected explicitly.
Evaluation does not require that highest category.

The repaired profile declares the full tissue vocabulary in one-hot encoding,
uses a private generator for seeded padding, uses Breslow handling of survival
time ties and averages a partial accumulation tail by its actual valid-batch
count. It retains the original architecture, compatibility and orthogonality
formulas, augmentation and optimizer/scheduler recipe.

Both profiles store all model/prototype buffers, optimizer/scheduler, random
states, exact completed-epoch/update counters, selection state and input
contracts. Resume is supported at an epoch boundary with an unchanged request.
It rejects changed assets, FM space, graph/category assignments or configuration.

## Data requirements and graph preparation

Use a survival task bundle with event, time and actual time unit. Native
ProtoSurv expects **one explicitly selected WSI per patient**. This adapter
does not silently choose the first slide, union separate slide graphs or
pretend multi-slide training was the original method.

Each slide needs FM features and real row-aligned physical coordinates. Add
real tissue assignments from a documented category model or use the explicitly
declared fixed-prototype mode below. Category arrays are not created from the
feature row number for real data. The demo's modulo categories are synthetic
software fixtures only.

The category index CSV has exactly `slide_id,path` and covers the exact task
slide inventory. Each path points to an H5 with `coords` and integer
`node_types`, in precisely the same patch order as the primary FM feature bag.
The JSON schema contains:

```json
{
  "schema_version": 1,
  "kind": "protosurv_spatial_graph",
  "synthetic": false,
  "tissue_classes": ["tumor", "stroma", "infiltrate", "necrosis", "others"],
  "node_type_mode": "predicted",
  "provenance": "versioned tissue-assignment procedure",
  "graph_provenance": "actual coordinate source and physical extraction protocol",
  "category_model": {
    "name": "actual category model",
    "revision": "fixed weight/configuration revision",
    "weights_sha256": "REPLACE_WITH_ACTUAL_64_HEX_DIGEST",
    "prompt_or_training_provenance": "actual prompts or training cohort provenance"
  }
}
```

The digest placeholder must be replaced with a real SHA256. FM weights,
preprocessing and physical metadata belong in the ordinary encoder registration.

```bash
milplus setup --method ProtoSurv
milplus prepare-protosurv --data /path/to/task --index /path/to/categories.csv \
  --schema /path/to/schema.json --output /path/to/new_graph_assets
```

Preparation executes the original HNSW spatial L2 construction with M16,
post0,efConstruction400,ef90 and an explicit single build thread. It freezes
four outgoing spatial neighbors per node, verifies real coordinate/category
alignment and binds the primary feature, coordinate and graph SHA256 values.
For fewer than five nodes the repaired route explicitly uses min(4,N-1)
neighbors; the source route rejects that unsupported original graph domain.
The generated manifest is registered as:

```yaml
encoders:
  your_fm:
    # Retain the ordinary feature index, weights, preprocessing and spatial metadata.
    protosurv:
      manifest: /path/to/new_graph_assets/manifest.json
```

Fixed prototypes must have shape[tissue_types,FM_dimension] and belong to the
**same registered FM weight/preprocessing space**, not merely the same width.
Pass `--prototypes` and `--prototype-provenance`. Provenance must declare either
TRAIN-only fitting with unique TRAIN global patient IDs or an explicitly
identified external population. VAL/TEST fitting is rejected.

For the author's unknown-category fixed-prototype path, use
`node_type_mode: unknown_fixed_prototype` and blank category-index paths.
The actual author dataset supplies-1 tissue sentinels. Its default nonzero
compatibility objective then fails on those sentinels. This port requires both
`model.pre_proto: true` and explicit `model.comp_alpha: 0` for that mode; it does
not fabricate tissue classes to make the original default loss run.

## Execution and evidence boundaries

```bash
milplus check --data /path/to/task --method ProtoSurv --deep
milplus run --data /path/to/task --method ProtoSurv --output /path/to/new_run
milplus evaluate --run /path/to/new_run
```

The 28 focused software checks include independently loaded original complete
networks at1024/1536 feature widths, all three objectives and all active
gradients; the actual original trainer/validation/Adam/scheduler/random state;
metric ties; true CLI execution; checkpoint recovery; graph/prototype
provenance; and both accumulation-tail policies. Full-library and independent
native-width training evidence is recorded separately when complete.
Synthetic acceptance does not establish performance on BRCA/COAD, completion
of a real cohort experiment or reproduction of the published paper metrics.

The accepted0.53independent-wheel receipt is retained as docs/validation/protosurv_wheel_053.json with its original environment identity. It is historical evidence, not a claim of GPU verification of a newer whole package.
