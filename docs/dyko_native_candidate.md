# DyKo native adapter candidate

DyKo now exposes eight public execution profiles marked
`adapted_under_verification`; see [usage and defaults](LAST_THREE_METHODS.md).
The adapter and its original-input guards are integrated; they do not establish
full TITAN initialization, original-model training or paper-score reproduction.

The original source is locked to
`a8b43645840cf283c334b07c42cf4ddf230f644a`. Its native inputs are 768-dimensional
CONCHv1.5 features, the released 1,000-concept bank and ordered class prompts,
and the original trainable TITAN contextual text tower. A generic feature
matrix of the same width or cached fixed class vectors cannot satisfy this
contract. The `UBC` prompt classes in this release describe ovarian subtypes.

The local asset preparation command is:

```bash
milplus prepare-dyko-assets \
  --titan-directory /absolute/path/to/authorized/original/TITAN \
  --output /absolute/path/to/titan_assets.json
```

This command checks files already present locally. It does not obtain model
access, download restricted files, accept model conditions or load substitute
weights. The file lock records publicly advertised Git/LFS identities for
revision `dac6773d9961cfc75503440676ff157a2c6e8d2e`. The original DyKo source
did not pin its TITAN revision, so this is an explicit asset reference rather
than a claim about the exact historical checkpoint used for its paper.

The feature/input preparation command binds patient partitions, WSI bytes,
visual extraction provenance, original concepts, original prompt/class order
and the local TITAN asset manifest. Its schema has these fields:

```json
{
  "schema_version": 1,
  "cohort": "NSCLC",
  "class_map": {"LUSC": "LUSC", "LUAD": "LUAD"},
  "titan_assets": "/absolute/path/to/titan_assets.json",
  "visual_extraction_manifest": "/absolute/path/to/visual_extraction.json",
  "provenance": "Actual extraction and patient-partition record"
}
```

Class mapping preserves the source prompt order and only translates labels
and probability columns at the common benchmark boundary. Few-shot selection
uses the original per-class pandas sampling on TRAIN rows, with explicit
patient leakage rejection and separately counted patients and slides. A
comparator must use the same selected training population.

The candidate model builder preserves the original full text transformer,
16 learned prompt context tokens, native adapters, two cross-attention paths,
cluster heads, per-forward concept retrieval and CE plus unweighted KL. Its
runtime uses source Adam parameter groups, FP32, accumulation by four,
normalization by four for short tail groups, validation accuracy selection,
and the original 1,000-epoch/40-patience limits. Complete checkpointing adds
explicit partial gradients, optimizer state and RNG state to the original
weights-only checkpoint capability.

There is a source dependency conflict: the requirements pin FAISS CPU while
the model requests GPU K-means. The candidate distinguishes a named CPU
reference from a real CUDA-FAISS route. It cannot silently fall back to CPU
and report a GPU-native validation. The common PyTorch 2.5.1 runtime is also
an explicit port from the source's PyTorch 2.1.2 environment.

Validation completed so far:

- 30 original-source component checks and 16 additional input-contract checks.
- A separate small dropout control graph compared with actual Lightning 2.3.0
  and an eight-worker DataLoader, using the original training-step body and
  source Adam grouping. All per-batch weights, optimizer tensors, gradients,
  RNG states, tail updates and early stopping matched exactly in two cases.
- Fresh-process interruptions after batches 3, 4, 7 and 8 recovered the same
  21-batch trajectory and final state in that control graph.

The control graph is not a DyKo/TITAN replacement. Its first comparison found
that Lightning clears gradients before validation; the candidate was fixed
and the original failed result was retained. Full original assets, complete
model execution, source/wheel validation and native-budget checks remain
required for full research validation, independently of code registration.
