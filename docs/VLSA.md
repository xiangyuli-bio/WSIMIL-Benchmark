# VLSA: native CONCH ordinal survival, development candidate

The adapter privately imports the complete author definitions from
`liupei101/VLSA@915f37afcbd86cd6c33ebb30ab0caa5def0f9ab9`.
The source is CC-BY-NC-ND-4.0; this package does not redistribute it.
This candidate is **not formally admitted**. The author-example numeric
discrepancy below remains open in addition to full-library/package gates.

## Inputs

Use a normal survival task bundle with `training_unit: patient`,
`evaluation_unit: patient` and explicit `time_unit: months` or `days`.
Every patient has actual event/time labels and all of its registered slides.
The encoder declares `dimension: 512`, `model: conch_ViT-B-16`,
`weights_sha256: 40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c`,
`feature_preprocessing`, and `vlsa: {manifest: vlsa_inputs.json}`.
The feature preprocessing must agree with the actual extraction. The author's
WSI pipeline uses 448x448 patches at20x and CONCH projected visual features.
Unrelated FM embeddings and learned projection substitutes are not this route.

Example `vlsa_inputs.json`:

```json
{
  "schema_version": 1,
  "kind": "vlsa_native_conch_survival",
  "synthetic": false,
  "checkpoint": "/actual/path/to/conch.pth",
  "rank_prompts": "/actual/path/to/VLSA/tools/survival_prompts.json",
  "query_prompts": "/actual/path/to/VLSA/tools/survival_text_prototypes.json",
  "query_key": "tcga_brca_0",
  "num_time_bins": null,
  "feature_order_provenance": "Describe actual slide and feature row identities",
  "feature_preprocessing": "Describe actual magnification, crop, normalization and output",
  "prompt_provenance": "Fixed author JSON and selected ordered cohort query list",
  "weights_provenance": "Actual native checkpoint and source of acquisition"
}
```

The manifest, prompt files, checkpoint and every ordered feature file are
content-bound. Patient slides concatenate in stable slide-ID order. The default
time-rank count is the author's `ceil(sqrt(TRAIN event count))`. Interior
interval edges use TRAIN event times. The terminal edge is infinity, replacing
the source's full-table max+EPS without consulting held-out follow-up; finite
left-edge coordinates and interval assignments retain the original semantics.

## Preserved mechanisms and training

- Actual full pretrained CONCH text:768hidden,12layers,12heads,32007vocabulary,
  128positions, causal mask and finalCLS pooling. All text weights are frozen.
- Actual source RankPromptLearner with4base ranks, linear interpolation and
  learned rank/context embeddings. Although the recipe declares8context
  tokens, source initialization infers4tokens from its provided phrase; this
  behavior is retained, including the author-checkpoint tensor dimensions.
- Full512 VLFAN visual adapter, prognostic text queries, TaskRes residual0.5,
  fixed cross-attention temperature100 and mean query pooling. BRCA uses10
  queries; the published BLCA example uses12.
- Learned prediction temperature. Literal IFMLE(alpha0,eps1e-7)+SurvEMD(p2,
  raw squared), including the source's second softmax and detached target
  temperature. Incidence is a categorical softmax, not independent hazards.
- Adam lr0.0002/wd0.00001 with the source's zero-decay bias/1D grouping.
  B1 patient forwards,32patients concatenated before each loss/backward/step,
  including the last incomplete window. Default10epochs, final checkpoint,
  no scheduler, gradient clipping, instance truncation or early stopping.
- Complete compact checkpoint with learned parameters, fixed prompt buffers,
  Adam, patient/window/update history and framework RNG. Frozen foundation
  tensors reconstruct strictly from the bound actual checkpoint.

Nondefault `update_patients` is an explicit software/recipe variant and is
reported in the protocol. Small-window unit checks do not prove the original
32-patient native budget. Full native-budget validation must retain32.

## Metrics and explicit ports

The source curve is `max(0,1-cumsum(incidence))`, formed in prediction dtype
before SurvivalEVAL's float64 conversion. Mean survival time uses the original
linear interpolation/tail integration. `vlsa_source_mean_time_c_index` uses the
original Comparable pairs/All ties. The common patient Harrell metric is
reported separately using negative source mean time as risk; it is not renamed
as the author's metric. Undefined source comparable-pair results remain null.

The unified loader uses0workers instead of4 and adds validation diagnostics
that never change final-epoch selection. Omitting the unused frozen CONCH
visual tower changes cold-start RNG allocation. No seedwise identity or paper
score claim is made. The original source did not identify a historical CONCH
asset hash; this benchmark declares one and pins its tokenizer dependencies.

## Open author-example discrepancy

The author supplies a six-parameter learned BLCA checkpoint, a cached
2798x512 CONCH feature tensor and a notebook with rounded predictions.
The first CPU check gave probability0.5367577 for rank0 versus the notebook's
0.4904, with maximum absolute difference0.0463577. The test failed at its
original0.00006 rounding tolerance. This is not a passed reference result.

`scripts/validate_vlsa_author_example.py` verifies each published binary
against its pinned Git object, uses safe tensor loading, reconstructs the full
actual text model and saves failed as well as passed receipts. The mismatch
must be resolved or rigorously attributed before claiming this reference is
reproduced. An independent native H200 run gave the same rank0 probability and
maximum error as CPU (differences between devices only at float32 roundoff),
so this comparison does not support a device-backend explanation. The R2
adapter also fixes per-epoch protocol-file counter refresh; the actual
checkpoint epoch/update histories remain independently checked. Do not increase tolerance or change the model merely to match
the notebook. An author cached example is distinct from independently
processed natural WSI training and from paper metric reproduction.

## Commands

```bash
python -m pip install '.[vlsa]'
milplus setup --method VLSA
milplus demo --vlsa --vlsa-weights /actual/path/to/conch.pth \
  --patients 20 --output /new/path/vlsa_demo
milplus check --data /new/path/vlsa_demo --method VLSA --deep
milplus run --data /new/path/vlsa_demo --method VLSA \
  --config configs/vlsa.yaml --output /new/path/run
milplus evaluate --run /new/path/run
```

Demo feature vectors and outcomes are explicitly synthetic. The full language
checkpoint is real. Clinical/data-derived conclusions require actual data and
separate validation; none are established by these commands.

## R3 tokenizer compatibility finding

Author transformers4.41.2/tokenizers0.19.1 retains the pinned JSON's
ByteLevel `add_prefix_space: true`. The modern4.51.3 loader resets it tofalse
unless explicitly passed, changing all51 original prompt/query token rows.
For example, the first content token of `X.` changes from1410 to94. R3
explicitly binds the original true setting in the original tokenizer factory;
it changes no texts, BPE weights, embeddings or model definitions. Independent
old-stack token oracles verify every original query plus Unicode, leading
spaces, empty input and truncation cases.

The complete unmodified original VLSA/CoCa factory (without importing PLUS),
using author text dependency versions and the actual checkpoint, produced
rank0 probability0.4920045 on the supplied cached example under torch2.5.1CPU.
The adapter under those same original text dependencies is bitwise identical
for all12 probabilities. The0.0016045 difference from the retained notebook
still fails its original0.00006 rounding tolerance; do not describe it as a
passed notebook reproduction. Original requirements list torch1.11.0; that
complete historical environment has not been reproduced.

The author requirements specify torch1.11.0. Its official
[Context.h](https://github.com/pytorch/pytorch/blob/v1.11.0/aten/src/ATen/Context.h)
sets both cuBLAS and cuDNN TF32 defaults totrue. The author-example validator
therefore exposes a **separate** `--legacy-tf32 --device cuda` diagnostic,
records both precision flags, and retains the unchanged0.00006 notebook
tolerance. This diagnostic does not silently change the training precision or
prove that the entire historical software/hardware environment was recreated.
