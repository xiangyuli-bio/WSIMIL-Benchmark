# TaxoMIL

Official source: [QuIIL/TaxoMIL](https://github.com/QuIIL/TaxoMIL), commit
`5e095f0c212b3a30c3f6c6c81b3ca1a18e0c676e`.
Primary protocol: [paper, Appendix D](https://arxiv.org/html/2606.31100v1).
The pinned source has no LICENSE. Fetch and execute author definitions externally;
the package does not redistribute them.

The complete architecture retains the actual pretrained GPT-2 small
(124M backbone, 12 layers, 768 hidden, 12 heads), two additional GPT2Block heads,
independent output projections, native1024D gated MIL, the512/512 visual split,
two adaptors, learnable image token, mean-pooled label contexts and ten prompts.
Both `[PAD]` and `<|img|>` are added to the real tokenizer, and the source's
required output-vocabulary repair is executed after model construction.
The complete resulting model has over200M parameters. No small/random language
model replaces the original pretrained backbone.

Training executes the pinned generation CE, text hierarchy objective, both
image/text alignment losses and both supervised contrastive losses. Source
batching pads complete variable-length bags with zeros without masking attention;
no patches are sampled/truncated. Batch64, four readers, shuffled TRAIN,
sequential VAL, AdamW/wd.01, cosine LR/eta1e-7 and three-cycle .3/.3/.1 weights
are retained. Default maximum100 epochs and patience20 use the first strict
improvement in the mean coarse/fine exact-match percentages. TEST access occurs
only through explicit evaluation.

Two explicitly named conditions:

| Profile | LR | Meaning |
|---|---:|---|
| `taxomil_gpt2_all_losses_key_repaired` | 1e-5 | Released CLI default; `TaxoMIL` alias |
| `taxomil_gpt2_paper_lr_key_repaired` | 1e-4 | Paper Appendix D BRACS/PANDA LR; other source objectives unchanged |

Required repair: the author runner creates `c_ital`/`f_ital` but its alignment
helper indexes `coarse_ital`/`fine_ital`, so `--use_all_loss` raises KeyError.
The adapter explicitly maps those two lookup keys; loss equations and weights
are unchanged. An independent source oracle keeps the original helper unchanged
and supplies the corresponding criterion aliases. Failed original attempts and
their exact code, logs and inputs are retained.

Source/paper discrepancies remain disclosed. Source sibling-margin loss uses
raw detached token embeddings with a .5 factor; paper Eq6 uses projected label
embeddings. Source label embedding pooling for auxiliary objectives is no_grad,
while its projection still receives gradients. Source validation uses a text
bank snapshot built before TRAIN. Its periodic scheduler normalizes to100 even
for explicitly shorter software checks. Source default LR differs from Appendix
D. These conditions are not a claim that the paper's reported results have been
reproduced.

Install the `taxomil` optional dependency group in the chosen environment.
Prepare actual assets on a networked CPU host:

```bash
milplus setup --method TaxoMIL
milplus prepare-taxomil-assets --output /your/owned/storage/taxomil-gpt2
```

The complete public GPT2 revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` is checked by fixed sizes and hashes.
The safetensors SHA256 is
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
Both full pretrained weights and the tokenizer/config files are required offline.

In an existing classification data bundle, declare an explicit `hierarchy`
(the same fine classes and clinical parent mapping as the labels), and:

```yaml
encoders:
  uni:
    dimension: 1024
    index: features.csv
    # Retain the other required encoder provenance fields.
    taxomil:
      assets: /your/owned/storage/taxomil-gpt2
      dataset: BRACS   # or PANDA; controls the original organ prompt and case rules
```

Then run:

```bash
milplus run --data /your/task --method TaxoMIL --output /your/run --device cuda:0
milplus evaluate --run /your/run --subset test --device cuda:0
```

The original model assumes1024D. Other feature dimensions are rejected clearly;
no hidden projection changes the original method. The adapter requires explicit
slide training/evaluation and patient-grouped splits. It does not invent labels
for TCGA-BRCA/COAD from cancer project names.

Outputs are actual free-generated coarse/fine diagnosis strings. Exact match
counts unknown, prefix-only and semantically similar strings as incorrect.
Report coarse, fine and holistic accuracy, weightedF1 and kappa. Do not fabricate
class probabilities or AUROC from generated vocabulary logits.

Full model, all AdamW moments, cosine position, sampler orders, actual per-parameter
update counts, RNG and source selector history are restored. Independently reload
selected checkpoints twice before admitting the method. Source-oracle software
checks use explicitly synthetic full-width bags plus real GPT2; real WSI and
paper-score experiments remain separately unrun.

## Version0.88 exact percentage regression

Full paper-LR BRACS validation in frozen0.87 failed at epoch11 with four
correct generated coarse labels among14 slides. The original source computes
`correct / total * 100`, whereas0.87 computed `100 * correct / total`. The
algebraically equivalent expressions differ in their last floating-point bits.
Version0.88 restores the literal division-before-percentage order; source VAL
equality remains exact, with no tolerance relaxation. A separate regression
executes the literal pinned `trainer/test.py::calculate_metrics` for both levels
and verifies exact equality. Frozen0.87 failures, inputs and checkpoints remain
unchanged. Four full native conditions must be rerun for the corrected wheel.

The native validation fixture also explicitly supplies12 TRAIN patients per
fine class for PANDA (72 total), so batch64 has a real second batch. The original
BRACS fixture remains10 per class (70 total). These are synthetic software
fixtures, not a change to the clinical protocol or source100max/patience20 budget.

## Actual BRCA binary-task precondition check (2026-11-06)

The unchanged `TaxoMILInputs` was invoked against the actual856-patient
IDC/ILC slide bundle. It rejects the absent coarse/fine taxonomy before loading
GPT2. Existing language-model assets alone cannot supply clinical hierarchy.
The current shared hierarchy interface requires at least two covered coarse
classes. With only two fine classes, that is a bijection, not an independently
provided nontrivial taxonomy. In-memory synthetic one-parent/bijection controls
were explicitly diagnostic only and never written into the real task.

The original BRACS/PANDA prompt domain must also be selected explicitly;
TCGA-BRCA is not silently labelled BRACS. No extra labels, ordinary bag-classifier
substitute, GPU model or training were created. This remains a prerequisite
blocker, not an extra exclusion beyond the user's25 task-mismatched methods.
Shared evidence:`brca_hierarchy_and_ahdmil_actual_scope_r1.json`.
