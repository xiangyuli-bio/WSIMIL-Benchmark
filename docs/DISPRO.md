# DisPro source and input integration — development, not admitted

The DisPro record remains pending. Complete original model kernels, actual
BioBERT assets, explicit incomplete-input contracts and objective/inference APIs
are implemented. The unified **UniPro-P + UniPro-G + MultiPro controller** now
includes all three stages, independent Adam/cosine optimizers, VAL-only teacher
selection and strict continuation. Controller tests pass; the actual complete
network has completed an explicitly short-budget three-stage plumbing test.
The isolated development profile is `dispro_source_repaired`; it is not a
formally accepted release. Full-budget validation is still required.

## Primary sources and actual assets

- Author implementation: <https://github.com/Innse/DisPro>, fixed commit
  `98473003de24e5a8f28143442d3dd48bd159aff1`.
- The author README declares Apache-2.0. This commit has no standalone LICENSE
  blob; that distinction is retained. Original definitions execute from a private
  verified Git checkout instead of copying whole third-party modules into this
  repository.
- Primary paper: <https://arxiv.org/abs/2503.01653v2>, 14 pages, main methods and
  supplementary implementation read. Journal/conference identity: CVPR2025,
  “Distilled Prompt Learning for Incomplete Multimodal Survival Prediction”.
- Exact BioBERT: `dmis-lab/biobert-base-cased-v1.2`, revision
  `67c9c25b46986521ca33df05d8540da1210b3256`.
  `pytorch_model.bin` is 435,783,451 bytes, SHA-256
  `70681f9fcd099d9d3c17680e6b885d986b2684a07eba84673b8fbc2431b0bd25`.
  Its bytes match the Hugging Face LFS object identifier. Config and vocabulary
  have separate checksums. No random or reduced pretrained substitute is used.

The native encoder is 12 layers, 768 hidden channels, 12 attention heads,
3072 FFN channels, 512 positions and 28,996 vocabulary entries. The actual
language trunk has 107,719,680 parameters. The source extracts
`AutoModelForMaskedLM(...).bert` and calls its encoder directly. The unused MLM
head and pooler do not participate in DisPro; the private loader strictly checks
every active pretrained tensor while excluding only these unused source parts
and derived HF position/token-type buffers. Asset loading uses `weights_only`
and does not execute arbitrary serialized model code.

Read-only pretrained weights are shared between the three source call sites;
the complete native layers and every forward invocation are retained. No model
depth, width, pathway encoder or attention sequence is shortened for the audit.
CUDA portability and the full end-to-end recipe still require validation.

## Complete source kernels already executed

The original factories retain the WSI linear/ReLU adapter, all per-pathway
768→768 SNN blocks, the two-layer genomic Transformer, class-specific prompt
learning, full BioBERT text paths, top-256 pooling, UniPro cross/self scoring,
the 512-token multimodal sequence, missing-modality placeholders, fused survival
and missing-modality UniPro distillation objectives.

The public pathway file contains **331** columns. The paper reports **330**
pathways after cohort preprocessing. The source dynamically skips empty pathway
intersections. Without the original cohort we cannot identify a historically
removed pathway and will not arbitrarily delete one. The native mechanism audit
uses all 331 published groups with their actual sizes (9,745 memberships), full
768-channel SNNs and a complete two-layer Transformer. A real task must explicitly
register its cohort's final ordered gene/pathway mapping.

Actual source-based CPU checks include UniPro-P and UniPro-G forwards,
backpropagation and Adam updates; MultiPro fused plus missing-modality losses;
and full-language inference with complete and incomplete inputs. Synthetic
mechanism checks do not establish complete training or original paper scores.

## Source findings and integration rules

| Source behavior | Evidence and required treatment |
|---|---|
| `.data` is passed to two placeholder optimizer groups | Actual placeholders receive nonzero gradients but the original optimizer does not update them. Binding real Parameters restores updates. |
| `cls_token` is excluded by the multi-LR replacement | Its gradient is nonzero and it stays unchanged under that optimizer. The named repaired optimizer includes it. |
| The engine replaces the optimizer after constructing its scheduler | The scheduler retains the old optimizer. A completed adapter must bind the actual stage optimizer; source behavior is documented separately. |
| Multi-LR Adam omits weight decay | The new optimizer defaults to zero decay, despite the earlier optimizer's 1e-5. This cannot be silently described as the same recipe. |
| Binary 2-D masks are passed directly to `BertEncoder` | They become additive scores rather than proper padding exclusion. Actual padded-key attention is nonzero in the source and exactly zero under the named 4-D additive-mask repair. |
| Self-scoring heads only determine hard top-k indices | The native gradient audit finds no loss gradient in these heads. No invented straight-through estimator is added. |
| Prompt context is shared between censored/uncensored descriptions | Source contexts are `(4,256,768)` and `(4,257,768)`; the paper describes eight groups and lengths255/256. Preserve and label the released architecture; do not assert these are identical. |
| UniPro validation chooses prompts using true censorship | The reference prediction API accepts no outcomes and uses fixed uncensored risk descriptions. This is a named outcome-free selection port, not literal validation-score reproduction. |
| MultiPro inference uses `label=None` | It scores all eight prototype descriptions. Actual fused risks are unchanged when the raw auxiliary censorship argument changes. The public wrapper supplies a constant and accepts no event/time/bin. |
| Dataset fallback time bins use all input rows | Only the fallback code is established; supplied historical label provenance is unverified. The complete trainer must fit bins on observed TRAIN event times only. |
| Missingness is inferred from zero-valued arrays | An observed all-zero assay is valid input under the new explicit presence table. Absence remains `None`; no synthetic WSI bag is created. |
| Pathway `.isin(...)` preserves each patient's CSV row order | The new input contract binds an explicit ordered gene list and ordered pathway indices, avoiding unverified cross-patient permutations. |

UniPro-P's public launch script specifies30 epochs. UniPro-G's script says30 but
`main.py` overrides it to50 at LR1e-5. MultiPro specifies50 epochs and multi-LR.
The paper's supplementary defaults are30, with BioBERT prompt models trained50.
The completed stage controller must expose its chosen source/reference recipe
and all three full stage budgets rather than silently selecting convenient
values. Batch size is1, and the native frozen language trunk stays complete.

## Prepared patient inputs

The ordinary task remains patient-level survival with declared time units.
Set `encoders.<FM>.dispro.manifest` to a JSON object:

```json
{
  "schema_version": 1,
  "kind": "dispro_incomplete_patient_inputs",
  "synthetic": false,
  "presence": "presence.csv",
  "expression": "expression.csv",
  "pathways": "pathways.json",
  "gene_columns": ["GENE_A", "GENE_B"],
  "normalization": "none",
  "missingness_provenance": "Actual observed or explicitly simulated missingness policy",
  "assay_preprocessing": "Actual RNA assay transformation before this bundle",
  "pathway_provenance": "Actual cohort-filtered signature source/version"
}
```

`presence.csv` contains exactly one row per global patient and literal0/1
`wsi_observed` / `omics_observed` fields. Both cannot be0. The expression table
must contain every observed RNA patient; genuinely absent assays need not have
a row. An observed all-zero row is retained as observed. `pathways.json` is an
ordered list of `{name, features}` mappings with real, unique, present genes.
Nothing silently filters missing genes or generates molecular values.

All actual slides of a patient are combined. A patient with no WSI may have an
explicit metadata-only patient slot whose feature-index URI is
`missing://wsi/GLOBAL_PATIENT_ID`; that URI is **not an image or array**.
`Bundle.load` rejects it, and inventory records it as absent. Alternatively,
explicit simulation flags may mask an existing file without reading it as
model input. `slides` counts observed WSI records; separate input-slot, patient,
observed-modality and patient-subset counts prevent absence slots becoming
imaginary WSIs in benchmark statistics.

RNA normalization is either declared upstream values (`none`) or explicit
`train_standard`, fitted only on observed TRAIN patients. Feature names, gene
ordering, pathways, missingness, files and statistics are bound into the input
contract. The model input dictionary contains only WSI features and pathway
values; event/time/bin labels are excluded. Methods requiring complete features
cannot silently consume a DisPro incomplete-input bundle.

## Three-stage controller and current evidence

`dispro.py` retains the author's full networks and explicitly budgets all three
phases. The released schedule is **30 WSI + 50 RNA + 50 MultiPro epochs**. RNA's
50 comes from the actual override in `main.py`, rather than its launcher's
apparent default. There is no common early-stopping shortcut. Shorter schedules
must be explicit in `stage_epochs` and cannot substantiate default-budget claims.

Each UniPro stage uses only TRAIN patients with that modality observed. Both
teachers are selected by their own observed VAL population using outcome-free
inference; the selected weights, rather than the final epoch, enter MultiPro.
All actually available modalities participate in the final joint stage. Both
missing-WSI and missing-RNA distillation losses remain active when applicable.
No UniPro epoch can become the final MultiPro `best.pt`.

The repaired MultiPro optimizer owns real holder Parameters and CLS, alongside
the source modality adapters/scorers/heads. Its native group rates remain
2e-4/1e-5 and its actual source replacement Adam has decay0. UniPro decay remains
1e-5. Each cosine scheduler belongs to the optimizer that really steps. Hard
top-k score heads still have no loss gradient; there is no invented estimator.

Four survival intervals are fitted only to observed-event TRAIN patient times.
The selected C-index follows the source's right-censored semantics: event/censor
time ties are comparable, two tied events are not, and risk differences <=1e-8
are ties. Its independent pair enumeration matched the actual scikit-survival
v0.24.0 primary iterator/estimator on 973 finite randomized cases; 27 undefined
cases were refused. This named `dispro_c_index` is separate from the library's
existing strict-time summary `c_index`.

Checkpoint state includes learned weights, both selected teacher snapshots and
their digests, all stage counters/orders, active Adam moments, all cosine states
and engine RNG. Once pretraining ends, selected teachers are loaded into the
actual MultiPro modules, and completed-stage Adam moments are released because
those optimizers will never step again. Frozen BioBERT weights are reconstructed
from the exact verified public asset; only those identifiable shared constant
weights are omitted from checkpoint payloads. Every other state remains strict.

Thirty controller-only tests use explicitly named tiny test doubles to
exercise lifecycle rules economically. They are not model or native-budget
evidence. The tests prove bitwise continuation at all five boundaries of a
2/2/2-stage fixture, including both stage transitions, actual Parameter ownership,
VAL-only selection and refusal of corrupt identities/counters/teachers. Combined
with40 input/objective cases, five asset-IO cases,20 training regression cases
and seven CLI regression cases, **102 checks pass**. Separate full-source tests execute the actual 12-layer
BioBERT and native networks; their input sizes and budgets must be read from
their individual receipts, not inferred from these controller tests.

Declared reference ports include deterministic VAL order, num_workers0 data
reading and one unified construction/RNG stream. These make the full unified
workflow deterministic without claiming identical random-call order to three
independent source launchers. Source prompt constants and attention-mask aliases
are registered buffers for device transfer. An H200 original-network short-budget check completed all three1/1/1 stages,
real Adam updates6/6/8, three selected-state/optimizer reloads and final TEST
predictions in16.55seconds using TWO synthetic pathways. It verifies the device
port and cached-GPU prompt reconstruction, not the full331-pathway130epoch gate.

## Remaining admission work

The actual full-network1/1/1-stage test completed343.05seconds with6/6/8 real
Adam updates, all three saved stage checkpoints reloaded, both missing-modality
losses exercised and final TEST predictions exported. A subsequent current-code
strict-loader check validated all actual Adam states/schedules and reproduced
the TEST CSV byte-for-byte. Its two synthetic pathway groups do not stand in
for the full331-pathway budget. The unified engine also passed interruption and
bitwise resume tests at both stage boundaries using explicit controller doubles.

Validate full source budgets at native channels and
the full declared pathway set for both1024/1536 FM widths, fresh whole-library
and all-profile runs, isolated final-wheel selected-checkpoint reload and the
full164-entry source archive. DisPro remains pending until these are achieved.

## Development entry points

```bash
python -m pip install '.[dispro]'
milplus setup --method dispro
# On a networked CPU host, fetch and checksum the exact complete public asset.
milplus prepare-dispro-assets --output /path/to/reviewed/dispro_biobert
export MILPLUS_DISPRO_BIOBERT=/path/to/reviewed/dispro_biobert
milplus demo --dispro --patients 30 --output /path/to/new_synthetic_bundle
milplus run --data /path/to/prepared_incomplete_survival_bundle --method DisPro \
  --config configs/dispro.yaml --output /path/to/new_run
milplus evaluate --run /path/to/new_run
```

The demo uses all331 public group sizes with explicitly synthetic disjoint gene
names and random molecular values; it is not real TCGA. Real tasks must register
their actual ordered genes/pathways and assay provenance. The default command
retains all130 training epochs and native networks. The separate all-profile
software fixture explicitly uses two synthetic pathways and2/2/2 epochs; it
cannot serve as complete-budget evidence.

The current focused+training+CLI check set contains102 passing cases, including
five isolated asset-IO checks. Asset download uses the exact immutable public
revision and three SHA-256 values, refuses wrong existing files/revisions, and
publishes only verified bytes. These network-response test doubles are IO
checks; actual model evidence uses the435,783,451-byte public pretrained file.

The model rebinds every stage optimizer list and original direct Parameter alias
after device conversion, including PyTorch policies that replace Parameters.
A regression forces that policy and executes all three stages. Device transfer
after optimizer construction is refused. Selected `best.pt` stores evaluation
weights/provenance; exact resumable Adam/RNG state stays in `last.pt`, avoiding
duplication of large genomic moments in inference checkpoints.
