# ConMIL: complete semantic attention, original KAN and training recipe

Author source: https://github.com/anxuanhan/ConMIL,
commit `aabb5386d15a6002a3a39437ea46c0d99d29a2f3`, MIT.
The original definitions are privately loaded from verified Git objects.

Three routes are registered:

| Profile | Meaning |
|---|---|
| `conmil_pos_source` | Original binary positive-query condition; negative queries may be absent |
| `conmil_posneg_source` | Original binary positive-minus-negative condition |
| `conmil_posneg_multiclass_repaired` | Explicit C-way label/evaluation and first-checkpoint extension |

`ConMIL` resolves to the original POS+NEG route for two classes, or the explicitly
named C-way extension for more classes. Missing negative queries never silently
turn a POS+NEG request into a positive-only experiment.

## Mechanisms and source conditions

The complete native network is retained:512D projected CONCH image embeddings,
512→256/ReLU/dropout0.25,128D gated attention, an unconstrained learned alpha
and1×1 similarity projection, and a256→128→C KAN with original cubic B-splines,
LayerNorm and dropout. Semantic guidance is mean positive similarity minus
0.5 times mean negative similarity, added before patch-attention softmax.
No MLP replacement, frozen attention, auxiliary-loss deletion or smaller model
is used. The source loss is CE alone.

Actual text preparation executes the author `encode_texts` using the full
pretrained native CONCH text tower:12layers/768width/12heads, causal and padding
masks, appended CLS,512D projection and L2 normalization. It executes the
original CoCa text methods and local tokenizer. The unused vision/decoder
allocations are omitted. It does not use FOCUS's different unmasked/BOS path.
The original pretrained weight SHA is
`40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c`;
CONCH source is pinned to`141cc09c7d4ff33d8eda562bd75169b457f71a62`.
Weights are separately supplied and are not redistributed.

Real visual features must come from the same projected, normalized CONCH space;
equal vector dimension from an unrelated FM does not establish alignment.
Synthetic software fixtures are separately marked and make no natural visual/
text alignment or real WSI claim. The task, exact query wording/order, actual
text tensors and their provenance, visual bytes, patient mapping and split are
bound in immutable manifests. Data-driven prompt review must identify TRAIN
patients only. Predefined queries explicitly declare that no data-driven
refinement was performed; expert approval is never invented.

The original trainer defaults to50epochs, Adam2e-4/weight_decay1e-5, cosine
T_max=epochs, batch1, no clipping or early stopping. TRAIN alone randomly caps
bags at4096patches; VAL and TEST retain all patches. The original sequential
and shuffled DataLoader RNG semantics are retained, with workers0 as an
explicit epoch-boundary recovery port. The README's20epoch/grid6 example is
an optional parameter condition, not the script's default. Native grid5 and
explicit grid6 are supported.

## Source issues and explicit ports

The published data branch converts`type == tumor`to1 and everything else to0;
its evaluation is binary despite the README mentioning BRACS. Source profiles
therefore require two classes and explicit negative0/positive1 roles. The
C-way extension uses the actual task class IDs and multiclass evaluation.

The source initializes best validation accuracy to0 and replaces it only on
strict improvement. Source routes preserve this rule, including its missing
best-checkpoint outcome if every validation accuracy is0. The named C-way
extension accepts its first finite checkpoint, then uses the same strict tie
rule. Neither route substitutes AUC, CE or test-side model selection.

The original KAN registers its knot grid as a zero-stride`expand`view. Loading
checkpoint tensors with separate storage raises a PyTorch overlapping-memory
error. The adapter materializes an identical writable grid. It changes no knot
values, spline equations, initialization draws, trainable parameters or losses.
Actual source/ported outputs and gradients are checked bitwise. All network
weights, fixed text embeddings, grids, Adam moments, cosine state, train order,
sampled indices and global RNG are retained and checked on resume.

## Use

```bash
python -m milplus setup --method ConMIL
python -m milplus prepare-conmil-text --prompts prompts.json \
  --weights /own/checkpoints/conch.pth --output /own/conmil-text --device cpu
python -m milplus prepare-conmil --data /own/task \
  --metadata /own/conmil_metadata.json --output /own/task/conmil/manifest.json
```

`prompts.json` contains ordered`positive`and`negative`query lists,
`task_semantics`,`prompt_provenance`, an explicit boolean`synthetic`marker and
`review`. Review declares`selection_subset: train`, either
`predefined_without_data_refinement`with empty patient IDs, or
`train_only_patch_retrieval`with actual global patient IDs, provenance and a
retrieval-receipt SHA. Positive-only experiments may provide an empty negative
list. A model never generates clinical prompts or claims expert review itself.

The binding metadata records schema_version1,
kind`conmil_original_semantic_attention_assets`, ordered classes,
`text_assets`pointing to prepared`text.json`, alignment_policy`native_conch`,
visual preprocessing/provenance, task semantics and ordered class roles.
For binary tasks the roles are`negative`,`positive`; C-way tasks explicitly
use`task_class`for each class. The visual encoder records
model`conch_ViT-B-16`, the actual weight SHA and normalization`L2`.

After preparation, add`encoders.<FM>.conmil.manifest`to the task configuration:

```bash
python -m milplus run --data /own/task --method ConMIL \
  --config configs/conmil.yaml --output /own/run --device cuda:0
python -m milplus evaluate --run /own/run --subset test --device cuda:0
```

For software-only examples, prepare actual text assets first, then use:

```bash
python -m milplus demo --conmil --patients 20 \
  --conmil-text /own/conmil-text/text.json --output /own/conmil-demo
```

The fixture marks synthetic visual data and
requires text assets marked for software use. The fixed author-query cache is
not clinical validation. Tests, native budgets, whole-repository checks, final
package acceptance, real WSI experiments and paper scores are distinct stages.
