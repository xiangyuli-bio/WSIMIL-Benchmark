# SLIP: original prompt learning with native CLIP features

This is the isolated 0.66 development implementation. It is not a formal
admission or a reproduction of clinical/paper scores. The accepted main release
and its evidence remain separate until all current package gates pass.

## Original source and preserved mechanisms

Official source: <https://github.com/LTS5/SLIP>, fixed at
`9351c4480ec8d7a8216316a1861025b4f086ac78`.

The adapter privately imports the actual author `PromptLearner`, `TextEncoder`,
SLIP pooling and contrastive objective. It retains class-specific learned context,
fixed tissue text, the native text transformer and class-correlation matrix.
Only prompt context is optimized. In particular, it does not replace the
original objective with ordinary bag cross entropy.

The two explicit profiles use the original CLIP branches:

| Profile | Patch embedding width | Text transformer | Original checkpoint SHA-256 |
|---|---:|---|---|
| `slip_clip_vit_b16_source_repaired` | 512 | 12 layers, width 512, 8 heads, context 77 | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |
| `slip_clip_rn50_source_repaired` | 1024 | 12 layers, width 512, 8 heads, context 77 | `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762` |

Every text parameter is strictly loaded from the actual full published model.
The image encoder is unused when reading frozen patch embeddings. Removing its
unused allocation does not reduce the text network. The vocabulary has 49,408
entries and the text projection retains the actual native output dimension.

All tissue aliases and templates are individually encoded and normalized, then
averaged and normalized within each tissue. The original slide prompt uses the
first template and first description of each ordered class. The source computes
patch-to-tissue and tissue-to-class attention, and deliberately detaches attention
and the pooled class representations. The final class correlation matrix still
propagates gradients into learned class text. Its contrastive target is the
`y * C + y` entry among all `C * C` entries. Predictions use the diagonal logits.

## Complete recipe and explicit ports

The default is the original 10 epochs, batch size one, SGD learning rate 0.002,
zero momentum, zero decay, no scheduler, no clipping and complete bags. Default
context size is one, initialization gain 0.01 and temperature 0.01. One/two-shot
training is sequential; other modes shuffle. All trained bags and updates are
recorded. Selection always uses the final epoch; validation diagnostics do not
select a preferred earlier epoch. The test split is evaluated after training.

The public repository imports an absent `datasets/tcga.py`, and its few-shot split
text files are not distributed. The benchmark therefore supplies its own
patient-safe loader and an explicit, already prepared TRAIN-only few-shot split.
`n_shots` must agree exactly with TRAIN slide counts. No TEST/VAL selection is
performed to create few-shot training data.

Ports are named rather than attributed to an unchanged source implementation:

- CPU/device-safe tokens and buffers; singleton bags preserve their `N,D` shape.
- Float32 pretrained text weights, following the original CLIP CPU loading
  branch. Equality to the author's CUDA-half computation is not claimed.
- Current learned evaluation text is recomputed instead of keeping a mutable
  cache. This prevents stale class weights after an optimizer update.
- Worker count zero and the common run RNG stream, including validation loader
  initialization. There is no claim of original paper seed identity.
- Multiclass and patient reporting use the common benchmark evaluator; original
  binary-specific reporting code is not treated as a multiclass specification.

Only native aligned CLIP features are admitted by these profiles. A 512/1024-wide
feature from another FM is not interchangeable merely because its shape matches.
Other VLM branches in the source are not silently redirected to these weights.

## Preparing and running

Install the optional tokenizer dependencies and fetch the fixed author source:

```bash
python -m pip install 'mil-baseline-puls[slip]'
milplus setup --method SLIP
milplus prepare-slip-assets --output "$HOME/.cache/milplus/assets/slip_clip"
```

An alternative asset directory is selected with `MILPLUS_SLIP_CLIP_ASSETS`.
Downloads are verified against the exact original SHA-256; existing unverified
files and symlinks are not replaced. Weights are not included in this package.

The task encoder declares its native `model`, `weights_sha256`,
`feature_preprocessing` and `slip: {manifest: slip_inputs.json}`. The manifest
contains ordered `classes`, `slide_classnames`, `tissue_classnames`, `templates`,
native architecture/weight/preprocessing identity, `n_shots`, and provenance for
prompts and the prepared few-shot subset. Its synthetic marker must match the
task. The manifest participates in the dataset fingerprint and strict checkpoint
binding.

After this one-time preparation:

```bash
milplus run --data /path/to/prepared_task --method SLIP --output /path/to/new_run
milplus evaluate --run /path/to/new_run
```

The family alias resolves the declared native architecture. Both explicit profile
names are also supported. For a permanently marked synthetic software fixture:

```bash
milplus demo --slip --slip-architecture clip_vit_b16 --patients 20 --output /tmp/slip_demo
milplus run --data /tmp/slip_demo --method SLIP --output /tmp/slip_run
```

The run preserves learned prompts, original fixed token/tissue buffers, SGD,
sampler orders, counters and RNG. Frozen text parameters are reconstructed from
the exact verified asset. Missing/extra states, altered fixed buffers, changed
inputs, nonfinite learned prompts, wrong optimizer settings and inconsistent
TRAIN histories are rejected. Moving a model after constructing its optimizer
is rejected to preserve parameter references.

## Required acceptance evidence

The development tests exercise actual native pretrained text for both profiles,
the original contrastive loss and gradients against an independent formula,
detached attention, prompt-only updates, singleton bags, byte-identical resumed
training, final-epoch selection and actual public CLI execution. Small asset IO
doubles test publication/refusal behavior; they are not pretrained-model evidence.

Formal acceptance additionally requires both independent installed-wheel
10-epoch native budgets, the fresh complete current test suite and every current
execution profile, final-package checkpoint reloads, the 164-spec sdist audit,
and integration on the exact accepted predecessor. Natural WSI results, paper
scores, and coverage of every VLM architecture remain unclaimed.
