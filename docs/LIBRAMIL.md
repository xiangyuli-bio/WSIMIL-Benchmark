# Libra-MIL

Original source: <https://github.com/zfy07/Libra-MIL>, commit
`f29921970bfa22d364e08ea1768b984b47e07a8d`. The pinned tree has no LICENSE;
author definitions are loaded privately and are not redistributed here.

Two complete profiles are implemented: `libramil_source_p4` (the source default
four visual prototypes) and `libramil_readme_p6` (the README's six prototypes).
`Libra-MIL` resolves to the first. They retain the original native 512D
projections, eight-head attention, learned structural/class prompts and visual
prototypes, 20-iteration Sinkhorn transport (epsilon .05), patch weighting with
temperature .2, and the complete class-query classifier and CE loss.

The text preparation uses the actual full pretrained CONCH-v1 text tower,
normalized CLS pooling and the original tokenizer. All 150 text tensors are
loaded strictly from the registered checkpoint. Encoding is performed once
under `no_grad`, as in the source; learned copies of the embeddings are updated
during training. The unused image/decoder allocations are omitted. Their RNG
initialization is not reproduced; no seedwise cold-start factory claim is made.

The original RCC assets comprise 46 structural descriptions and three
high-resolution class descriptions, ordered KICH, KIRC, KIRP. Clinical data
must explicitly use that order for the RCC preset. The README's illustrative
`--num_classes 2` is inconsistent with the released three-class RCC prompts.
Other classification tasks can provide complete ordered structural/class
descriptions with explicit provenance; they are not silently given RCC prompts.

The source recipe is AdamW at LR .0001 with its actual default weight decay
.01, per-update cosine scheduling after 100 warmup updates, batch 1 and 20
epochs, with full bags and no clipping or early stopping. `temp_struct` and
`temp_vis` remain allocated and unused exactly as in the released forward.

The released checkpoint code has a quirk: a strict validation accuracy gain
does not update `best_auc`. A subsequent accuracy tie can select a lower-AUC
checkpoint. These profiles preserve the literal branch and save its full state;
they do not silently present a corrected selection rule as the original method.

TEST is locked until explicit evaluation. The original training entry evaluates
TEST every epoch, though it does not select by TEST. This port retains only the
unused sequential TEST DataLoader's CPU base-seed RNG draw during training.
It reads no TEST examples or labels for training or checkpoint selection.

For an explicitly synthetic software demonstration:

```bash
python -m milplus demo --libramil \
  --libramil-weights /absolute/path/to/conch.pth \
  --patients 30 --output /tmp/libra_demo
python -m milplus run --data /tmp/libra_demo --method Libra-MIL \
  --output /tmp/libra_run
```

Real tasks require a standard patient-safe task bundle and a metadata JSON.
`prepare-libramil --data TASK --metadata META --output INPUT_VERSION/manifest.json`
encodes the full language descriptions and binds all input paths, bytes, labels,
splits, native alignment, class order and provenance. Then set the encoder's
`libramil.manifest` field to that manifest. Generic FM features do not acquire
CONCH text alignment merely by having 512 dimensions.

Source checks include actual full pretrained language encoding, complete
native512/46description/257patch models, all gradients, exact two-epoch original
training-loop/AdamW/scheduler/RNG equivalence, and the source selection branch.
Both full 20-epoch CPU source references have passed (360 updates each), with
complete state restoration and two independent selected-checkpoint reloads.
Unified CLI, installed wheel, complete regression, CUDA and real WSI evidence
remain separately tracked. These software/native-dimensional checks do not
constitute paper-score reproduction or clinical performance validation.
