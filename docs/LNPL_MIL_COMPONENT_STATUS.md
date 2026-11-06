# LNPL-MIL complete software pipeline, pending public-method admission

This document records the frozen parent's154-family/329-profile software proof.
The new public-integration tree contains155candidate families/331profiles with
formal admission still false. Current conditions and pending gates are described
in [LNPL_MIL.md](LNPL_MIL.md); the original numerical evidence is preserved.

The original source at commit `1dc4e703a2b86dd0950e0b75aa7d45337d2127cb` is loaded privately.
The candidate retains all 164 planned method records, 154 existing implemented families,
and 329 existing profiles. No complete LNPL profile has been added.

Seventeen component checks passed, including the original four-layer, 256D,
eight-head, 512D-input, 200-instance TOD model, every literal released-training
gradient and seven original Lookahead/RAdam updates, exact restores on both
sides of the slow-weight synchronization, original HNSW superpatch filtering,
Top-200 ordering and the paper auxiliary-loss interval boundaries.

The [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Shao_LNPL-MIL_Learning_from_Noisy_Pseudo_Labels_for_Promoting_Multiple_Instance_ICCV_2023_paper.pdf)
and [official code](https://github.com/szc19990412/LNPL-MIL) describe distinct
conditions that must remain explicit. The paper has a convolution residual,
a disjoint iterative superpatch algorithm, and an auxiliary distribution loss.
The released active code replaces the convolution input, retains one-shot
representatives, and overwrites the computed auxiliary objective with plain CE.
The current `model()` component is the released architecture, not the full paper
condition. `objective(..., variant='paper_auxiliary')` tests the actual missing
auxiliary mechanism but does not by itself establish the full paper method.

The LPA-supervised full ResNet18, actual raw-patch probabilities, separate frozen
ImageNet features, source/paper preprocessing and complete TOD training now have
executed software lifecycles. Public method registration/CLI, full native
budgets, independent installed-wheel and whole-library/release gates remain. The
referenced SuNCEt/PAWS repository has been separately obtained at
`731547d727b8c94d06c08a7848b4955de3a70cea`; obtaining it is not equivalent to
recovering the unpublished LNPL patch-classifier configuration.

The new `paper_model()` and `paper_disjoint_superpatches()` reference components
implement Eq.2 residual convolution and Algorithm1 disjoint iterative HNSW.
Additional checks preserve full dimensions, compare the literal residual
equations, partition every padded patch exactly once and reject an insufficient
ROI instead of silently replacing it with the unfiltered bag.

## Actual raw/LPA weak stage

`lnplmil_raw.prepare` binds actual RGB224 HDF5 arrays, ordered patch IDs and
level0 coordinates, measured magnification/MPP, observed TRAIN-only patch
annotations, real patient identities and fixed partitions. Held-out observed
patch labels, patient overlap and changed annotation/image bytes are rejected.
Weak classes and the positive-class mapping are explicit; missing patches never
inherit their slide's class as an invented patch annotation.

`lnplmil_weak.fit` runs the complete 11,177,538-parameter binary ResNet18 using
the verified official TorchVision0.6.1 operators and ImageNet weight SHA256
`5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8`.
The weak configuration is a disclosed reference choice: SGD/Nesterov0.9,
LR0.02, 50epoch cosine, batch256, horizontal flip, ImageNet normalization and
minimum observed TRAIN patch CE. This is not claimed as the unpublished original
LNPL weak recipe. A nine-class weak model constructor is available, but this
software pipeline acceptance covers binary slide diagnosis, not COAD survival.

Every batch checkpoint includes all BN/model parameters, actual accumulated
gradients, SGD momentum, scheduler, patch order/offset, Python/NumPy/Torch RNG and
the exact data/source/device/recipe contract. Actual interrupts at1/2/3updates
match uninterrupted two-epoch software checks bitwise. Native50epochs are a
separate pending gate; the short checks do not replace that budget.

`score_raw` exports actual selected weak softmax probabilities and a **separate,
untouched, frozen ImageNet ResNet18 512D branch**. It never substitutes the trained
weak features for ImageNet features. All original RGB, coordinates, patch IDs,
weak checkpoint and source identities are bound into the scored cache.

## Complete SP/Top200/TOD software paths

`lnplmil_tod.prepare_selection` has two explicit conditions:

- `released_source_with_reference_weak`: literal original one-shot HNSW SP,
  minimum-size rule, actual Top200 ordering, released convolution replacement
  and released CE overwrite.
- `paper_reference_with_reference_weak`: disjoint iterative HNSW clusters,
  retained complete superpatches, actual Top200 ordering, Eq2 convolution
  residual and the0.001 four-bin distribution auxiliary CE.

Both retain the full four-layer256D/eight-head/FF1024/512D-input200-instance TOD,
original portable Lookahead/RAdam, two-slide gradient accumulation,
VAL-bag-CE checkpoint selection, maximum200epochs and patience10. CUDA uses
explicit float16 autocast/GradScaler; CPU float32 software checks are separately
labelled. A literal released-training-body oracle validates the actual accumulated
update. Mid-gradient/epoch interruptions1/2/3 preserve every learned/RNG state.
Selected checkpoint inference preserves real patient IDs and exports held-out
probabilities only after fitting is complete.

The software source condition executed all six401-RGB-patch bags. The paper
condition executed actual nine disjoint superpatches per bag, auxiliary-head
gradients, exact recovery and selected prediction equality. These two-epoch
synthetic checks are not native-budget or clinical metric reproduction.

## Preserved failures and numerical scope

The first weak literal oracle retained channels-last input while the adapter
used contiguous NCHW. Exactly equal pixel values can select CPU convolution
kernels with different rounding. Both are now explicitly contiguous NCHW; all
bitwise logits/gradient/SGD assertions remain. The original failed receipt is
preserved. This does not claim to solve the separate historical PMIL global
raw-CPU oracle failure.

The released SP implementation can call `np.stack([])` before its minimum-size
fallback on tied weak scores with no eligible ROI. This is now an explicit
rejection with an original-error cause, not an invented unfiltered fallback.
The failed constant-RGB lifecycle and a new exact rejection test are retained;
the successful lifecycle uses401 varied actual RGB patches per slide.

The current collection contains3,738tests, preserving every prior3,723node ID.
Collection is not a whole-suite pass. There is no complete LNPL profile/family
increment until the remaining public/native/wheel gates are connected and run.
