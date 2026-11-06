# DyKo code integration and validation boundaries

Pinned author source: <https://github.com/junjianli106/DyKo>, commit
`a8b43645840cf283c334b07c42cf4ddf230f644a`.

The eight original-model execution profiles are registered as
`adapted_under_verification`; see [public usage](LAST_THREE_METHODS.md).
This is code availability, not full-method validation. The original TITAN language model is required.
The existing CONCH-v1 weights are not a replacement for that model. Authorized
TITAN model/config/tokenizer assets must be supplied and verified locally;
registration does not grant access. Earlier asset-access checks are historical.

The new `dyko_source.py` loads the actual author definitions in a private
namespace. It preserves the two 768→192→768 adapters, 16 learned prompts, two
768D/8-head attention modules, visual clustering, confidence-based knowledge
selection, temperature .01 retrieval, cluster heads and original CE+KL loss.
There is no third-party source redistribution; the pinned tree has no LICENSE.

All four released concept banks contain 1,000×768 embeddings. The prompt files
contain 2/2/3/5 ordered class descriptions for CAMELYON16/NSCLC/RCC/UBC. This is
inconsistent with the README's illustrative 512D input. The model's default
input is 768D and its released YAML does not override that dimension.

The source normalizes `[1,N,10]` cluster assignments over `dim=1`. Its structural
loss is unweighted `kl_div(image.log(), text, reduction='batchmean')`. Changing
either normalization would change the released method; both remain intact.
Every forward clusters detached visual features with FAISS KMeans, 20
iterations, minimum-points setting 5, seed 42. The native route explicitly
requires CUDA-enabled FAISS; CPU numerical checks have a separate backend name.

The classification recipe is Adam (bias/1D zero decay), LR .001, other-weight
decay .00001, batch 1, accumulation 4, FP32, up to 1,000 epochs, patience 40,
validation **standard slide accuracy** selection. It has no scheduler or
gradient clipping. The source's exact Lightning accumulation boundary/RNG and
full pretrained text training still require reference verification.

Thirty actual source-component checks pass, including all four full concept
banks; all twelve released 4/8/16-shot seed-1024 CSVs; full 768D retrieval values
and feature/concept gradients at 7 and 257 patches; original CE+KL gradients;
Adam states; and explicit rejection of missing pretrained assets or GPU FAISS.
These checks do not prove a full TITAN forward, end-to-end training, native GPU
budgets, clinical evaluation or paper-score reproduction.

Source shot caches do not bind their input split digest. A benchmark manifest
must bind patient identities, split bytes and selected TRAIN IDs before real
experiments. Source slide CSVs alone cannot prove patient separation.
