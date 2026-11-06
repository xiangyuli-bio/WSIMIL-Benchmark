# MExD complete two-stage reference

This is an explicitly named reconstruction of the complete paper mechanism,
including Dyn-MoE training and conditional diffusion training/inference. It is
not the released incompatible trainer executed unchanged, and it is not a
paper-score reproduction.

Original repository: [JWZhao-uestc/MExD](https://github.com/JWZhao-uestc/MExD),
commit `1d4d438e2b19d0fee9cb55328faa84b6ca37fe0b`. The author TransMIL,
conditional denoiser, shifted diffusion operators and RAdam definitions are
loaded privately from verified Git objects. No repository LICENSE was present;
the upstream code and private pretrained checkpoints are not redistributed.
Nyström attention is explicitly bound to the bundled TransMIL reference
implementation, not claimed to be an unidentified historical package version.

The released root MoE returns three values but its trainer expects two. Its
classification counterpart has a different router count for Camelyon16, and
the pretrained branch first loads an unrelated missing hardcoded gate file.
Consequently this adapter unifies the graphs using paper equations 2, 3 and 9.
It trains both complete TransMIL adapters from scratch on the actual registered
frozen features; no absent private pretrained MoE is replaced with a claim of
author initialization. This initialization is an explicit reference condition.

The reference retains two full 512D TransMIL adapters, two 8-head layers each,
256 landmarks, six pseudoinverse iterations, residual convolution 33 and PPEG
kernels 7/5/3. Binary expert routers use hard argmax and top-k selection
(negative .25, positive .5). There is no task gradient for the router parameters
through this literal hard selection; no soft or straight-through surrogate is
inserted. Expert confidence weights are summed without renormalization.

Internal TransMIL pseudo-grid padding is preserved. Duplicated grid rows are
excluded from original-instance routing identities. Empty experts use a zero
sentinel, excluded from actual-patch counts; undefined empty-expert CE is
skipped. Stable ties, the sentinel convention, stored patch order, zero loader
workers and fixed final-epoch selection are declared reconstruction choices.

The complete default recipe is 100 MoE epochs with original RAdam (2e-4,
weight decay 1e-5), followed by 200 diffusion epochs with Adam (1e-3,
betas .9/.999, eps 1e-8, no weight decay, norm clipping 1). Each stage has its own
cosine scheduler. Stage one trains the prior CE, negative-expert CE and
bag-matched positive-expert CE. Stage two freezes the complete auxiliary graph
and trains the unchanged conditional denoiser with shifted q-sampling and noise
MSE. The original optimizer's otherwise unregistered ten-slot RAdam cache is
saved alongside both optimizers, schedulers, stage and RNG state.

The full denoiser keeps the original condition encoder and three 512D
timestep-conditioned layers. Its posterior runs all 500 linear-beta reverse
steps for all 100 paths per slide. Only independent paths are vectorized; the
condition is broadcast without dropping steps. Hash-seeded slide sampling
preserves global training RNG state. Raw 100-by-class signals and probabilities
are saved with input identities and hashes. Repeated locked evaluation rejects
disagreement with existing proof files.

The normalized output is the mean of the 100 softmax probabilities. The source
mean sigmoid-of-softmax scores are exported separately. The top-two paired
t-test at alpha .05 and PAvPU are slide-level quantities; they are not called
patient uncertainty after patient aggregation. Model selection uses the fixed
final epoch and never the test labels.

Four profiles are registered:

| Profile | Observed labels | Experts |
|---|---|---|
| `mexd_camelyon16_paper_reference` | two; class 0 explicitly background | 2 |
| `mexd_tcga_nsclc_paper_reference` | two positive subtypes | 3 |
| `mexd_bracs_paper_reference` | three positive subtypes | 4 |
| `mexd_class_port_reference` | arbitrary C >= 2 | C with observed class-0 background; otherwise C+1 |

`MExD` resolves to the general class/FM port. Every positive observed label has
one matching expert; an unobserved background expert is separate. If there is
an observed background label, declare `class_roles.background` equal to the
first item in `task.yaml`'s `classes`. Relabel the complete task consistently;
the adapter does not infer background from a label spelling. Different FM
dimensions retain the full native 512D architecture. All original instances
are loaded before learned routing; generic bag truncation is rejected.

Use a prepared patient-disjoint classification bundle with finite ordered
features (.npy, tensor .pt, or embedded HDF5 datasets). Patch IDs are retained
when provided; otherwise original row identities are content-bound explicitly.
Feature files, the actual float32 computation tensors, patients, labels, FM,
class roles and splits all enter the input identity. HDF5 external links,
virtual datasets and external backing storage are rejected because hashing a
container alone cannot lock those external bytes.

```bash
milplus fetch-source --source mexd --destination external/MExD
milplus run --data /path/to/prepared_task --method MExD --device cuda:0 \
  --output runs/mexd_full
milplus evaluate --run runs/mexd_full --device cuda:0
```

For an explicitly synthetic software check, keep the architecture and full
500-by-100 posterior but shorten both stages using a YAML config:

```yaml
threads: 2
model:
  moe_epochs: 2
  diffusion_epochs: 2
```

The total epoch count is derived from both budgets. A single-stage run is
rejected. A short software check is never reported as a native-budget pass.
There is no hidden dimension, timestep or posterior-sample fallback.

As of the initial 0.108 freeze, 54 related focused/interface checks passed
(20 MExD checks plus HDF5 and parent-interface regressions). An independent
CPU CLI run passed both stages, interruptions at epochs 1/2/3, all-state exact
recovery, four fresh selected-checkpoint reloads and exact posterior-file
replay. CUDA source/wheel, full 100+200 epoch budgets, whole-library and final
admission are separate gates. Those initial software checks did not constitute
a real WSI experiment or paper metric reproduction.

## Complete BRCA UNI reference run (2026-11-06)

The separately frozen production source finished the complete 100+200 epoch
reference on the actual BRCA IDC/ILC cohort at 2026-11-05T21:59Z. The existing
worker completed locked TEST evaluation at 22:06Z. Both subtypes are positive
observed classes; IDC is not treated as benign/background. This is the explicit
1024D UNI reference condition, not the paper encoder or private initialization.

An independent read-only audit checked the selected final epoch299 checkpoint,
all300 history entries,74200 MoE updates,148400 diffusion updates and all222600
original full-bag TRAIN visits. No heldout patient appeared in those visits.
All45 TEST posterior caches contain100 finite float32 paths per slide; their
hashes, softmax probabilities, mean/source scores and paired-t uncertainty were
independently recomputed. The500 reverse-step declaration is bound to the frozen
source/protocol, not claimed as an independently instrumented historical trace.
No training, inference, checkpoint selection or TEST threshold was rerun/tuned.

The native posterior evidence is preserved under the shared BRCA study's
`results/mexd/native_posterior_evidence_r1/`, including `audit_receipt.json`,
protocol, input/environment identity and history. The independent completed
prediction audit passed all58 completed methods with zero errors at22:08Z.
MExD TEST AUROC is0.8900862, accuracy41/45, balanced accuracy0.8890086 and
macroF1 0.9 on this single locked45-patient split. These are neither paper-score
reproduction nor evidence of statistical superiority. This production run also
does not certify the later current-source wheel's complete native-budget gate.
