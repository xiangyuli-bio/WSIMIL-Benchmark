# HDMIL source audit

Fixed source: [JiuyangDong/HDMIL](https://github.com/JiuyangDong/HDMIL),
`ef3d4681054970208e94f8205bd96b3180c00926`.
Read `models/hdmil.py`, `models/mobilenetv4.py`, `DMIN.py`, `LIPN.py`,
`baseline.py`, `run.py`, `run_scripts.sh` and README. The source definitions are
loaded from verified Git blobs in a private namespace; unrelated imports and
hardcoded personal data paths are not executed.

| Observed source behavior | Adapter handling |
|---|---|
| DMIN v4 and LIPN v5 share compatible MIL tensors | Two explicit stages, validated DMIN checkpoint transferred strictly into v5 |
| Full512/256 gated attention with per-class KAN bag and instance heads | Preserved; degree12/16 follows each task README |
| DMIN loss: .7 bag CE + .3 instance SmoothTop1 + .5 KL + .5 representation MSE + 2 retention MSE | Every term active, including gradients through both KL branches |
| Original training selected attention divides by one global sum over classes/instances | Preserved, not silently replaced with per-class normalization |
| Source inference instead normalizes selected instances per class | Preserved and compared to independent original definitions |
| Retention union uses channels0 and1, also in the published RCC route | Explicitly retained source behavior; no claim of an all-class union repair |
| LIPN uses hard straight-through image masks with L1 + union-rate MSE | Original MobileNetV4 tiny and full objective preserved |
| Frozen teacher is `train()` during LIPN, therefore dropout remains active | Preserved; `no_grad` does not turn it into an eval teacher |
| Separate scripts reset the same seed and construct v5 then RGB processor | Explicit phase transition preserves this initialization/RNG order |
| Stage1 minimum is only tracked from one-based epoch50, ties count against patience20 | Preserved; stage2 uses strict mean-CE minimum and patience5 |
| README retention ratios.6/.7 conflict with old run_scripts.sh .1 | Named profiles use README, conflict recorded |
| Source skips NaN batches | Adapter fails and records the failing run; it never counts omitted updates as a complete budget |
| Source has hardcoded `.cuda()` loss construction | Device-neutral equivalent binary SmoothTop1, independently checked in both smooth and hard regimes |
| No native patch/FM cropping in released execution path | Matched low-RGB and cached-feature route; no end-to-end inference speedup claim |

The README declares GPLv3 and non-commercial academic intent. Third-party source
is not vendored. Source acquisition and local review remain explicit.

Validation is recorded separately: native-width author graph/loss/gradient
comparisons; direct execution of the author's two training loops; exact CPU
interruption/resume across both phases; actual measured input failure cases;
CLI routing; installed-wheel complete200+200-max budget cases; and the full
library regression. A shorter software run is not a native budget receipt.
No historical paper scores or natural WSI validation have been produced.
