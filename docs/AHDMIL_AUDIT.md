# AHDMIL source audit

Fixed source: [JiuyangDong/AHDMIL](https://github.com/JiuyangDong/AHDMIL),
`2f38354386165167815eb2998424ad63792a65bf`. Read `Joint.py`, both native model
files, `baseline.py` and README. DMIN preparation uses the fixed preceding
HDMIL source `ef3d4681054970208e94f8205bd96b3180c00926`.

| Source observation | Adapter treatment |
|---|---|
| README requires a prior trained DMIN | Execute the complete original DMIN stage internally and bind its strict VAL-selected weights |
| Two native MobileNetV4 experts | Both complete models, their parameters and all BatchNorm buffers retained |
| Hard/attention branch chosen by NumPy | Preserve probability, conditional RNG draw and raw-attention targets |
| Frozen-in-loss DMIN still in train mode | Preserve stochastic dropout while computing LIPN targets |
| Every bag updates LIPN then DMIN | Two joint Adams in exactly that order; pretraining Adam is separately retained |
| Second expert inference is no-grad, but train-mode | Preserve its additional BatchNorm updates |
| Cross-EMA before first epoch and after each epoch | Copy all expert parameters first, then update simultaneously; never average buffers |
| Epoch-end cross-EMA precedes validation | Preserve exact ordering and include it in checkpoint counters |
| Inference averages raw expert logits before thresholding | Retained; not majority-vote hard masks |
| RCC retention includes all three classes | Retained; differs from the preceding HDMIL source's two-class union |
| CE improvement also requires retained fraction in[.3,1] | Preserve eligibility, strict ties and patience5 |
| No eligible best triggers network and optimizer reset | Restore the same initial selected DMIN; freshly initialize experts using current RNG, no extra EMA |
| `baseline.py:8` imports a nonexistent class name | Explicit AHDMIL/HADMIL entry compatibility repair |
| `Joint.py:292` accesses uninitialized `args.lower_ratio` | Explicit `self.lower_ratio` correction; retain original decrement and epoch condition |
| Source skips NaN LIPN losses | Fail explicitly rather than counting skipped updates as a complete recipe |
| Public source uses cached high features and low RGB | No claim that this port measures online high-resolution feature acquisition cost |

Validation includes original complete graph/gradient and BatchNorm/RNG
comparisons, independent original `HADMIL.train` execution, exact pretraining
identity, all three Adams and continuation at stage boundaries, eligibility
and reset behavior, original physical-input contract, and public source setup.
The fixed-source classes remain private local imports, with source licensing
and README academic-use terms recorded in `sources.lock.json`.
