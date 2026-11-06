# QPMIL-VL implementation work in progress

Original: can-can-ya/QPMIL-VL, commit 3a7a7698582dec866d43eb748f8c3599f7be4391.
This is a component integration branch, not a runnable/admitted method yet.

The original frozen CONCH text tower is twelve layers, 768 hidden, twelve heads,
128 positions and 512 output dimensions. Training propagates through all frozen
text layers into the selected 24-token prompts. Twenty key/prompt entries, five
matched entries, majority matching across the minibatch and task-frequency
penalties must remain. Class-ensemble text is generated using the actual local
CONCH tokenizer and real pretrained text weights. Cached class features alone
cannot replace differentiable prototype prompt encoding.

The original update uses CE, 0.5 matching loss, 0.5 class-similarity loss,
gradient norm clipping at 1, then clears exactly zero parameter gradients before
Adam, including weight decay. Earlier task tunable vectors are frozen. Task
frequency tables must retain the last actual epoch's counts; evaluation matches
without the training penalty. The default is the final task epoch, not the best
validation checkpoint (`load_best_ckpt: false`). The combined validation monitor
is (loss + 1 - accuracy)/2, with plateau LR and early stopping.

A source behavior needing explicit preservation: `_train_each_epoch` collects
16 slide bags and drops the remaining incomplete group. Fewer than 16 TRAIN
slides causes the original collector to fail. An adapter must either explicitly
reject that original-mode input or name a separate tail-flush repair. The
reference tests/native budget must contain actual complete minibatches, not
silently change the 16-bag recipe. No future/TEST feature decode may influence
training or validation selection.

The source R2 full four-task trainer and three exact software interruption points
have passed with real pretrained weights and complete original CoCa construction.
The current integration successor adds both original query variants and the
unified registry/data/CLI interface. Focused, independent-wheel, complete native
and whole-library gates are still being verified; the method is not counted
from this audit or component-only evidence.

Author code and weights stay private external inputs; no author ND source or
CONCH weights are redistributed in the package.

The full source CoCa constructor also matters for random initialization order:
its visual tower and unused caption decoder consume RNG before key/prompt
initialization. The next source revision preserves that complete constructor,
uses the explicitly declared timm 1.0.11 version and the source nn.LayerNorm,
and strictly checks every used text/visual checkpoint key. R1's component-only
text-tower probe stays preserved as preliminary evidence; it is not a complete
original-constructor oracle.
