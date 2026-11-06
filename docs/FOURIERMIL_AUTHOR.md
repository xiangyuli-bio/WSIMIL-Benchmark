# Original Fourier-MIL training profiles

The existing `fouriermil` profile remains the packaged MIL_BASELINE adaptation.
Its model definition, initialization and generic training recipe remain
unchanged. Two additional profiles privately load the original author's fixed
Git objects at commit `0192fe7f7a4fdbb4161148d0b88b0d67dbaa14f5`:

* `fouriermil_author_200_fp16_val_loss_locked_test`: single visible CUDA GPU,
  native FP16 autocast with GradScaler.
* `fouriermil_author_200_cpu32_reference_val_loss_locked_test`: explicit CPU
  float32 numerical reference. This does not validate native CUDA FP16.

Both preserve the original 1024-dimensional patch input, 512-dimensional
hidden layers, two Fourier blocks, native eight-block APFF, original CLS-token
initialization, zero dropout, original weight-decay parameter groups and
unchanged Lookahead/RAdam update definitions. Input width is not silently
changed for a different foundation model.

The fixed Camelyon configuration specifies 200 maximum epochs, batch size one,
eight workers for both TRAIN and VAL, shuffled TRAIN bags, unchanged patch
order, gradient accumulation two, learning rate 0.0002, weight decay 0.00001,
and early stopping after ten validations without a strictly smaller loss.
There is no gradient clipping or scheduler. Every loss is divided by two,
including a final singleton accumulation group, following Lightning 1.2.3.
The original detached CPU `log(softmax)` validation function is retained,
including its FP16 quantization. Stable cross entropy is not substituted for
checkpoint selection.

The author's YAML seed is 2021. The unified CLI default remains 2026; supply
`--seed 2021` to request the author configuration's seed. Every seed is recorded.
The patient's identity and fixed split come from a validated Bundle. No TEST
prediction participates in fitting or checkpoint selection.

## Compatibility and checkpoint repairs

The original environment was PyTorch 1.7.1 with CUDA 11.0 and Lightning 1.2.3.
These profiles run under the declared current runtime and do not claim that
the old binary environment has been reproduced.

The only model rewrite replaces an unconditional CLS-token `.cuda()` with the
input's device. Author training and validation step definitions are loaded
unchanged; the adapter replaces the obsolete framework/logging wrapper.

Original Lookahead does not initialize modern Optimizer hook registries.
Calling current `zero_grad()` lazily wraps `step()` and then fails on a missing
hook attribute. The compatibility layer supplies empty hook registries while
retaining the original step body. `zero_grad(set_to_none=False)` preserves the
PyTorch 1.7 default.

Original Lookahead's loader also fails on current optimizer hooks and stores
slow buffers using process-specific parameter IDs. Portable checkpoints bind
the original fast RAdam state, ten-slot cache and slow buffers to stable
parameter names, shapes, dtypes and optimizer-group order. They additionally
preserve all gradients, GradScaler state, sampler order and cursor, class
statistics, model mode and all RNG states. A recovery in the middle of
accumulation retains scaled gradients. Legitimate AMP overflow is handled by
GradScaler and recorded as a skipped optimizer step.

Requests bind source and software identities, numerical policy, data and
feature inventories. Changing these requires a separate run. Selected TEST
evaluation is an explicit command after fitting has completed.

## Commands

Prepare the original source on a networked host:

```bash
milplus setup --method fouriermil_author_200_fp16_val_loss_locked_test
```

Run on a dataset Bundle containing the required 1024-dimensional features:

```bash
milplus run --data /path/to/task \
  --method fouriermil_author_200_fp16_val_loss_locked_test \
  --device cuda --seed 2021 --output /path/to/new_run
milplus evaluate --run /path/to/new_run --subset test --device cuda
```

The original source has a fixed-tree MIT license and is privately loaded.
The adapter, software checks, complete-budget reference runs, source/wheel
validation, real WSI experiments and paper-score reproduction are distinct
evidence stages. Consult the current review receipt for the achieved stages.
Synthetic fixtures exercise the complete model and training control; they do
not establish clinical performance or reproduce the author's paper scores.
