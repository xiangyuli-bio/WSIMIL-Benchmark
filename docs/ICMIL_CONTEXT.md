# In-context ICMIL

This is `injurise/ICMIL` at commit `23bb7203777891987ec3c18a1233ffe238b032b3`,
not the iteratively coupled method. Three separately hashed original checkpoints
are retained as execution profiles. Each is the complete 8,257,552-parameter,
256-embedding, 1054-MLP, six-iteration pretrained model, saved at epoch 8000.

The original downstream algorithm does not train an optimizer. Fitting binds all
labelled TRAIN bags as context. Evaluation supplies unlabelled query bags to the
actual frozen model, with original TRAIN-only feature normalization and original
softmax. VAL is reported without checkpoint or hyperparameter selection. TEST
is only evaluated by an explicit evaluate call. Query labels stay outside the
native inference process. A saved context binding can be interrupted and resumed.

The source silently drops channels beyond 25; this adapter rejects that case.
For an FM with more than 25 features, prepare an explicitly named reference:

```bash
milplus setup --method icmil_context
milplus prepare-icmil-context --data /path/to/task_bundle --output /path/to/context_bundle \
  --bag-size 10 --projection train_pca25_reference
MILPLUS_ICMIL_PYTHON=/path/to/native/python3.11 milplus run \
  --data /path/to/context_bundle --method icmil_context --output /path/to/run
milplus evaluate --run /path/to/run
```

Use `--projection native` for an already prepared <=25D representation. The
published TCGA reference uses bags of 10 instances and 25D precomputed UNI2
features; its historical PCA fit is not supplied. Our reference PCA uses every
TRAIN patch and all source FM channels, with frozen serialized mean/components;
no held-out features or labels are fitted. Uniform bag selection is explicit,
without replacement, with sorted original row indices and corresponding physical
coordinates preserved. Bags smaller than the requested budget fail rather than
receiving synthetic instances. This preparation is not claimed to reconstruct
the authors' historical PCA or to process new raw WSI with TRIDENT.

The native environment requires Python 3.11 and actual original dependencies.
The host launcher can remain in the common Python3.10 environment. Set
`MILPLUS_ICMIL_PYTHON` to the native interpreter; `--device cuda` requires that
interpreter to have working CUDA Torch and never falls back to CPU. The CPU
reference uses Torch2.5.1+cpu/NumPy2.2.6/h5py3.16.0/scikit-learn1.7.2,
SciPy1.15.3/schedulefree1.4.1. Install schedulefree because the original
architecture imports its training helpers. No TabPFN baseline is required for
the released ICMIL model itself.

The original author reference on all three released checkpoints and all fifty
provided TCGA splits was executed separately. It concerns author-provided
precomputed features, not new WSI cohorts or a new BRCA/COAD experiment.
Downstream adapter equivalence, independent installed-wheel checks and the
complete library regression remain separate acceptance evidence. Do not call
50 overlapping split draws 50 independent cohorts, or count three pretrained
seeds as three independent patient populations.
