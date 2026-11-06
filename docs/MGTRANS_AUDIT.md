# MG-Trans fixed-source audit

Source: [Jiangbo-Shi/MG-Trans](https://github.com/Jiangbo-Shi/MG-Trans), commit
`a18d886b89ff0eddf5ae0b4ae16058623fe34dbc`.

Reviewed files include `models/model_MG_Trans.py`, `models/model_utils.py`,
`models/configs.py`, `utils/core_utils.py`, `utils/utils.py`,
`datasets/dataset_generic.py`, `patch_alignment.py`, `main.py`, and the BRIGHT
training script. Author definitions are loaded from verified Git objects in a
private namespace. The original checkout is not vendored into this package. No
license declaration was found in the fixed repository; no redistribution
permission is inferred from public visibility.

The complete released learning path is retained. The model uses shared
two-scale Transformer parameters and the actual three-layer GENConv component.
The source configuration's unused `num_layers=12` does not create twelve encoder
blocks. `pt2graph` detaches projected features before constructing its PyG graph;
that behavior is preserved and must not be described as graph-input gradient
propagation. The learned graph parameters still receive gradients.

The semantic term is the source's negative cross-entropy on the classwise L1
distance of the high/low logits, with weight `1e-4`. It is not replaced with MSE.
The trainer's two Renyi MI terms also retain their separate `1e-4`
multiplications, original 32-slide window, dummy zero initialization, detached
previous examples, and first-ten-distance-columns bandwidth heuristic. Every
epoch resets that history. Data with fewer than 32 training slides cannot prove
that the MI stage actually executed.

Compatibility repairs are explicit:

- Place graph tensors on the input device instead of hardcoded `cuda:0`.
- Map the removed `_LinearWithBias` name to the equivalent current linear class.
- Replace removed `torch.symeig(...)[0]` with upper-triangle
  `torch.linalg.eigh(..., UPLO='U')[0]`.
- Query the latent HNSW index with its hidden feature vectors. The released code
  builds that index in hidden space but queries it with two-dimensional
  coordinates. The current model consumes the spatial edge set; the latent index
  construction is nevertheless retained and dimensionally repaired.

HNSW reproducibility needs more than restoring PyTorch's RNG. The native library
uses a [thread-local random generator](https://github.com/nmslib/nmslib/blob/master/similarity_search/include/utils.h),
and the source's duplicate-coordinate padding creates equal-distance neighbor
ambiguity. R1 measured different logits for identical inputs and restored Torch
state. R2's fresh native thread fixed one controlled probe but did not remove
equal-distance membership/order ambiguity in the broader regression test. Both
failed source revisions, logs and training inputs are retained outside this tree.

R3 retains actual NMSLIB HNSW, `M=16`, construction ef 400, query ef 90, radius
nine, and the original spatial distance cutoff. Each build uses one index thread
and a fresh native caller thread. Spatial queries resolve exact geometric
distance ties by node ID and enforce the source's assumed self-first result
before removing self. This does not replace HNSW with another neighbor algorithm,
reduce search effort, reduce the network, or omit losses. It is an explicit graph
realization correction, not evidence that historical uncontrolled graph draws or
paper metrics have been reproduced.

Verification has separate scopes: measured-input corruption checks; full-width
forward/backward and repeated-coordinate graph checks; direct execution of the
author's training loop including the MI stage; exact model/Adam/RNG resume;
native installed-wheel complete training budgets; and the fresh full repository
suite/profile gate. A forward pass or a successful short fit cannot stand in for
all of these. Development results remain unadmitted until their required gates
are complete.
