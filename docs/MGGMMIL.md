# MGGM-MIL candidate implementation

This is the original JiuyangDong/MGGMMIL family, not GMMamba.
Source commit: `dfbe91120e0c7c822abdbbba758c307ddabd2826`.
Original source is loaded privately from its pinned Git objects; no LICENSE was
found in the pinned source tree, and third-party source is not redistributed.

The complete candidate implementation retains four384D blocks, three original
GENConv graph scales, full SRMamba16state/4conv/expand1, sequence-reordering rate10,
GABMIL256, binary CE, original constant Adam3e-5/wd1e-5, one complete slide per
update, first strict minimum validation-CE selection at1-basedepoch50,patience20,
and the original200epoch maximum. Complete model, all Adam moments, random
states, selection and exact per-slide update history are checkpointed.

## Inputs

An ordinary binary patient-disjoint task bundle contains actual512DResNet18
ImageNet orPLIP feature bags and their true coordinates. Declare
`encoder.mggmmil.manifest: graphs.json`. That manifest binds the actual K4/8/12
edge files to feature and coordinate SHA256s, row order, imaging modality(UAF or
H&E), encoder identity and graph provenance. See `mggmmil_demo.py` for the exact
schema. Both original int64 `.pt` tensors and int64 `[edges,2]` `.npy` files are
accepted. The source converts its stored `[edges,2]` edge lists to GENConv's
`[2,edges]` inside every forward; input layout is never silently guessed.

The public repository reads precomputed graph files but does not release their
construction. We preserve supplied original edge lists, directedness, self edges
and order. We do not invent graph preprocessing or claim that a different KNN
builder reproduces the authors' graph construction. Graph/coordinate/feature
bytes and patient partitions are checked before training and on reload.

## Invocation

After obtaining the separately pinned source and preparing the data bundle:

```bash
milplus run --data /path/to/task --method MGGMMIL --output /path/to/new/run
milplus evaluate --run /path/to/new/run
```

The default exact PyTorch selective recurrence is explicit and works on CPU or
CUDA. It retains both original SRMamba paths, including padded transposition and
padding removal; it is not ordinary Mamba. `model.backend: cuda` instead requests
the pinned original fused1.1.2/1.1.1 kernels and fails if unavailable. There is no
automatic backend fallback. CUDA native execution and software numerical
validation have separate evidence.

## Current acceptance

The first eight component checks passed: full native model/every gradient/Adam,
literal author training loop and no-grad validation, rate10padding/reordering,
source selection boundary, graph mutation and exact state checks. The initial
component-stage receipt predates runnable registration. Public CLI/installed
wheel/full200epoch native/global acceptance are still required. This candidate
is not included in the published150-method development count or stable117.
Synthetic fixtures are not real WSI results or paper-score reproductions.
