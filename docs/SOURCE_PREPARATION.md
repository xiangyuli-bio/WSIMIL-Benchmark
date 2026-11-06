# Prepare the complete pinned code for a method

Prepare each method once on a networked CPU host, then reuse the source cache
from its reviewed training environment. Training still uses the usual prepared
task path and method name.

First inspect the actual source URLs, commits, destinations and dependency list:

```bash
export MILPLUS_SOURCE_ROOT=/your/owned/milplus-sources
milplus setup --method ProtoMIL --plan
```

The plan does not create directories, clone repositories, install dependencies,
load a model or download weights. It includes any public initialization download
that the existing setup command would perform. Then prepare the code:

```bash
milplus setup --method ProtoMIL
```

The command obtains both `protomil` and the reviewed original
`protomil_dtn` sparse-autoencoder dependency, at their existing fixed commits.
For KMP-MIL it obtains both `kmpmil` and the reviewed official `conch_kmp`
source. It retains existing dependencies such as RMDL's Keras-contrib,
SparseConvMIL's SparseConvNet, AHDMIL's HDMIL, and source-specific metric,
Mamba, text and timm references. Shared transitive dependencies are obtained once.

The full dependency plan is validated before the first repository download.
The existing fetcher checks the actual Git origin, commit and local changes
before reusing a destination. It does not overwrite a different or edited tree.
The existing full commit locks and sparse checkout settings remain in force.

MeanMIL and MaxMIL use packaged implementations and have no external source
checkout. Their setup command completes without attempting to clone a
nonexistent repository. Other bundled adaptations retain their existing optional
reference-source preparation where the catalog has a source lock.

`setup` prepares code. It does not install incompatible method extras together,
create patient labels/splits, generate real annotations, grant gated model access,
or verify a method's original assets. DyKo, FEATHER and DSPA-MIL2025 now have
public code profiles with original asset and validation prerequisites visible
in the plan. DSPA-U remains unimplemented; see [usage](LAST_THREE_METHODS.md).
Use the method documentation to prepare its real inputs and separate runtime.

This follow-up changes the setup command only. Training, losses, sampling,
recovery and evaluation implementations remain byte-identical to the parent
candidate. Its software preparation checks do not replace original training or
native-budget validation.
