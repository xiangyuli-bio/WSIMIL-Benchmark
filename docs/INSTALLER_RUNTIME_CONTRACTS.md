# Reviewed runtime requirements and installer metadata

The optional method extras describe separate Python environments. Do not install
all extras together: the original CONCH-dependent methods and pending DyKo and
FEATHER candidates intentionally require different timm/Transformers stacks.
The shared task/CLI interface does not imply one dependency environment for all
164 admission records.

Three installer declarations are corrected in this candidate:

| Extra | Declaration | Existing runtime reason |
|---|---|---|
| `protomil` | `setuptools==80.9.0` | Original wandb 0.16.1 imports `pkg_resources`; the tested newer setuptools 84 installation no longer provided it. |
| `dyko` | `torch==2.5.1`, `torchvision==0.20.1` | The explicit reviewed DyKo runtime rejects other Torch versions; its timm dependency must not resolve an unrelated newer torchvision/Torch pair. |
| `feather` | `torch==2.5.1` | The explicit reviewed FEATHER numerical port rejects other Torch versions. |

The previous DyKo/FEATHER wheel metadata permitted Torch 2.6 despite those
runtime checks. Actual rebuilt wheel metadata now rejects 2.6 and 2.4 while
allowing the reviewed 2.5.1 and its CPU local version. No runtime/model/loss code,
source pin, training budget, profile, or test assertion changes in this metadata
follow-up. Other optional dependencies are preserved.

ProtoMIL's actual declared extra has a 52-distribution dependency closure in the
reviewed CPU environment, including the library itself. All installed versions
in that closure satisfy their selected declarations. `timm`, `torchvision` and
Transformers are not requirements of that frozen-feature three-stage adapter;
their incidental presence in the broad regression environment is not a reason
to add them to the ProtoMIL extra.

The public `demo-protomil`, `run --method ProtoMIL` and `evaluate` commands also
completed in a new isolated environment exposing only those 52 distributions.
It reused exact existing dependency files through explicit read-only links;
it did not run a new pip dependency resolver. The environment probe confirmed
that timm, torchvision, Transformers and pytest were unavailable. The documented
synthetic fixture retained the full 512-to-2048 SAE and 2048-concept MIL with
its explicit one-SAE/two-MIL-epoch software budget. Two actual fresh evaluation
processes produced byte-identical prediction files. This extra installation
check is not added to the 4021-source/70-wheel regression counts and does not
replace the original method or full-budget checks.

The 42 original ProtoMIL source checks and five independent installed-wheel CLI
checks passed using the real setuptools 80.9.0 distribution. These checks and the
dependency-closure audit are distinct evidence. The closure audit alone does
not prove a fresh pip resolver installation, original full-budget execution,
authorized pretrained-weight loading, or real WSI performance.

Typical installation in a separate environment is:

```bash
python -m pip install '.[protomil]'
```

The documented source acquisition, native feature/probe inputs and method
preparation still apply. Installing an extra does not grant access to model
weights. DyKo and FEATHER remain `pending_adapter` with no executable public
profile until the original authorized TITAN/FEATHER24K assets and their full
validation evidence are available. DSPA-MIL/DSPA-U-MIL remains pending for its
original assets, alignment and missing complete specification.

This candidate has not replaced the default development or stable entry.
The full software suite, historical CPU numerical issue and remaining native
validation gates are not declared complete by these installer corrections.
