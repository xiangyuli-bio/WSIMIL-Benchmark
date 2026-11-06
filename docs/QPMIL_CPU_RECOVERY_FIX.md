# QPMIL-VL deterministic execution candidate

Candidate0.118 retains all164 registered entries,161 written entries and352
profiles. No new method has passed admission merely because this fix exists.

The original0.117 full CPU recovery test failed at strict tensor equality.
Its complete failed directory and reports are retained. The first validation
loss already differed before the first interruption; final sample generators
and RNG states were equal. The failure is not explained by missing RNG restore.

The original full CoCa/16bag computation was instrumented once. Replaying its
actual prompt/key index-backward input tensors20times on Torch2.5.1+cpu with
two threads yielded20/7 distinct outputs under the default policy. Strict
deterministic Torch execution yielded one output in each case. This identifies
a nondeterministic operator used by the actual original backward graph.

The candidate explicitly selects strict deterministic original Torch operators
for QPMIL train/evaluate. It records the effective policy in runtime provenance,
keeps float32 and the requested threads, and restores the caller's numerical
flags after the public entry returns or raises. Direct low-level pretrained
component loading configures the same process policy for native component
oracles. No model, loss, batch, task, budget or equality assertion is reduced.

Four operator/control regression cases were added. All prior4012 test bodies
are retained unchanged. Full four-task recovery, both query profiles, literal
author update/gradient/Adam comparisons, independent-wheel execution and the
complete release suite remain required. At creation these checks are pending;
the operator diagnosis alone is not a complete-method acceptance.

The default development/stable entries and the running original RMDL native
jobs are unchanged. DyKo, DSPA-MIL/DSPA-U-MIL and FEATHER still require their
documented original assets/specification gates.
