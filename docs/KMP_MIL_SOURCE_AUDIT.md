# KMP-MIL full-method integration work

Pinned source: Lingling-Yuan/KMP-MIL,
77137ac84a9a27b16a2c07aca414c534f7cf5226. This branch does not increase the
runnable/admitted method count from a component probe.

The source declares timm1.0.20 and requires an omitted top-level official CONCH
package. The dependency is therefore explicitly reviewed and pinned to official
CONCH141cc09c7d4ff33d8eda562bd75169b457f71a62, not presented as the authors'
historical dependency pin. Use the actual authorized checkpoint and full source
CoCa constructor. Learned prototypes must differentiate through all frozen
12-layer/768-hidden/12-head/110M-parameter text layers. No cached/random text
substitute or QPMIL dependency-version substitution is used.

All four source tasks have ten actual semantic descriptors. Descriptor text
initializes current keys through actual CONCH encode_text; the key, 24-token
prompt, PFC gamma-delta and beta memories grow after each task. TC residuals are
current-task-only. The source has distinct universal-bank class-CL and
single-task-bank task-CL inference. The default reloads the best VAL epoch before
appending the actual model key/prompt/PFC tensors to frozen banks.

The source epoch FLUSHES incomplete16bag tails, unlike QPMIL. Current-task masked
VAL uses the literal evaluator's contiguous-global-to-local label mapping and
binary >=0.5 tie rule. Preserve all CE, matching, routing, class-similarity and
PFC regularization terms, clipping, zero-gradient parameter exclusion, Adam,
plateau and early stopping. Four native maximum budgets are20epochs; initial
class counts are3/2/2/2. No old WSI replay occurs.

New task pools are randomly initialized during each transition. Boundary resume
must restore saved RNG BEFORE constructing those new pools. Mid-task resume
instead restores parameter/optimizer state and then saved RNG. Do not apply the
QPMIL all-pools-at-start RNG schedule here.

BOM-bearing UTF8 source/text files are normalized only while reading pinned Git
objects. Tracked author files stay untouched. Full pipeline, true interruptions,
private-source reference oracles, publicCLI, independent wheel/native/global
checks are separate gates. TEST is reserved for explicit evaluation in the
patient-safe port.
