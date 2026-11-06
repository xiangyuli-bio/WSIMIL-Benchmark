# KMP-MIL original complete continual protocol

Pinned author: `Lingling-Yuan/KMP-MIL`, `77137ac84a9a27b16a2c07aca414c534f7cf5226`.
The author repository has no LICENSE at that revision; source is privately fetched
and verified rather than redistributed. Its omitted CONCH dependency is the
explicitly reviewed official revision `141cc09c7d4ff33d8eda562bd75169b457f71a62`,
not an asserted historical author dependency pin. Authorized real weights are
required separately. The isolated runtime uses author-declared timm1.0.20 and
huggingface-hub0.36.2, independent of QPMIL's timm1.0.11 environment.

The native method retains the full original110M parameter pretrained CONCH text
tower, actual BPE and class ensembles, semantic descriptor initialization,24token
learned prompts, five per-bag matches, PFC affine visual calibration, current-task
TC residuals, past-key routing, all five source loss terms, clipping and exact-zero
gradient removal, Adam0.005, plateau and early stopping. It runs the literal author
epoch including the incomplete16-bag tail. Each task reloads its best combined
VAL loss/accuracy checkpoint before caching the learned key/prompt/PFC bank. Old
WSI bags are not replayed. Max and mean query variants are named separately.

Prepare an ordinary512D projected CONCH feature dataset with true patient IDs,
patient-disjoint TRAIN/VAL/TEST and ordered original patch identities. A version1
preparation schema includes `tasks` (ordered IDs/classes), `slide_tasks`,
`provenance`, `conch` (authorized checkpoint path/SHA256 and actual feature encoding),
`class_descriptions`, `templates` and `semantic_descriptors` (at least five actual
texts per task). No invented random key fallback is substituted. Run
`milplus prepare-kmpmil --data DATA --schema schema.yaml --output manifest.json`
and bind `encoders.FM.kmpmil.manifest` to that manifest in task.yaml. The first task
must have at least two classes; every task class needs independent TRAIN/VAL/TEST.

Then use `milplus run --data DATA --method KMP-MIL --output RUN --device cuda:0`.
The optional epoch override is for explicit experiment budgets; the reference
budget remains20 epochs per task with patience10. Fresh-process `milplus evaluate
--run RUN --device cuda:0` evaluates selected checkpoints. Global class-CL, universal
bank task-masked diagnostics, and final per-task memory-bank task-CL predictions
are saved separately. TEST inference is deferred from original post-task reporting
to this explicit command; it never controls updates or checkpoint selection.

Epoch-boundary recovery retains learned tensors and aliases, all gradients,
current TC state, immutable selected banks, optimizer/scheduler/early-stop state,
all loader generators, actual patch ordering and RNG. At a task boundary RNG is
restored before the next task's random prompt construction; midtask recovery
restores it after reconstructing and loading the current model.

The explicitly synthetic demo preserves original four tasks with3/2/2/2 classes,
ten semantic descriptors per task and1,144 clinical class descriptions. It is not
TCGA data, raw WSI extraction, clinical validation or paper-score reproduction.
All software, installed-wheel, full native and final library gates are separate.
