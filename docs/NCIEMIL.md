# NcIEMIL: source review and current component boundary

Install this method's optional dependencies from the checkout before using its
preparation/training entry: `python -m pip install -e '.[nciemil]'`.
The base-only installation does not supply timm/torchvision; their absence must
not be mistaken for a failed model mechanism check. Weights and actual inputs
remain separate prerequisites.

Official [repository](https://github.com/polyethylene16/NcIEMIL), pinned
`780d8b4c04ef6266e552bfd4c47b76470f7510fd`.
Primary [MIDL/PMLR paper](https://proceedings.mlr.press/v250/qiehe24a.html).
The pinned tree has no LICENSE and is loaded externally without redistributing
author code. The default branch's21-commit history has been inspected as well.

The complete paper method has three indispensable stages:

1. Task-specific Swin-Tiny extraction/filter pretraining. Paper Eq3 uses an
   explicit binary bag target and the highest positive-probability instance;
   the feature extractor and projection head both update. The paper states
   pretraining batch128, but no standalone filtering trainer, configuration,
   checkpoint or training budget was found in this pinned source/history.
2. Score every original patch using the trained filter; order by positive
   probability and keep highK followed by lowK. K512 is the CAMELYON16
   setting; K128 is BgIM. Selected features are produced by the task-trained
   extractor, not arbitrary substitute scores or an untrained encoder.
3. Full native aggregation:768to1024 projection, four-head instance attention,
   depthwise3x3/channel squeeze attention, cross-attention merge, classifier.
   Original main uses batch4, constant SGD2e-4/wd1e-5 and100epochs. CLI `--opt`
   defaults to Adam but is ignored by its literal optimizer construction.

Current implementation verifies the complete original aggregation graph at
native256/1024 instances, all parameter initialization, logits, every gradient
and RNG. The input width can use the original constructor's declared FM port;
the hidden width stays1024. It also defines the actual binary max-instance
objective and deterministic highK/lowK selection components. This does not yet
constitute a complete filtering/extraction training pipeline and therefore adds
no runnable method profile or completed-method count.

The released `_kv` layer outputs1024, then reshapes neighboring instance pairs
into keys/values with half the original token count. Paper Eq7 describes
independent K and V projections for all channel instances. Components explicitly
distinguish `released_interleaved_kv` from
`paper_independent_kv_repaired` (1024to2048); the latter is only the named KV
repair, not a claim that every source/paper difference is resolved. Source channel
attention also includes depthwise convolution and BatchNorm beyond the compact
SE expression in Eq5. Its grid is probability-rank order, not measured WSI
coordinates. Original `_init_weights` is defined but never applied; this is
preserved, including the original CLS randn then truncated-normal draws.

Remaining integration must retain actual raw-image/filter checkpoint provenance,
TRAIN-only pretraining and selection, complete patch ID/feature alignment,
explicit binary weak-target mapping for multiclass tasks, frozen selected-feature
cache, stage costs and full-state recovery. The full Swin stage cannot be replaced
by mean pooling or generic bag CE.

Further source runner issues are tracked for the future explicit recipe:
`utils.py` is absent; test split reading is commented out; SGD restoration requires
an `lr_scheduler` key that is never saved; selection uses a harmonic mean of
accuracy/macroF1/sourceAUC and accepts ties. Source multiclass AUC is micro rather
than macro. The reciprocal harmonic expression divides by zero when a component
is zero. Any finite-zero repair and a separate reporting metric must be explicit.
The channel squeeze BatchNorm fails for a singleton training batch; no silent
normalization replacement, duplicated instance or dropped patient is permitted.

Real WSI evaluation and paper metric reproduction remain unrun.

## Development0.89 complete reference pipeline

The new full filtering path uses the entire pretrained Torchvision0.20.1
Swin-Tiny (public ImageNet1K weights SHA256
`704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b`),
a learned binary768-to1 head, full RGB256 patches and explicit ImageNet
normalization. These are declared reference reconstruction choices: the
author's exact filtering weights, head, normalization, optimizer, epochs and
bag minibatch were not published in the reviewed material. The user must
specify pretraining epochs, AdamW LR/decay, bag minibatch and seed. There is
no invented paper-default training budget. Patch execution chunks retain128
instances, while the maximum is selected across every original patch.

To preserve stochastic DropPath exactly with a bounded graph, the method scans
every chunk in training mode, records its RNG, and replays the winning original
full chunk with the same RNG and batch shape. All logits must be bit-identical
before backward. The post-scan RNG is restored. This is verified against
independent retained-graph maximum losses, including257 patches across
128+128+1 chunks, and against the full pretrained Swin graph. No encoder or
projection is frozen. Resume preserves every AdamW moment, native model
parameter, bag order, selected winner and RNG state.

`milplus prepare-nciemil --data RAW_BUNDLE --weights SWIN_PTH
--pretrain-config SETTINGS_YAML --k 128 --output PREPARED --device cuda:0`
trains only TRAIN bags, then scores every original patch of all splits with
the frozen trained filter. The cache preserves all768D embeddings, positive
probabilities, original patch IDs and selected indices; it verifies highK then
lowK exactly and binds raw-image hashes, original patient splits and the
actual trained checkpoint. Bags with fewer than2K distinct patches fail.
Use the returned dataset with `milplus run --data PREPARED/dataset
--method NcIEMIL`. The explicit independent-KV condition is
`nciemil_full_reference_independent_kv_repaired`.

Aggregation uses literal author train/VAL code, native1024D/four heads,
batch4/four readers and SGD2e-4/wd1e-5 without momentum. It runs100epochs
without early stopping; validation selection accepts the latest harmonic
ACC/macroF1/source-AUC tie. Zero-valued components use an explicit continuous
zero extension, avoiding the author's division-by-zero crash. Multiclass
source AUC is micro, alongside the benchmark's separately reported macro
metrics. Singleton training tails fail explicitly; no patients are dropped
or duplicated. All source initializers, dropout and BatchNorm remain intact.

Full native-budget/public-CLI acceptance remains in progress. The reconstruction
is not a numerical reproduction of the authors' unavailable filtering stage.

Native source-checkout verification completed on H200: both K128/four-class
and K512/binary cohorts ran all three actual Swin reference pretraining epochs
(with interruption/resume), full original RGB256/2K+1 extraction, and both
released/repaired aggregation conditions for all100epochs. Actual updates were
200 for each K128 condition and100 for each K512 condition. Both selected
checkpoints were independently reloaded twice. These are synthetic native
software experiments with public pretrained weights; the3-epoch filtering
budget is explicitly a reference-validation choice, not a recovered author
budget. The failed initial fixture missing task_id is preserved separately.
See validation/nciemil_native_source_r2.json. Installed-wheel, complete-library
and final release acceptance remain separate and pending.

## Early native selection-capacity guard (2026-11-06)

Candidate preparation now checks the same native K128/K512 and2K-distinct-patch
requirements before creating a filter run or training Swin. Cache extraction
uses the same shared guard. The previous flow could complete the filtering
budget and only then discover that selected bags were impossible. Valid inputs
still use the identical complete filter, settings, checkpoint, highK/lowK
selection and aggregation; no K, patch, model or training-budget fallback exists.
The already verified input object is passed through both stages rather than
constructing it again. Standalone Swin filtering itself remains unchanged.

All39 new boundary/orchestration plus existing full native graph/source checks
passed on the actual220GiB host using an isolated CPU-visible candidate. Original
native test widths, inputs and assertions were not reduced. The development
environment's first attempt failed collection because timm was absent; that XML
is retained and is not called a passed run. These39 checks do not replace a fresh
full Swin/100-epoch/installed-wheel acceptance of this candidate.

Actual BRCA capacity was independently checked against all856 delivered raw RGB
H5 shapes and original coordinate SHA values:8 bags have fewer than256 patches,
146 fewer than1024, and the smallest has128. Patch IDs were additionally checked
for distinctness in every bag below1024. Thus the full current cohort cannot
satisfy even native K128 without genuine new patch extraction. This diagnostic
is explicitly not another complete RGB pixel/archive SHA audit and is not a fit.
No patient, patch, image or native K was changed to make the cohort pass.

The live GPU source/runtime was not edited. Isolated source is
`code_nciemil_capacity_20261106_r1`; BRCA shared evidence is
`nciemil_capacity_candidate_r1/source_receipt.json`,
`nciemil_capacity_and_native_source_actual_r1.xml` and
`brca_continual_scope_and_nciemil_real_capacity_r1.json`.

### Independent installed-wheel check for this guard

An isolated snapshot of the current0.131.0 checkout built successfully without
changing the original worktree's build artifacts. The wheel contains all164
registered method records and415 profiles; all512 package files match the source
snapshot byte-for-byte, including the new capacity module. These are packaging
facts, not full-method acceptance or clinical completion counts.

The wheel was installed offline with pip/Python3.10 into a new shared target,
without installing/upgrading dependencies in any active environment. On the
existing H200 host with its verified dependencies, a CPU-only child passed
`python -m milplus --help` and all39 unchanged related cases. Every actually
imported `milplus` module was checked to come from that installation target,
not the working source/candidate directory. CUDA visibility was changed only
for the verifier, never for a running training process. This is not a fresh
dependency-resolution test, console-script shebang test, full164/415 execution
gate or new native Swin/100-epoch budget run.

Scoped evidence lives under the BRCA shared directory's
`nciemil_capacity_wheel_r1/`: `wheel_integrity.json`,
`installed_wheel_actual_receipt.json`, complete build log and the candidate wheel.
Wheel SHA256 is `0d1d2f97eb5a184e502b8aef2f61cf0b70f6e9a9d5e8e463d31e4d191cbe827f`;
test report is `nciemil_capacity_and_native_source_installed_wheel_r1.xml`.
The same version number does not identify the bytes: use this exact hash and
scope, and do not call this candidate a new public release.
