# ConVL: complete development integration; acceptance pending

Source: junjianli106/ConVL at `f1017a7ea0f51ff8d436b1334f937b85aa86c46b`. No repository-wide license was found at the fixed revision. Source is fetched privately, not redistributed. This is one of137 implemented development families, with two explicit execution profiles among288 configurations. The accepted release remains0.62/117families; native CUDA, whole-library and final-wheel gates remain pending.

## Complete source network

The actual full native CONCH text tower uses12layers,768hidden,12heads and512D projection with the exact original pretrained checkpoint. The original source fork returns contextual token embeddings as well as normalized pooled text. Its dynamic16token prompt learner preserves those initial contextual prefix/suffix buffers, rather than substituting raw token embeddings or precomputed final concept vectors. All17 original concept dictionaries retain their complete ordered wording.

The trained text path deliberately follows the public ConVL TextEncoder: unmasked transformer, source EOT-position pooling and projection. This is distinct from the initialization path using the original causal/padding/CLS text tower. Every released concept was checked for safe original128token/EOT indices. Only unused vision and multimodal-decoder allocations are omitted; this changes unused factory initialization RNG draws, so no cold-start seedwise equality to the full upstream factory is claimed. The original unused768D fc_patch allocation is preserved; actual forward projects native512D visual and text features through the shared512to256 fc_concept.

The real original four-head concept-guided attention, per-class instance pooling, visual/concept heads, all aggregation modes and concept attention outputs remain present. Trainable prompts, image-conditioned MetaNet, original12layer text blocks, shared projection, attention and both output heads receive actual finite gradients. Original unused token embedding/CLS and fc_patch are not falsely reported as trained.

The source distribution alignment is one-way KL between sigmoid values, with batchmean reduction, and gradients through both arguments. It is not symmetric normalized-distribution KL. The optional source SinkhornLoss is a softmax Euclidean distance. The source clustering terms maximize each patch across concepts and each concept across patches, using negative configured coefficients. The released survival trainer adds these to NLL on blended hazards, while validation monitors NLL alone. Paper-vs-source objective differences must remain explicit.

## Optimizer and data

Original Lookahead/RAdam update definitions and original bias/normalization decay partition are imported privately. Modern Optimizer hook initialization, reconnecting wrapper/base groups after restore, and retaining the ten-entry RAdam rectification buffer are explicit serialization ports. All37 original updates match every fast weight, slow weight, moment and counter; four recovery cuts cover first-step and interpolation/rectification boundaries.

Immutable patient inputs bind every complete slide bag in stable registered order. Actual source ProgData indexes slide-level case_id while reporting unique-patient length; the unified interface must use each true patient once and concatenate all slides. Time/event labels and supplied original four-bin labels remain bound to each true patient. Source event1 is converted to censorship0 only at the original NLL boundary.

The input layer supports the authors supplied four-bin label table without replacing its labels or inventing unknown bin edges. An explicitly separate unified preparation policy derives three quartile cuts from TRAIN observed events only, with0/+infinity terminal bounds. Source code does not disclose the historical split-label generation protocol, so this extension is not represented as historical label reproduction. Input checks cover complete multi-slide patient ordering, exact provided labels, TRAIN-only quantiles, incompatible-FM rejection, feature mutation/overwrite and immutable label-table bytes. Source and optimizer checks use the complete original network and optimizer definitions, not a miniature substitute.

## Complete patient training and precision

`convl_yaml_fp16_source_repaired` retains the released default FP16/autocast/GradScaler path and requires actual CUDA. `convl_yaml_fp32_reference` is explicitly named full FP32, including on CPU; it is never substituted for an FP16 run. Both preserve batch1, gradient accumulation2 (including dividing a final odd tail by2), Lookahead/RAdam at2e-4, weight decay1e-5 with the original bias/normalization partition, maximum200epochs, VAL-NLL-only strict-first-tie checkpoint selection and patience8. TEST is evaluated only from a saved checkpoint. Source event-to-censorship conversion occurs at the NLL boundary, and risk is negative summed survival.

The full source loss is NLL(alpha0) plus0.1 distribution alignment, minus0.5 instance clustering and0.3 concept clustering. Visual and concept hazards are blended as in the original code. There is no generic bag cap, gradient clipping, substitute frozen text, or extra survival objective. Exact GradScaler scale/growth/backoff/tracker history and skipped-overflow updates are bound to complete training traces and checked on restore. The shared engine preserves global Python/NumPy/Torch RNG state.

A complete original two-epoch source-method execution matches all network weights, fast/slow optimizer moments, counters, validation loss/risk and RNG exactly, including a3-patient2+1 accumulation tail. Actual disk checkpoint recovery (>1GiB) matches uninterrupted training in all of those states. The unified CLI executes full native model training, validation selection and two independent checkpoint reloads with identical predictions. These are synthetic software prechecks, not a real WSI or historical metric replication.

The accumulation reference is the actual `pytorch-lightning==2.3.0` wheel declared by the upstream requirements, SHA256 `b8eec361f4342ca628d0d8e6985511c9515435e4db62c5e982bb1c53a5a5140a`; its automatic optimization, training-epoch, precision and module source files were inspected separately without replacing the installed environment. The unified execution port currently uses Torch2.5.1, which differs from the historical Torch2.1.2 requirement and is disclosed rather than claimed bitwise equivalent across software versions.

## Usage and native acceptance

Prepare a standard patient survival bundle with native512D CONCH features, complete slide identity, time/event labels and original pretrained weights. `milplus prepare-convl --data PATH --metadata METADATA.json --output MANIFEST.json` binds the original cohort concept table, immutable patient labels, feature hashes and the explicitly chosen binning policy. The metadata schema and required fields are enforced by `milplus.convl_data`; an ordinary unlabeled feature folder is insufficient.

```bash
milplus check --data /path/to/patient_bundle --method ConVL --deep
milplus run --data /path/to/patient_bundle --method ConVL --device cuda:0 --output /path/to/run
milplus run --data /path/to/patient_bundle --method convl_yaml_fp32_reference --output /path/to/fp32_run
milplus evaluate --run /path/to/run --device cuda:0
```

`validate_convl_wheel.py` requires four actual CUDA conditions: full FP16 and FP32 for both BRCA and COADREAD. Each retains all112M parameters,12text layers,16prompt tokens, native512D,20patients,257patches per slide (including complete514patch multi-slide patient bags),200epoch limit and the original patience8 stopping rule. It checks every mechanism, complete fast/slow/scaler state, NLL-only selection and two real independent reloads. A CPU all-profile run must supply a verified current-source four-case CUDA receipt before the FP16 configuration is credited; unsupported CPU half precision is rejected.

For temporary synthetic verification only, `--metadata-only` hashes every completed weight file and removes it only after that case has actually completed its entire budget and both reload checks. Full prediction/report/input files and weight hashes remain; deleted synthetic weights require rerunning for future inference. This option does not alter training budgets or original model assets and is not the default for ordinary user runs.

The four full CUDA budgets, whole-library and final-wheel release gates remain pending. No real WSI experiment, paper-score reproduction or formal method acceptance is claimed.


## Version0.83 strict CUDA compatibility port

The first full0.82 CUDA attempt failed during the first backward pass: PyTorch2.5.1 cumprod backward internally uses CUDA cumsum, which rejects strict deterministic algorithms. All four failures and144original native input files were preserved. The successor keeps strict determinism enabled. Native cumulative-product forward bytes are unchanged; its four-bin CUDA backward now evaluates the same derivative in a fixed order, including one/multiple exact zeros. CPU still uses unmodified native autograd, preserving the original bitwise source-loop comparisons. The derivative follows the [PyTorch2.5.1 implementation derivation](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ReduceOps.cpp).

All32 actual CUDA dtype/zero cases passed an independent native backward oracle; maximum absolute differences were5.96e-8(float32),4.88e-4(float16),3.91e-3(bfloat16),1.11e-16(float64). These are explicit floating-point reduction-order differences, not a cross-version bitwise-gradient claim. Analytic double-precision Jacobian/zero cases and gradcheck passed. Full four-condition native budgets must run again for the new source hash; kernel checks do not replace them.
