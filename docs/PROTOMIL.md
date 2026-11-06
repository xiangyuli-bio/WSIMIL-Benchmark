# ProtoMIL: complete SAE, concept inspection and concept MIL

This adapter refers to the MICCAI 2025 concept ProtoMIL by ss-sun, not the
similarly named prototype-based PMIL. The source is loaded privately from
`ss-sun/ProtoMIL` at `3e3745d7e4fecccb24aa486bd2af8b0dc6493711`.
Its author-referenced Discover-then-Name SAE dependency is explicitly reviewed
at `03484baef77747596712cf971e402f77b9da5a9a`; the author did not pin that
revision. No unlicensed author code or pretrained weights are redistributed.

## Scope and original conditions

The original interface requires projected CONCH features with 512 columns,
unique ordered patch IDs, binary slide labels and patient-disjoint partitions.
UNI/UNI2 features with a coincidentally convenient shape cannot be renamed to
CONCH. A different encoder or more than two classes needs its own reviewed
variant. This is an additional annotation track: actual normal/tumor probe
regions and their RGB tiles/masks are required, not only slide labels.

There are four explicit profiles:

| Profile | SAE epochs | MIL epochs | Concept intervention | Selection |
|---|---:|---:|---|---|
| `protomil_concept_paper_patient_reference` | 60 | 200 | No | Best validation AUROC |
| `protomil_intervened_paper_patient_reference` | 60 | 200 | Supplied review | Best validation AUROC |
| `protomil_concept_released_defaults_patient_reference` | 200 | 1 | No | Final epoch |
| `protomil_intervened_released_defaults_patient_reference` | 200 | 1 | Supplied review | Final epoch |

`--method ProtoMIL` chooses the first profile. The paper's 60/200 setting is
kept separately from the released command parser's 200/1 defaults. A one-epoch
software test never counts as validation of either full native budget.

The complete stages retain:

* The original 512→2048 SAE, decoder gradient projection hook and post-step
  unit normalization, batch4096 with incomplete tails, activation L1=0.0003,
  reconstruction loss, AdamWithReset, metrics and resampler state.
* The original 80/20 SAE file split, drawn only from explicitly declared
  training patients across the chosen cohorts. Downstream validation/test
  patients are excluded from all SAE fitting. The author's resampler guard
  uses 500000 piece epochs; the adapter preserves that behavior rather than
  silently replacing it with an activation count. The original activation
  report rounds incomplete updates to4096; both the quirk and actual updates
  are recorded.
* Actual normal/tumor probing groups, NumPy per-patch normalization, separate
  full-group SAE forwards, positive-before-negative ordering and original
  descending NumPy top10 concept inspection. Actual supplied RGBs appear in
  the report; the software does not invent pathology explanations.
* Full 2048-concept gated MIL with cross entropy plus **mean** absolute
  classifier weight (coefficient1), Adam1e-4. No hidden dimensional reduction,
  concept truncation, artificial grid or replacement attention model is used.

The reference ports use separate deterministic TRAIN/VAL generators, local
logging, lossless float32 HDF5 caches and explicit held-out evaluation. The
original per-epoch TEST reporting is deferred, including its DataLoader RNG
side effect. Paper selection initializes from the first valid validation
measurement so an AUROC of0 still has a valid checkpoint. These changes are
recorded in each run. The isolated reviewed runtime uses Torch2.5.1 in place
of the author's Torch2.1.2; literal author numerical functions and complete
trainer comparisons are verified in this declared runtime, not a claim of
historical binary/environment identity.

## Install and acquire source

Use a separate Python environment; numerical pins differ from other methods.
Install `.[protomil]` (or the corresponding wheel extra), then:

```bash
export MILPLUS_SOURCE_ROOT=/your/owned/milplus-sources
milplus setup --method ProtoMIL --plan
milplus setup --method ProtoMIL
```

Sources can also be provided as verified full Git checkouts under
`MILPLUS_SOURCE_ROOT` with the aliases `protomil` and `protomil_dtn`.
`setup` obtains both at the reviewed commits. For manual acquisition, use
`fetch-source` separately for each alias with its corresponding destination.
No GPU-side public download is required once these inputs exist.

## Prepare real data once

A regular task bundle holds `task.yaml`, slide/patient/label CSV, patient-level
train/val/test splits and the real CONCH feature index. HDF5 feature bags must
contain `features` and aligned `patch_ids`; other supported arrays need an
explicit `patch_ids_path` column. Each bag needs at least two finite rows and
no zero-norm rows.

Create a schema following `examples/protomil_schema.yaml`. Every SAE source
points to a regular unbound task bundle. All of its TRAIN rows enter the SAE
population; repeated physical feature files, omitted main TRAIN slides and
held-out patient overlap are rejected. Use consistent `global_patient_id`
across cohorts. Renaming patients cannot establish independence.

The `probe_index` JSONL has one entry per **actual annotated TRAIN patch**:

```json
{"slide_id":"actual_slide","patch_index":3,"region_label":1,"image":"actual_patch.png","mask":"actual_region_mask.png"}
```

Alternatively each image/mask can use
`{"path":"actual.h5","dataset":"rgb","index":3}`. Supply true256×256
uint8 RGB and an integer256×256 region mask. Explicit mask values and
thresholds are declared in the schema. Region labels must come from these
annotations, not the slide-level tumor diagnosis. Native profiles require
exactly10000 normal and10000 tumor probes. Probe order is explicit and frozen.

```bash
milplus prepare-protomil --data /data/task --schema schema.yaml \
  --output /data/task/protomil_manifest.json
```

Bind the returned manifest to the corresponding encoder in `task.yaml`:

```yaml
encoders:
  conch_v1:
    dimension: 512
    index: features.csv
    protomil:
      manifest: protomil_manifest.json
```

Source metadata, actual input bytes, patient/partition joins, ordered patch
identities, probe references and annotation counts are checked. A recomputed
JSON checksum alone cannot change these original-source relationships.

## Train, review and evaluate

```bash
milplus run --data /data/task --method ProtoMIL --output /runs/proto --device cuda:0
milplus evaluate --run /runs/proto --device cuda:0
```

For the intervened profile, the first invocation fits SAE, prepares the actual
concept bank and writes `concepts/concept_inspection.html`. It then returns
`awaiting_concept_review` with `review_request.json`, before creating a MIL
trainer. Inspect the actual report and put your review at the requested path.
It must include the exact `sae_state_sha256`, `concept_bank_sha256`, supplied
`reviewer` and `provenance`, `removed_concepts` and descriptions under
`concept_names`. Concept IDs must belong to this fitted SAE and probe bank.
Do not reuse published neuron IDs after retraining or fabricate a reviewer.
The hash placeholders in the example prevent a new fit from overwriting an
old review. Then resume with the same configuration:

```bash
milplus run --data /data/task \
  --method protomil_intervened_paper_patient_reference \
  --output /runs/proto-reviewed --device cuda:0 --resume
```

`--stop-after N` stops after actual method progress units: an SAE WSI piece,
the concept preparation stage, or a MIL epoch. Immutable checkpoint payloads
and atomic pointers preserve models, gradients, both optimizers, SAE metrics,
resampler/counters, data orders, generators and RNG. Existing run directories
are not overwritten. Selected SAE/MIL bytes and actual review are checked
again during explicit validation/test inference.

## Explicit software fixture

```bash
milplus demo-protomil --output /tmp/proto-demo
# Small test budget; this is not either original full epoch/probe protocol.
printf '%s\n' 'sae_epochs: 1' 'mil_epochs: 2' 'probe_per_class: 5' > /tmp/proto-software.yaml
milplus run --data /tmp/proto-demo/task --method ProtoMIL \
  --config /tmp/proto-software.yaml --output /tmp/proto-software-run
milplus evaluate --run /tmp/proto-software-run
```

This fixture is marked synthetic, including its review decisions. Its RGBs
were not encoded by CONCH; numeric512D arrays test the frozen-feature
interface. No real WSI, BRCA/COAD experiment, pathologist validation or paper
score reproduction is implied.
