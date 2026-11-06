# Center Embeddings: exact original reader, cached top-left representation

The original `MILdataset.norm_coord` uses a nominal `2**level` integer offset
to obtain the OpenSlide `read_region` top-left. These API anchors need not be
the exact geometric centers of measured noninteger OpenSlide footprints.
After pixels are read, coordinates do not enter the ResNet/FPN, top32 selector,
three losses, two optimizers, BN or bag/center inference.

The existing `level0_centers` interface remains unchanged. The explicit new
`level0_top_left_pixels` interface requires
`source_coordinate_policy=author_norm_coord_inverse_verified` and structured
actual WSI/level/size/coordinate-SHA provenance. It verifies every original row
by executing the pinned author `norm_coord`: `anchor=top_left+128*2**level`,
mult1/read_size256, then `norm_coord(anchor)==top_left` exactly. Anchors are
explicitly not claimed to be measured physical centers. No physical centers
are invented/rounded, no source coordinates or pixels are modified, and no
regular lattice or duplicated-patch workaround is introduced.

Real check: five genuine original WSIs, including three with noninteger
downsamples,29,137 original rows checked geometrically.15 first/middle/last
patches were independently read from original WSIs through the original
reader coordinates and matched cached RGB SHA256 exactly. This is a bounded
implementation check, not full856 readiness or classifier training evidence.
The native preparer still binds every full bag and original742/69/45 split
before admission. Raw RGB remains a separate input group from UNI.

Verification:26 new/old metadata, row preservation, loader, provenance and
invalid-input checks passed. The separate larger-host full source/training/CLI
suite retained K32,256RGB and complete two-epoch source comparisons; its first
run produced12 passes and2 CPU-reference mixed-device failures. The original
reference explicitly declares CPU but `DataParallel` automatically moved its
model to visible CUDA while labels remained CPU. A minimal actual-runtime
diagnostic reproduced this. Only those same two unchanged full source tests
were repeated with CUDA hidden from their CPU-reference processes. Both passed
with exact original weights, optimizer moments, BN and RNG comparisons. All14
original cases have now been verified case-by-case across the preserved,
scope-matched reports; the original2-failure XML stays intact. This is not a
claim of a green first run, full paper-budget validation or completed BRCA fit.

Development-host full/CLI test processes terminated with137 and no complete
report. Its14GiB cgroup limit is recorded, but its observed OOM counter was0;
the termination cause is not asserted from that alone. No reduced K, pixels
or epochs were used to obtain a pass.

The original active GPU package is not edited. A frozen metadata-only copy
`/root/milplus_brca_uni_20261105/code_center_top_left_20261106` differs only in
`src/milplus/center_embeddings_data.py`. The native preparer was replaced while
waiting for full856, preserving actual controller/fit/receiver/extractor PIDs.
Only Center uses this source copy; the other three native RGB entries retain
the original frozen code prefix. Center admission additionally requires all14
scope-matched original checks, complete data and source-byte identity.

Evidence in the BRCA shared experiment directory:

- `center_reader_metadata_tests.xml`
- `center_top_left_candidate/source_receipt.json`
- `center_top_left_candidate/real_reader_samples.json`
- `center_top_left_candidate/real_reader_equivalence.json`
- `center_top_left_candidate/development_full_test_attempt.json`
- `center_top_left_candidate/full_oracles_state_r2.json`
- `center_top_left_candidate/full_source_training_tests_r2.xml`
- `center_top_left_candidate/cpu_reference_dataparallel_diagnostic.json`
- `center_top_left_candidate/cpu_reference_followup_state.json`
- `center_top_left_candidate/full_original_validation_attestation.json` (all14 cases verified; XML SHA and case IDs independently rechecked)
- `native_rgb_preparer_v2_handoff.json`
