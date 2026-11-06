# MG-Trans development adapter

This is an unadmitted development candidate. The stable repository remains on its
separately verified release. Synthetic software checks and native training
budgets are recorded independently; no TCGA or historical paper score is claimed.

The adapter uses the actual [author source](https://github.com/Jiangbo-Shi/MG-Trans)
at `a18d886b89ff0eddf5ae0b4ae16058623fe34dbc`. It preserves the released 192-wide,
three-head model, 768-wide MLPs, 512-anchor PAM, two-scale shared Transformer,
three-layer GENConv branch, semantic objective, and 32-slide rolling Renyi MI
objective. A generic cross-entropy loop cannot stand in for this recipe.

Prepare the usual task bundle first, including patient mapping, fixed splits,
labels, FM identity, and the primary coarse feature index. Then provide an
additional CSV with `slide_id,path`. Each path identifies an H5 archive containing
two distinct measured feature arrays and their actual coordinates. For example:

```json
{
  "schema_version": 1,
  "kind": "mgtrans_dual_scale_level0_alignment",
  "coordinate_frame": "level0_top_left_pixels",
  "coordinate_provenance": "Describe the actual WSI coordinate extraction",
  "weights_provenance": "FM checkpoint identity and output definition",
  "preprocessing": "Actual magnifications, pixel size, transforms and tissue QC",
  "patch_size_provenance": "Actual level0 coverage for both patch sizes",
  "levels": [
    {"magnification": 5, "patch_size_level0": 2048,
     "feature_key": "features_low", "coords_key": "coords_low"},
    {"magnification": 10, "patch_size_level0": 1024,
     "feature_key": "features_high", "coords_key": "coords_high"}
  ]
}
```

These sizes describe the source reference geometry; they are not universal WSI
scanner settings. The current registration requires one explicitly declared pair
of physical patch sizes across the bundle, aligned nonoverlapping coarse grids,
and measured fine coverage for every coarse patch. Preserve actual level0 units;
do not relabel resized coordinates as original level0 coordinates. Heterogeneous
scanner/input protocols need their own explicit preparation rather than a guessed
grid. The released graph retains its 2048-level0-pixel distance cutoff.

The primary coarse arrays and row order must exactly match the additional scale
archive. The preparation rejects missing/duplicate patches, orphan fine patches,
out-of-bounds coordinates, inconsistent labels/splits, or altered feature bytes.

```bash
python -m milplus prepare-mgtrans --data /data/task \
  --index /data/task/mgtrans_index.csv \
  --schema /data/task/mgtrans_schema.json \
  --output /data/task/mgtrans/manifest.json
```

Register the produced manifest under the selected encoder in `task.yaml`:

```yaml
mgtrans:
  manifest: mgtrans/manifest.json
```

Then use the common entry:

```bash
python -m milplus run --data /data/task --method MG-Trans \
  --config configs/mgtrans.yaml --output /data/runs/mgtrans --device cuda
```

The default recipe retains Adam at `1e-4`, weight decay `1e-5`, batch size one,
random sampling without replacement, 200 maximum epochs, and the author's latest
tie rule on validation error. Early stopping requires patience 20 and zero-based
epoch greater than 50. The declared source Plateau scheduler is never stepped in
the released trainer, so this adapter does not silently activate it.

The native loader uses four readers on CUDA and none on CPU. Explicit reader or
epoch overrides are recorded software experiments and cannot satisfy the native
admission gate. That separate gate requires four complete source budgets across
1024/1536 feature widths and two/three classes, with actual 512/640 coarse
patches, the full 512-anchor network, and the MI stage active every epoch.

The source repeats short bags to 512 using their existing features, coordinates
and coverage rows. This behavior is retained and disclosed. No synthetic physical
coordinates are generated for real inputs. See [the audit](MGTRANS_AUDIT.md) for
device/API repairs and the explicit HNSW reproducibility correction.
