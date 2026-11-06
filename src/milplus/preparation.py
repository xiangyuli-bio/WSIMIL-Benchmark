"""Explicit preparation of raw WSI manifests and frozen-feature task bundles."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .io import digest_file, write_json


def prepare_tcga(inventory, split_files, project, labels, output):
    """Preserve existing case splits; never infer molecular status from a filename."""
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    table = pd.read_csv(inventory, dtype=str, keep_default_na=False)
    table = table[(table.project_id == project) & (table.slide_type_code.str.upper() == "DX")].copy()
    table = table[table.OncoTreeCode.isin(labels)].copy()
    if table.empty:
        raise ValueError("No DX slides with requested labels")
    if len(labels) != len(set(labels)) or len(labels) < 2:
        raise ValueError("At least two distinct labels are required")
    split_map, evidence = {}, {}
    for subset, path in split_files.items():
        source = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "case_id" not in source:
            raise ValueError(f"{path}: expected case_id")
        for patient in source.case_id.unique():
            if not patient:
                raise ValueError("Empty patient ID in source split")
            if patient in split_map and split_map[patient] != subset:
                raise ValueError(f"Source split leakage: {patient}")
            split_map[patient] = subset
        evidence[subset] = {"path": str(Path(path).resolve()), "sha256": digest_file(path)}
    table["subset"] = table.case_id.map(split_map)
    if table.subset.isna().any():
        raise ValueError("Some cases have no source split; do not silently randomize or discard")
    if table.slide_id.duplicated().any():
        raise ValueError("Duplicate raw slide identifiers")
    if (table.groupby("case_id").OncoTreeCode.nunique() > 1).any():
        raise ValueError("Conflicting case labels")
    for subset in ("train", "val", "test"):
        if set(table[table.subset == subset].OncoTreeCode) != set(labels):
            raise ValueError(f"{subset}: missing class after raw WSI filtering")
    root.mkdir(parents=True)
    slides = table.rename(columns={"case_id": "patient_id", "OncoTreeCode": "label"})
    slides["global_patient_id"] = "TCGA:" + slides.patient_id
    slides[["slide_id", "patient_id", "global_patient_id", "label", "wsi_path"]].to_csv(root / "slides.csv", index=False)
    slides[["patient_id", "subset"]].drop_duplicates().to_csv(root / "split.csv", index=False)
    # TRIDENT accepts absolute WSI paths in its custom list; verify with pinned integration.
    pd.DataFrame({"wsi": table.wsi_path}).to_csv(root / "wsis.csv", index=False)
    config = {"schema_version": 1, "task_id": project.lower() + "_" + "_".join(labels).lower(),
              "task_type": "classification", "classes": labels, "synthetic": False,
              "slides": "slides.csv", "splits": {"fold0": "split.csv"}, "defaults": {"split": "fold0", "fm": None},
              "encoders": {}, "evaluation_unit": "patient", "training_unit": "slide",
              "label_provenance": "Existing TCGA-OncoTree label table; clinical provenance review required",
              "label_review_status": "provisional", "wsi_qc_status": "inventory_only_not_decoded",
              "cohort": project, "inclusion": "DX with explicit requested OncoTree labels; no TS/BS/MS",
              "endpoint_note": "Morphology labels; COAD/MACR is NOT MSI status"}
    (root / "task.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(root / "preparation.json", {"inventory_sha256": digest_file(inventory), "source_splits": evidence,
               "slides": len(slides), "patients": slides.patient_id.nunique(),
               "counts": slides.groupby(["subset", "label"]).size().reset_index(name="slides").to_dict("records")})
    return config


def index_features(data, feature_dir, fm, dimension, weights, preprocessing, coords_dir=None):
    root, source = Path(data).resolve(), Path(feature_dir).resolve()
    task = yaml.safe_load((root / "task.yaml").read_text())
    if fm in task["encoders"]:
        raise ValueError(f"Encoder {fm} already registered; create a versioned encoder ID")
    slides = pd.read_csv(root / task.get("slides", "slides.csv"), dtype=str, keep_default_na=False)
    candidates = {}
    for extension in ("*.h5", "*.hdf5", "*.npy", "*.pt"):
        for path in source.rglob(extension):
            key = path.stem.upper()
            if key in candidates:
                raise ValueError(f"Ambiguous feature files for {key}: {path}, {candidates[key]}")
            candidates[key] = path
    from .data import load_array
    index = []
    for row in slides.to_dict("records"):
        path = candidates.get(row["slide_id"].upper())
        if path is None:
            raise ValueError(f"Missing feature file for {row['slide_id']}")
        arr = load_array(path)
        if arr.ndim != 2 or arr.shape[1] != dimension or not len(arr) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid feature array: {path} {arr.shape}")
        record = {"slide_id": row["slide_id"], "path": str(path), "sha256": digest_file(path), "instances": len(arr)}
        if coords_dir:
            coord_files = [p for p in Path(coords_dir).glob("*.h5") if p.stem.upper() == row["slide_id"].upper()]
            if len(coord_files) != 1:
                raise ValueError(f"Expected one coordinate file for {row['slide_id']}")
            coords = load_array(coord_files[0], "coords")
            if coords.shape != (len(arr), 2):
                raise ValueError(f"Coordinate length mismatch for {path}")
            record["coords_path"] = str(coord_files[0].resolve())
            record["coords_sha256"] = digest_file(coord_files[0])
        index.append(record)
    # Atomic metadata replacement; caller must preserve encoder preprocessing provenance.
    name = f"features_{fm}.csv"
    if not fm.replace("_", "").replace("-", "").isalnum():
        raise ValueError("FM ID must contain only letters, numbers, _ or -")
    pd.DataFrame(index).to_csv(root / name, index=False)
    task["encoders"][fm] = {"dimension": dimension, "index": name, "weights": weights,
                            "preprocessing": preprocessing, "coordinate_alignment": "same TRIDENT run required"}
    if not task["defaults"].get("fm"):
        task["defaults"]["fm"] = fm
    tmp = root / "task.yaml.tmp"
    tmp.write_text(yaml.safe_dump(task, sort_keys=False))
    tmp.replace(root / "task.yaml")
    return {"fm": fm, "indexed": len(index), "dimension": dimension}
