"""Synthetic fixtures are permanently marked synthetic in experiment records."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml


def create_demo(output, patients=36, dimension=16, classes=2, survival=False, seed=11, omics=False, multiscale=None, regression=False):
    if regression and survival:
        raise ValueError('Regression and survival demos are different tasks')
    if multiscale not in {None, 'aligned_centers', 'quadtree'}:
        raise ValueError('Unknown synthetic multiscale layout')
    if patients < 9 * classes:
        raise ValueError("Demo needs at least 9 patients per class")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "features").mkdir()
    rng = np.random.default_rng(seed)
    slides, features, splits = [], [], []
    for i in range(patients):
        patient = f"patient_{i:04d}"
        subset = ("train", "train", "train", "val", "test")[i // classes % 5]
        splits.append({"patient_id": patient, "subset": subset})
        target = i % classes
        for j in range(1 + i % 2):
            slide = f"{patient}_slide_{j}"
            n = int(rng.integers(20, 41))
            x = rng.normal(0, 1, (n, dimension)).astype(np.float32)
            x[:, target] += 1.25
            path = f"features/{slide}.h5"
            with h5py.File(root / path, "w") as f:
                f["features"] = x
                f["coords"] = np.column_stack([np.arange(n) * 256, np.zeros(n)]).astype(np.int64)
            slides.append({"slide_id": slide, "patient_id": patient, "label": f"class_{target}",
                           "time": float(patients - i + 1), "event": int(i % 3 != 0),
                           'width_level0': n*256, 'height_level0': 256})
            features.append({"slide_id": slide, "path": path})
    pd.DataFrame(slides).to_csv(root / "slides.csv", index=False)
    pd.DataFrame(features).to_csv(root / "features.csv", index=False)
    pd.DataFrame(splits).to_csv(root / "split.csv", index=False)
    task = {"schema_version": 1, "task_id": "synthetic_survival" if survival else "synthetic_classification",
            "synthetic": True, "task_type": "survival" if survival else "classification", "classes": [f"class_{i}" for i in range(classes)],
            "evaluation_unit": "patient", "training_unit": "patient" if survival else "slide",
            "label_provenance": "synthetic fixture", "slides": "slides.csv", "defaults": {"fm": "toy", "split": "fold0"},
            "splits": {"fold0": "split.csv"}, "encoders": {"toy": {"dimension": dimension, "index": "features.csv",
             "weights": "synthetic", "coordinate_frame": "synthetic", "preprocessing": "synthetic"}}}
    if not multiscale:
        task['encoders']['toy']['spatial'] = {'unit': 'level0_pixels', 'origin': 'top_left',
            'patch_size': 256, 'provenance': 'synthetic 1D regular grid; not real WSI coordinates'}
    if survival:
        task['time_unit'] = 'months'
    if omics:
        import json
        genes = [f'gene_{i}' for i in range(18)]
        molecular = rng.normal(size=(patients, len(genes)))
        frame = pd.DataFrame(molecular, columns=genes)
        frame.insert(0, 'global_patient_id', [s['patient_id'] for s in splits])
        frame.to_csv(root/'omics.csv', index=False)
        pathways = [{'name': f'synthetic_signature_{i}', 'features': genes[i*3:(i+1)*3]} for i in range(6)]
        (root/'pathways.json').write_text(json.dumps(pathways, indent=2)+'\n')
        task['omics'] = {'table': 'omics.csv', 'columns': genes, 'pathways': 'pathways.json',
                         'normalization': 'train_standard', 'preprocessing': 'synthetic Gaussian values; no clinical interpretation'}
    if regression:
        genes = ['synthetic_gene_A', 'synthetic_gene_B', 'synthetic_gene_C']
        frame = pd.DataFrame(rng.uniform(1, 100, (patients, 3)), columns=genes)
        frame.insert(0, 'global_patient_id', [s['patient_id'] for s in splits])
        frame.insert(1, 'project', 'synthetic_cohort')
        frame.to_csv(root/'expression.csv', index=False)
        task.update(task_id='synthetic_regression', task_type='regression', classes=[])
        task['regression_targets'] = {'table': 'expression.csv', 'columns': genes,
            'transform': 'log10_1p', 'preprocessing': 'synthetic positive values',
            'label_provenance': 'synthetic only; no clinical expression assays'}
    (root / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
    if multiscale:
        add_multiscale_demo(root, task, slides, rng, dimension, multiscale)
    return root


def add_multiscale_demo(root, task, slides, rng, dimension, kind):
    """Synthetic physical centers and independent features; never registered as real WSI."""
    levels = []
    tables = [[], [], []]
    fovs = [512., 256., 128.]
    for level, name in enumerate(('coarse', 'middle', 'fine')):
        (root/name).mkdir()
        levels.append({'name': name, 'field_of_view_um': fovs[level], 'index': name+'.csv'})
    for row in slides:
        centers = np.column_stack([np.arange(16)*1024., np.zeros(16)])
        for level, name in enumerate(('coarse', 'middle', 'fine')):
            entry = {'slide_id': row['slide_id'], 'path': f"{name}/{row['slide_id']}.h5"}
            if level and kind == 'quadtree':
                parent = np.repeat(np.arange(len(centers)), 4)
                offsets = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * fovs[level-1]/4
                centers = centers[parent] + np.tile(offsets, (len(centers), 1))
                entry['parent_path'] = f"{name}/{row['slide_id']}_parents.npy"
                np.save(root/entry['parent_path'], parent)
            features = rng.normal(size=(len(centers), dimension)).astype(np.float32)
            features[:, task['classes'].index(row['label'])] += 1.25
            with h5py.File(root/entry['path'], 'w') as f:
                f['features'], f['coords'] = features, centers
            tables[level].append(entry)
    for level, table in zip(levels, tables):
        pd.DataFrame(table).to_csv(root/level['index'], index=False)
    encoder = task['encoders']['toy']
    encoder['index'] = levels[0]['index']
    encoder['multiscale'] = {'kind': kind, 'coordinate_frame': 'micrometers',
                           'preprocessing': 'synthetic common centers and independent Gaussian features', 'levels': levels}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))


def create_region_demo(output, patients=20, dimension=16, semi=False):
    """Complete synthetic 4x4 grids; never infer such groups from arbitrary WSI bags."""
    root = create_demo(output, patients=patients, dimension=dimension, survival=True)
    task = yaml.safe_load((root/'task.yaml').read_text())
    task['encoders']['toy']['regions'] = {'coordinate_frame': 'level0_pixels', 'patch_stride': 256,
        'preprocessing': 'synthetic nonoverlapping 4x4 grids for software validation'}
    rng = np.random.default_rng(87)
    features = pd.read_csv(root/'features.csv')
    for i, row in features.iterrows():
        nregions = 1 + i % 3
        coords = np.array([[r*2048 + col*256, line*256] for r in range(nregions) for line in range(4) for col in range(4)])
        with h5py.File(root/row.path, 'w') as handle:
            handle['features'] = rng.normal(size=(16*nregions, dimension)).astype(np.float32)
            handle['coords'] = coords
        path = 'features/'+row.slide_id+'_regions.npy'
        np.save(root/path, np.repeat(np.arange(nregions), 16))
        features.loc[i, 'region_ids_path'] = path
    features.to_csv(root/'features.csv', index=False)
    if semi:
        splits = pd.read_csv(root/'split.csv')
        task['adversarial_training'] = {'mode': 'UD+LD', 'visible_training_patients': splits[splits.subset == 'train'].patient_id.tolist()[::2],
            'unlabeled_folds': 2, 'label_provenance': 'synthetic training-outcome masking only'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root


def create_hierarchy_demo(output, patients=30, dimension=16):
    """Synthetic three-fine/two-coarse task, unrelated to any clinical hierarchy."""
    root = create_demo(output, patients=patients, dimension=dimension, classes=3)
    path = root / 'task.yaml'
    task = yaml.safe_load(path.read_text())
    task['hierarchy'] = {'coarse_classes': ['coarse_A', 'coarse_B'],
        'fine_to_coarse': {'class_0': 'coarse_A', 'class_1': 'coarse_A', 'class_2': 'coarse_B'},
        'label_provenance': 'synthetic grouping for software checks only'}
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return root


def create_dual_scale_demo(output, patients=20, dimension=16, high_dimension=None):
    """Two independently sampled synthetic bags per slide, with no text resources."""
    root = create_demo(output, patients=patients, dimension=dimension)
    task = yaml.safe_load((root/'task.yaml').read_text())
    high_dimension = dimension if high_dimension is None else high_dimension
    if type(high_dimension) is not int or high_dimension < 2:
        raise ValueError('Synthetic high_dimension must be an integer >=2')
    (root/'high').mkdir()
    rng, records = np.random.default_rng(231), []
    for row in pd.read_csv(root/'slides.csv').to_dict('records'):
        path = f"high/{row['slide_id']}.npy"
        array = rng.normal(size=(int(rng.integers(12, 23)), high_dimension)).astype(np.float32)
        array[:, task['classes'].index(row['label'])] += 1.25
        np.save(root/path, array)
        records.append({'slide_id': row['slide_id'], 'path': path})
    pd.DataFrame(records).to_csv(root/'high.csv', index=False)
    task['encoders']['toy']['dual_scale'] = {'high_index': 'high.csv', 'low_field_of_view_um': 512.,
        'high_field_of_view_um': 128., 'high_dimension': high_dimension,
        'provenance': 'independent synthetic bags; random features, not extracted or pretrained'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root
