"""Synthetic actual512D spatial bags and explicit three-scale graph fixtures."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from .io import digest_file


def create_demo(output, patches=13):
    if type(patches) is not int or patches < 13:
        raise ValueError('Use at least13 actual synthetic nodes for K12')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'features').mkdir()
    (root/'graphs').mkdir()
    rng = np.random.default_rng(101)
    slides, index, splits, graphs = [], [], [], {}
    for i in range(6):
        name = 'synthetic_'+str(i)
        feature_path, coords_path = root/'features'/(name+'.npy'), root/'features'/(name+'_coords.npy')
        # Distinct true fixture coordinates and order, not features reshaped
        # into a guessed spatial graph. This construction is disclosed as a
        # synthetic fixture, never an assertion about the unpublished builder.
        coords = np.stack([np.arange(patches)*256, np.arange(patches)**2], axis=1).astype(np.int64)
        np.save(feature_path, rng.normal(0, .25, (patches, 512)).astype(np.float32))
        np.save(coords_path, coords)
        record = {'features_sha256': digest_file(feature_path), 'coordinates_sha256': digest_file(coords_path),
                  'nodes': patches, 'edges': {}}
        distances = ((coords[:, None, :].astype(np.float64)-coords[None, :, :])**2).sum(2)
        np.fill_diagonal(distances, np.inf)
        for k in [4, 8, 12]:
            near = np.argsort(distances, axis=1, kind='stable')[:, :k]
            pairs = np.stack([near.reshape(-1), np.repeat(np.arange(patches), k)], axis=1).astype(np.int64)
            path = root/'graphs'/(name+'_k'+str(k)+'.npy')
            np.save(path, pairs)
            record['edges'][str(k)] = {'path': str(path), 'sha256': digest_file(path)}
        graphs[name] = record
        slides.append({'slide_id': name, 'patient_id': name, 'global_patient_id': name,
                       'label': ['negative', 'positive'][i % 2]})
        index.append({'slide_id': name, 'path': str(feature_path), 'coords_path': str(coords_path)})
        splits.append({'patient_id': name, 'subset': ['train', 'val', 'test'][i//2]})
    pd.DataFrame(slides).to_csv(root/'slides.csv', index=False)
    pd.DataFrame(index).to_csv(root/'features.csv', index=False)
    pd.DataFrame(splits).to_csv(root/'split.csv', index=False)
    (root/'graphs.json').write_text(json.dumps({'schema_version': 1,
        'kind': 'mggmmil_original_precomputed_graphs', 'node_order': 'stored_feature_rows',
        'edge_layout': 'E_by_2_source_target', 'scales': [4, 8, 12], 'imaging_modality': 'UAF',
        'feature_encoder': 'ResNet18_ImageNet',
        'graph_provenance': 'Synthetic software fixture only: stable distance-ranked directed no-self KNN; not original unpublished author graph construction',
        'slides': graphs}, indent=2)+'\n')
    task = {'schema_version': 1, 'task_id': 'synthetic_mggmmil', 'synthetic': True,
        'task_type': 'classification', 'classes': ['negative', 'positive'],
        'training_unit': 'slide', 'evaluation_unit': 'slide', 'slides': 'slides.csv',
        'label_provenance': 'Synthetic software fixture only',
        'defaults': {'fm': 'synthetic_resnet18', 'split': 'fold0'}, 'splits': {'fold0': 'split.csv'},
        'encoders': {'synthetic_resnet18': {'dimension': 512, 'index': 'features.csv',
            'spatial': {'unit': 'level0_pixels', 'origin': 'top_left', 'patch_size': 256,
                        'provenance': 'Synthetic unique stored fixture node coordinates'},
            'mggmmil': {'manifest': 'graphs.json'}}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root
