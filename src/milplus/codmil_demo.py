"""Explicitly synthetic complete CoD physical scales, real row maps and text."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .codmil_data import KIND, prepare
from .data import Bundle
from .demo import create_demo
from .io import write_json


def create_codmil_demo(output, patients=30, dimension=1024, classes=3, text_dimension=None):
    root = Path(create_demo(output, patients=patients, dimension=dimension, classes=classes))
    text_dimension = dimension if text_dimension is None else text_dimension
    if type(text_dimension) is not int or text_dimension < 1:
        raise ValueError('CoD demo requires a positive complete text dimension')
    cfg_path = root/'task.yaml'; cfg = yaml.safe_load(cfg_path.read_text())
    cfg['evaluation_unit'] = 'slide'
    encoder = cfg['encoders']['toy']
    encoder.update(model='synthetic_visual_FM', embedding_space='synthetic_codmil_joint',
        spatial={'unit': 'level0_pixels', 'origin': 'top_left', 'patch_size': 256,
                 'provenance': 'synthetic physical coarse256 / fine128 grid'})
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    bundle = Bundle(root); folder = root/'codmil_assets'; folder.mkdir()
    generator = np.random.default_rng(2060); index = []
    for row in bundle.rows.to_dict('records'):
        _x, coords = bundle.load(row, True)
        offsets = np.asarray([[0, 0], [128, 0], [0, 128], [128, 128]], dtype=np.int64)
        fine_coords = (coords.numpy().astype(np.int64)[:, None]+offsets[None]).reshape(-1, 2)
        features = generator.normal(0, 1, (len(fine_coords), dimension)).astype(np.float32)
        features[:, row['target']] += 1.25
        fine = folder/(row['slide_id']+'_fine.h5')
        with h5py.File(fine, 'w') as f:
            f['features'], f['coords'] = features, fine_coords
        mapping = folder/(row['slide_id']+'_map.npy')
        np.save(mapping, np.arange(len(fine_coords)).reshape(len(coords), 4))
        index.append({'slide_id': row['slide_id'], 'fine_path': str(fine.relative_to(root)),
                      'fine_coords_path': str(fine.relative_to(root)), 'map_path': str(mapping.relative_to(root))})
    pd.DataFrame(index).to_csv(folder/'index.csv', index=False)
    prompts = [{'role': role, 'class': name, 'text': f'Synthetic {role} morphology of {name}'}
               for role in ['low', 'high'] for name in bundle.classes]
    prompts += [{'role': 'normal', 'text': 'Synthetic normal tissue morphology'},
                {'role': 'normal', 'text': 'Synthetic normal stromal tissue'},
                {'role': 'excluded_last', 'text': 'Synthetic last source prompt, explicitly excluded'}]
    np.save(folder/'text.npy', generator.normal(0, 1, (len(prompts), text_dimension)).astype(np.float32))
    metadata = {'schema_version': 1, 'kind': 'codmil_ordered_text_embeddings', 'synthetic': True,
        'classes': bundle.classes, 'model': 'synthetic_text_FM', 'weights': 'synthetic',
        'preprocessing': 'synthetic vectors', 'text_provenance': 'Synthetic mechanical test, not a real text encoder',
        'embedding_space': 'synthetic_codmil_joint', 'dimension': text_dimension,
        'features': str((folder/'text.npy').relative_to(root)), 'prompts': prompts,
        'alignment_policy': 'matched_vlm' if text_dimension == dimension else 'learned_projection_extension',
        'image_model': encoder['model'], 'image_weights': encoder['weights'],
        'alignment_provenance': 'Explicit synthetic full-D/T projection experiment, no natural VLM alignment claim'}
    write_json(folder/'text.json', metadata)
    write_json(folder/'schema.json', {'schema_version': 1, 'kind': KIND, 'synthetic': True,
        'coordinate_unit': 'level0_pixels', 'origin': 'top_left', 'coarse_patch_size': 256,
        'fine_patch_size': 128, 'mapping_provenance': 'Synthetic explicit four-child physical grid',
        'fine_extraction_provenance': 'Independent synthetic fine features, not duplicated coarse features'})
    manifest = prepare(bundle, folder/'index.csv', folder/'schema.json', folder/'text.json', folder/'manifest.json')
    cfg['encoders']['toy']['codmil'] = {'manifest': str(Path(manifest).relative_to(root))}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root
