"""Synthetic native512RGB/fine-mask engineering fixtures, not pathology data."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .hybrid_supervision_data import KIND


def create_demo(output, weights, patches=16):
    if type(patches) is not int or patches < 16:
        raise ValueError('Use at least16 actual512RGB patches in each synthetic hybrid bag')
    root = Path(output);root.mkdir(parents=True, exist_ok=False);(root/'pixels').mkdir()
    generator = np.random.default_rng(102)
    slides, index, splits = [], [], []
    for i in range(6):
        sid = 'synthetic_'+str(i);label = i % 2
        path = root/'pixels'/(sid+'.h5')
        with h5py.File(path, 'x') as f:
            data = f.create_dataset('rgb', shape=(patches, 512, 512, 3), dtype=np.uint8,
                                    chunks=(1, 512, 512, 3), compression='lzf')
            masks = []
            for j in range(patches):
                coarse = generator.integers(50, 220, (32, 32, 3), dtype=np.uint8)
                pixels = coarse.repeat(16, axis=0).repeat(16, axis=1)
                mask = np.zeros((512, 512), dtype=np.uint8)
                if j % 3 or label:
                    pixels[144:368, 144:368] = np.array([155, 65, 170], dtype=np.uint8)
                    if label:
                        mask[144:368, 144:368] = 255
                data[j] = pixels
                if i < 2 and j == 0:
                    masks.append(mask)
            f.create_dataset('patch_ids', data=[sid+'/original_patch_'+str(j) for j in range(patches)],
                             dtype=h5py.string_dtype('utf-8'))
            f['coords'] = np.stack([np.arange(patches)*384, np.zeros(patches, dtype=np.int64)], axis=1).astype(np.int64)
            f['fine_indices'] = np.array([0] if masks else [], dtype=np.int64)
            f.create_dataset('fine_masks', data=np.stack(masks) if masks else np.zeros((0, 512, 512), dtype=np.uint8))
        slides.append({'slide_id': sid, 'patient_id': sid, 'global_patient_id': sid,
                       'label': ['negative', 'positive'][label]})
        index.append({'slide_id': sid, 'path': str(path)})
        splits.append({'patient_id': sid, 'subset': ['train', 'val', 'test'][i//2]})
    pd.DataFrame(slides).to_csv(root/'slides.csv', index=False)
    pd.DataFrame(index).to_csv(root/'pixels.csv', index=False)
    pd.DataFrame(splits).to_csv(root/'split.csv', index=False)
    task = {'schema_version': 1, 'task_id': 'synthetic_hybrid_full', 'synthetic': True,
        'task_type': 'classification', 'classes': ['negative', 'positive'], 'training_unit': 'slide',
        'evaluation_unit': 'slide', 'slides': 'slides.csv', 'label_provenance': 'Synthetic software fixture only',
        'defaults': {'fm': 'raw_rgb', 'split': 'fold0'}, 'splits': {'fold0': 'split.csv'},
        'encoders': {'raw_rgb': {'dimension': 3, 'index': 'pixels.csv'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    schema = {'schema_version': 1, 'kind': KIND, 'layout': 'NHWC', 'dtype': 'uint8', 'color': 'RGB',
        'patch_pixels': 512, 'stride_pixels': 384, 'mpp': .5, 'coordinate_frame': 'level0_top_left_pixels',
        'normalization': 'RGB_uint8_to_float32_div255_no_ImageNet_normalization',
        'pixel_key': 'rgb', 'patch_ids_key': 'patch_ids', 'coords_key': 'coords',
        'fine_indices_key': 'fine_indices', 'fine_masks_key': 'fine_masks',
        'image_provenance': 'Synthetic original512RGB software images, not clinical WSI',
        'patch_provenance': 'Synthetic distinct stored pixel blocks with actual512shape and384stride coordinates',
        'fine_annotation_provenance': 'Synthetic observed binary masks for TRAINonly software fixtures',
        'initial_weights': str(Path(weights).resolve())}
    (root/'schema.yaml').write_text(yaml.safe_dump(schema, sort_keys=False))
    from .data import Bundle
    from .hybrid_supervision_data import prepare
    prepare(Bundle(root), schema, root/'hybrid_manifest.json')
    task['encoders']['raw_rgb']['hybrid_supervision'] = {'manifest': 'hybrid_manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    Bundle(root).check(deep=True)
    return root
