"""Small synthetic text inputs and weights for software validation only."""
import json

import numpy as np
import pandas as pd
import torch
import yaml

from .demo import create_demo


def create_vision_demo(output, kind, patients=20, dimension=16, classes=2):
    root = create_demo(output, patients=patients, dimension=dimension, classes=classes)
    task = yaml.safe_load((root/'task.yaml').read_text())
    rng = np.random.default_rng(231)
    meta = {'classes': task['classes'], 'synthetic': True, 'model': 'synthetic_text_fixture',
            'weights': 'random synthetic; not pretrained', 'preprocessing': 'synthetic',
            'text_provenance': 'synthetic class descriptors; no clinical meaning'}
    if kind == 'concept_anchors':
        meta.update(embedding_space='synthetic_joint_space', groups=[])
        task['encoders']['toy']['embedding_space'] = meta['embedding_space']
        for name in task['classes']+['agnostic_tissue']:
            path = name+'.npy'
            np.save(root/path, rng.normal(size=(3, dimension)).astype(np.float32))
            meta['groups'].append({'name': name, 'path': path, 'prompts': [f'{name} concept {j}' for j in range(3)]})
    elif kind == 'learned_dual_scale_prompts':
        from .vila import CLIPText
        with torch.random.fork_rng():
            torch.manual_seed(33)
            model = CLIPText(width=16, layers=1, heads=2, length=48, vocabulary=64, dimension=dimension)
            torch.nn.init.normal_(model.positional_embedding, std=.01)
            torch.nn.init.normal_(model.text_projection, std=.1)
        torch.save(model.state_dict(), root/'text_weights.pt')
        tokens = np.zeros((2*classes, 48), dtype=np.int64)
        for i in range(2*classes):
            tokens[i, 0] = 62
            tokens[i, 1:20] = rng.integers(1, 60, 19)
            tokens[i, 20] = 63
        np.save(root/'tokens.npy', tokens)
        meta.update(token_ids='tokens.npy', checkpoint='text_weights.pt', text_heads=2,
            tokenizer='synthetic', sot_token_id=62, eot_token_id=63, prompt_order=['low', 'high'],
            prompts=[f'{scale} synthetic descriptor {name}' for scale in ('low', 'high') for name in task['classes']])
        (root/'high').mkdir()
        entries = []
        for row in pd.read_csv(root/'slides.csv').to_dict('records'):
            path = f"high/{row['slide_id']}.npy"
            np.save(root/path, rng.normal(size=(int(rng.integers(12, 23)), dimension)).astype(np.float32))
            entries.append({'slide_id': row['slide_id'], 'path': path})
        pd.DataFrame(entries).to_csv(root/'high.csv', index=False)
        task['encoders']['toy']['dual_scale'] = {'high_index': 'high.csv', 'low_field_of_view_um': 512.,
            'high_field_of_view_um': 128., 'provenance': 'independent synthetic bags'}
    else:
        raise ValueError('Unknown text demo')
    (root/'text.json').write_text(json.dumps(meta, indent=2)+'\n')
    task['vision_language'] = {'kind': kind, 'metadata': 'text.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task, sort_keys=False))
    return root
