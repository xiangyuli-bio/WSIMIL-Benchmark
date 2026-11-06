"""Explicit text provenance and independently sampled dual-scale bags."""
import json
import math

import numpy as np
import pandas as pd
import torch

from .data import load_array, resolve
from .io import digest_file


def finite_matrix(value, dimension, label):
    array = np.asarray(value)
    if array.ndim != 2 or len(array) < 1 or array.shape[1] != dimension or not np.isfinite(array).all():
        raise ValueError(f'{label}: expected nonempty finite [N,{dimension}]')
    return torch.as_tensor(array, dtype=torch.float32)


class VisionLanguageData:
    def __init__(self, bundle):
        self.bundle = bundle
        config = bundle.encoder.get('vision_language', bundle.config.get('vision_language'))
        self.kind = config['kind']
        self.metadata_path = resolve(bundle.root, config['metadata'])
        self.metadata = meta = json.loads(self.metadata_path.read_text())
        if meta.get('classes') != bundle.classes:
            raise ValueError('Text classes must exactly match ordered task classes')
        for key in ('model', 'weights', 'preprocessing', 'text_provenance'):
            if not isinstance(meta.get(key), str) or not meta[key].strip():
                raise ValueError(f'Text metadata requires {key}')
        if bool(meta.get('synthetic', False)) != bool(bundle.config.get('synthetic', False)):
            raise ValueError('Text and task synthetic markers differ')
        self.files = [self.metadata_path]
        if self.kind == 'concept_anchors':
            if not meta.get('embedding_space') or meta['embedding_space'] != bundle.encoder.get('embedding_space'):
                raise ValueError('CATE requires explicitly matching image/text embedding_space')
            groups = meta['groups']
            if len(groups) < len(bundle.classes) or [g['name'] for g in groups[:len(bundle.classes)]] != bundle.classes:
                raise ValueError('Concept groups must begin with ordered class names')
            if len({g['name'] for g in groups}) != len(groups):
                raise ValueError('Duplicate concept group name')
            self.anchors = []
            for group in groups:
                path = resolve(bundle.root, group['path'])
                features = finite_matrix(load_array(path), bundle.dim, group['name'])
                if not group.get('prompts') or len(group['prompts']) != len(features):
                    raise ValueError('Every anchor requires its exact input prompt')
                if not all(isinstance(t, str) and t.strip() for t in group['prompts']):
                    raise ValueError('Empty concept prompt')
                self.files.append(path)
                self.anchors.append(features)
        elif self.kind == 'learned_dual_scale_prompts':
            self.tokens_path = resolve(bundle.root, meta['token_ids'])
            self.checkpoint_path = resolve(bundle.root, meta['checkpoint'])
            self.files += [self.tokens_path, self.checkpoint_path]
            raw = load_array(self.tokens_path)
            if raw.dtype.kind not in 'iu' or raw.ndim != 2 or raw.shape[0] != 2*len(bundle.classes):
                raise ValueError('ViLa needs integer token ids [2*classes,context_length]')
            self.tokens = torch.tensor(raw.astype(np.int64))
            self.prompts = meta['prompts']
            if len(self.prompts) != 2*len(bundle.classes) or not all(isinstance(t, str) and t.strip() for t in self.prompts):
                raise ValueError('ViLa needs one low- then high-scale prompt per ordered class')
            if meta.get('prompt_order') != ['low', 'high']:
                raise ValueError('ViLa prompt_order must be [low, high]')
            if meta.get('tokenizer') != 'openai_clip_bpe' and not meta.get('synthetic'):
                raise ValueError('ViLa requires original OpenAI CLIP BPE tokenization')
            if not meta.get('synthetic') and meta['model'] != 'OpenAI-CLIP-RN50':
                raise ValueError('Native ViLa requires OpenAI-CLIP-RN50 text weights')
            if not meta.get('synthetic'):
                if (bundle.encoder.get('model') != meta['model'] or
                        bundle.encoder.get('weights') != meta['weights']):
                    raise ValueError('Native ViLa image model/weights must match the declared RN50 text model/weights')
                from .text_preparation import clip_tokenize
                if not np.array_equal(raw, clip_tokenize(self.prompts)):
                    raise ValueError('Token ids do not match declared text prompts')
        else:
            raise ValueError('Unknown vision_language kind')
        for path in self.files:
            if not path.is_file():
                raise FileNotFoundError(path)
        for path, expected in meta.get('sha256', {}).items():
            if digest_file(resolve(bundle.root, path)) != expected:
                raise ValueError(f'Text input hash mismatch: {path}')


class DualScaleData:
    def __init__(self, bundle):
        self.bundle = bundle
        cfg = bundle.encoder['dual_scale']
        self.dim = cfg.get('high_dimension', bundle.dim)
        if type(self.dim) is not int or self.dim < 1:
            raise ValueError('Dual scale high_dimension must be a positive integer')
        low_fov = float(cfg['low_field_of_view_um'])
        high_fov = float(cfg['high_field_of_view_um'])
        if not all(math.isfinite(v) and v > 0 for v in (low_fov, high_fov)) or low_fov <= high_fov:
            raise ValueError('Dual scales require distinct positive low FOV > high FOV')
        if not cfg.get('provenance'):
            raise ValueError('Dual scale extraction provenance is required')
        path = resolve(bundle.root, cfg['high_index'])
        self.files = [path]
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not {'slide_id', 'path'} <= set(frame) or frame.slide_id.duplicated().any():
            raise ValueError('Dual scale index needs unique slide_id and path')
        if set(frame.slide_id) != set(bundle.rows.slide_id):
            raise ValueError('Dual scale slide identities differ')
        paths = [resolve(bundle.root, p).resolve() for p in frame.path]
        lows = {resolve(bundle.root, p).resolve() for p in bundle.rows.path}
        if len(set(paths)) != len(paths) or set(paths) & lows:
            raise ValueError('Dual scales must reference independent per-slide feature files')
        self.rows = frame.set_index('slide_id').to_dict('index')

    def load(self, slide_id):
        row = self.rows[slide_id]
        path = resolve(self.bundle.root, row['path'])
        if row.get('sha256') and digest_file(path) != row['sha256']:
            raise ValueError('Dual scale feature hash mismatch')
        return finite_matrix(load_array(path), self.dim, slide_id)

    def inventory(self, row, deep):
        path = resolve(self.bundle.root, self.rows[row['slide_id']]['path'])
        result = {'path': str(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
        if deep:
            result.update(instances=len(self.load(row['slide_id'])), sha256=digest_file(path))
        return result
