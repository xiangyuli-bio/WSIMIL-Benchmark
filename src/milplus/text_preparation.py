"""Prepare traceable local text assets; never download or execute serialized weights."""
import gzip
import html
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .io import digest_file, write_json
from .upstream import load_definitions


def clip_tokenize(prompts, length=77):
    try:
        import ftfy
        import regex
    except ImportError as exc:
        raise ImportError('Install mil-baseline-puls[visionlanguage] for CLIP text preparation') from exc
    definitions = load_definitions('openai_clip', 'clip/simple_tokenizer.py',
        ['default_bpe', 'bytes_to_unicode', 'get_pairs', 'basic_clean', 'whitespace_clean', 'SimpleTokenizer'],
        {'gzip': gzip, 'html': html, 'os': os, 'lru_cache': lru_cache, 'ftfy': ftfy, 're': regex})
    tokenizer = definitions.SimpleTokenizer()
    result = np.zeros((len(prompts), length), dtype=np.int64)
    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Text prompts must be nonempty strings')
        ids = [tokenizer.encoder['<|startoftext|>']] + tokenizer.encode(prompt.replace('_', ' ')) + [tokenizer.encoder['<|endoftext|>']]
        if len(ids)+16 > length:
            raise ValueError(f'Prompt {i} exceeds CLIP context after 16 learned tokens; shorten explicitly')
        result[i, :len(ids)] = ids
    return result


def prepare_text(kind, features, metadata, output):
    """Metadata supplies ordered classes, provenance and all original descriptors."""
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise FileExistsError('Text output must be a new directory')
    meta = json.loads(Path(metadata).read_text())
    for key in ('model', 'weights', 'preprocessing', 'text_provenance', 'classes'):
        if not meta.get(key):
            raise ValueError(f'Text preparation metadata requires {key}')
    if len(meta['classes']) < 2 or len(set(meta['classes'])) != len(meta['classes']):
        raise ValueError('At least two unique ordered classes are required')
    state = torch.load(features, map_location='cpu', weights_only=True)
    meta['input_weights_sha256'] = digest_file(features)
    meta['synthetic'] = bool(meta.get('synthetic', False))
    meta.pop('sha256', None)
    if kind == 'concept_anchors':
        if not isinstance(state, (list, tuple)) or len(state) != len(meta['groups']):
            raise ValueError('CATE features must contain one tensor per ordered concept group')
        if [g['name'] for g in meta['groups'][:len(meta['classes'])]] != meta['classes']:
            raise ValueError('Concept groups must begin with ordered task classes')
        dim = state[0].shape[-1]
        for value, group in zip(state, meta['groups']):
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or len(value) < 1 or value.shape[1] != dim or not torch.isfinite(value).all():
                raise ValueError('Concept arrays must be finite nonempty [N,D] with matching dimensions')
            if len(group['prompts']) != len(value) or not all(isinstance(p,str) and p.strip() for p in group['prompts']):
                raise ValueError('Provide the exact input prompt for every concept row')
        if not meta.get('embedding_space'):
            raise ValueError('Declare the image/text embedding_space')
        root.mkdir(parents=True)
        files = []
        for i, (value, group) in enumerate(zip(state, meta['groups'])):
            path = root/f'concept_{i}.npy'
            np.save(path, value.detach().cpu().float().numpy())
            group['path'] = str(path)
            files.append(path)
    elif kind == 'learned_dual_scale_prompts':
        if meta['model'] != 'OpenAI-CLIP-RN50' or meta['synthetic']:
            raise ValueError('ViLa text exporter requires actual OpenAI-CLIP-RN50 metadata; use demo for synthetic data')
        if len(meta['prompts']) != 2*len(meta['classes']) or meta.get('prompt_order') != ['low', 'high']:
            raise ValueError('ViLa needs low then high descriptors for each ordered class')
        tokens = clip_tokenize(meta['prompts'])
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict):
            raise ValueError('CLIP requires a tensor state dict')
        state = {k:v for k,v in state.items() if k.startswith(('transformer.', 'token_embedding.', 'positional_embedding', 'ln_final.', 'text_projection'))}
        # Validate the native architecture before writing any output.
        from .vila import CLIPText
        reference = CLIPText(512, 12, 8, 77, 49408, 1024)
        reference.load_state_dict(state, strict=True)
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Nonfinite text weights')
        root.mkdir(parents=True)
        torch.save(state, root/'text_weights.pt')
        np.save(root/'tokens.npy', tokens)
        meta.update(checkpoint=str(root/'text_weights.pt'), token_ids=str(root/'tokens.npy'),
                    tokenizer='openai_clip_bpe', text_heads=8, sot_token_id=49406, eot_token_id=49407)
        files = [root/'text_weights.pt', root/'tokens.npy']
    else:
        raise ValueError('Unknown text preparation kind')
    meta['sha256'] = {str(p):digest_file(p) for p in files}
    write_json(root/'text.json', meta)
    return root/'text.json'
