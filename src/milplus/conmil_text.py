"""Actual frozen native CONCH causal/CLS prompt encoding for original ConMIL."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .focus_network import CONCH_COMMIT, TOKENIZER, WEIGHTS_SHA, native_text_model
from .io import digest_file, digest_object, read_json, write_json
from .upstream import load_definitions, load_single, pinned_text, verified_source

KIND = 'conmil_native_conch_positive_negative_queries'


def default_weights():
    return Path(os.environ.get('MILPLUS_CONMIL_WEIGHTS',
        Path.home()/'.cache/milplus/assets/focus/conch.pth')).expanduser().resolve()


def native_encoder(weights):
    root, sha = verified_source('conch_focus')
    if sha != CONCH_COMMIT or (root/TOKENIZER).read_text() != pinned_text(root, sha, TOKENIZER):
        raise ValueError('ConMIL actual CONCH source or local tokenizer changed')
    model = native_text_model(Path(weights))
    tree = ast.parse(pinned_text(root, sha, 'conch/open_clip_custom/coca_model.py'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CoCa')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {'_encode_text', 'encode_text'}]
    if {node.name for node in methods} != {'_encode_text', 'encode_text'}:
        raise ValueError('ConMIL original CONCH text encoding definitions are missing')
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(root/'conch/open_clip_custom/coca_model.py'), 'exec'), namespace)
    for name in ['_encode_text', 'encode_text']:
        setattr(model, name, namespace[name].__get__(model, type(model)))
    tokenizer = load_single('conch_focus', 'conch/open_clip_custom/custom_tokenizer.py')
    return model.eval().requires_grad_(False), tokenizer


def validate_prompts(prompts):
    if not isinstance(prompts, dict):
        raise ValueError('ConMIL prompt specification must be explicit JSON metadata')
    if type(prompts.get('synthetic', False)) is not bool:
        raise ValueError('ConMIL synthetic prompt marker must be a boolean')
    for key in ['positive', 'negative']:
        if (not isinstance(prompts.get(key), list) or (key == 'positive' and not prompts[key])
                or any(not isinstance(text, str) or not text.strip() for text in prompts[key])
                or len(set(prompts[key])) != len(prompts[key])):
            raise ValueError('ConMIL requires ordered unique queries, with at least one positive query')
    for key in ['task_semantics', 'prompt_provenance']:
        if not isinstance(prompts.get(key), str) or not prompts[key].strip():
            raise ValueError('ConMIL actual prompt source and task semantics must be declared')
    review = prompts.get('review')
    if (not isinstance(review, dict)
            or review.get('policy') not in {'predefined_without_data_refinement', 'train_only_patch_retrieval'}
            or review.get('selection_subset') != 'train'
            or not isinstance(review.get('global_patient_ids'), list)
            or len(set(review['global_patient_ids'])) != len(review['global_patient_ids'])
            or any(not isinstance(value, str) or not value.strip() for value in review['global_patient_ids'])
            or not isinstance(review.get('provenance'), str) or not review['provenance'].strip()):
        raise ValueError('ConMIL prompt review/refinement must explicitly use TRAIN only')
    if review['policy'] == 'predefined_without_data_refinement' and review['global_patient_ids']:
        raise ValueError('Predefined ConMIL prompts cannot claim data-driven review patients')
    if review['policy'] == 'train_only_patch_retrieval' and (not review['global_patient_ids']
            or not isinstance(review.get('retrieval_receipt_sha256'), str)
            or len(review['retrieval_receipt_sha256']) != 64):
        raise ValueError('ConMIL train-only retrieval needs actual review identities and receipt hash')
    return prompts


def prepare_text(prompts, weights, output, device='cpu'):
    """Execute the actual author encode_texts on complete original CONCH text."""
    spec_path = Path(prompts).expanduser().resolve()
    spec = validate_prompts(read_json(spec_path))
    weights = Path(weights).expanduser().resolve()
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise FileExistsError('ConMIL text output exists; preserve it and prepare a new immutable version')
    # This is the entire actual native text tower, not FOCUS's alternative
    # unmasked/BOS text path. Original CoCa strips the final token, appends CLS,
    # applies causal+padding masks, pools CLS and normalizes its512D projection.
    model, tokenizer = native_encoder(weights)
    model.to(device)
    function = load_definitions('conmil', 'encode_text_queries.py', ['encode_texts'],
        {'torch': torch, 'np': np, 'get_tokenizer': tokenizer.get_tokenizer, 'tokenize': tokenizer.tokenize}).encode_texts
    values = {role: function(model, spec[role], device) if spec[role] else np.empty((0, 512), dtype=np.float32)
              for role in ['positive', 'negative']}
    if any(array.shape != (len(spec[role]), 512) or not np.isfinite(array).all()
           or not np.allclose(np.linalg.norm(array, axis=1), 1., rtol=2e-5, atol=2e-5)
           for role, array in values.items()):
        raise ValueError('ConMIL actual native CONCH returned invalid normalized prompt embeddings')
    root.mkdir(parents=True, exist_ok=False)
    for role, array in values.items():
        np.save(root/(role+'.npy'), array.astype(np.float32))
    source, sha = verified_source('conch_focus')
    meta = {'schema_version': 1, 'kind': KIND, 'prompts': spec,
        'input_spec': {'path': str(spec_path), 'sha256': digest_file(spec_path)},
        'model': 'conch_ViT-B-16', 'weights_sha256': WEIGHTS_SHA,
        'conch_commit': CONCH_COMMIT, 'tokenizer_sha256': digest_file(source/TOKENIZER),
        'source_encode_text_sha256': digest_object({'commit': sha,
            'coca_model': digest_file(source/'conch/open_clip_custom/coca_model.py'),
            'transformer': digest_file(source/'conch/open_clip_custom/transformer.py')}),
        'preparer_sha256': digest_file(Path(__file__)), 'encoder_port': 'Omit unused vision/decoder allocations only; execute original full pretrained TextTransformer and original CoCa/ConMIL encoding methods',
        'encoding': 'original_CONCH_causal_padding_CLS_project512_L2_normalized',
        'files': {role: {'path': str(root/(role+'.npy')), 'sha256': digest_file(root/(role+'.npy'))}
                  for role in ['positive', 'negative']},
        'synthetic': bool(spec.get('synthetic', False)), 'device': str(device)}
    meta['contract'] = digest_object(meta)
    write_json(root/'text.json', meta)
    return root/'text.json'


def load_text(path):
    path = Path(path).expanduser().resolve(); metadata = read_json(path)
    if (metadata.get('schema_version') != 1 or metadata.get('kind') != KIND
            or metadata.get('model') != 'conch_ViT-B-16' or metadata.get('weights_sha256') != WEIGHTS_SHA
            or metadata.get('conch_commit') != CONCH_COMMIT
            or type(metadata.get('synthetic')) is not bool
            or metadata.get('encoding') != 'original_CONCH_causal_padding_CLS_project512_L2_normalized'
            or metadata.get('preparer_sha256') != digest_file(Path(__file__))
            or metadata.get('contract') != digest_object({k: v for k, v in metadata.items() if k != 'contract'})):
        raise ValueError('ConMIL actual CONCH encoder/source/text asset identity changed')
    spec = validate_prompts(metadata['prompts'])
    if digest_file(metadata['input_spec']['path']) != metadata['input_spec']['sha256'] or read_json(metadata['input_spec']['path']) != spec:
        raise ValueError('ConMIL original prompt wording or review provenance changed')
    tensors, files = {}, [path, Path(metadata['input_spec']['path'])]
    for role in ['positive', 'negative']:
        record = metadata['files'][role]; file = Path(record['path'])
        if digest_file(file) != record['sha256']:
            raise ValueError('ConMIL actual encoded positive/negative query bytes changed')
        array = np.load(file, allow_pickle=False)
        if (array.shape != (len(spec[role]), 512) or array.dtype != np.float32
                or not np.isfinite(array).all()
                or not np.allclose(np.linalg.norm(array, axis=1), 1., rtol=2e-5, atol=2e-5)):
            raise ValueError('ConMIL text tensor no longer matches its actual prompts and native embedding space')
        tensors[role] = torch.from_numpy(array.copy())
        files.append(file)
    return SimpleNamespace(metadata=metadata, positive=tensors['positive'], negative=tensors['negative'],
                           files=files, contract=digest_file(path))
