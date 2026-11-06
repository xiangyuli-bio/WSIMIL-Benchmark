"""Complete original ConVL model and native pretrained contextual text path.

The registered development profiles retain the complete survival recipe and
patient input binding. Full native and release acceptance are separate gates.
"""
import ast
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .focus_network import CONCH_COMMIT, WEIGHTS_SHA, native_text_model
from .convl_scan import survival_cumprod
from .io import digest_file
from .upstream import load_definitions, load_single, pinned_text, verified_source

COMMIT = 'f1017a7ea0f51ff8d436b1334f937b85aa86c46b'
CONCEPT_PATH = 'models/text_prompt/concepts_patch.json'
FORK_ROOT = 'external/conch/open_clip_custom/'
TOKENIZER = FORK_ROOT+'tokenizers/conch_byte_level_bpe_uncased.json'
NAMES = ['KL_Divergence_Loss', 'SinkhornLoss', 'Attn_Net_Gated', 'CrossAttentionAdapter',
         'MetaNet', 'PromptLearner', 'InstanceConditionedPromptLearner', 'TextEncoder', 'ConVL']


def checked_source():
    root, commit = verified_source('convl')
    if commit != COMMIT:
        raise ValueError('ConVL requires the reviewed original author source')
    return root, commit


def native_contextual_text_model(weights):
    """Actual full pretrained12layer tower with the author's tuple-return API.

    The original vision and multimodal decoder are never called by ConVL's
    cached-feature forward. Only those unused allocations are omitted; all text
    tensors and their original trainability are retained. Unused factory RNG
    draws differ, so this does not assert cold-start seedwise factory equality.
    """
    root, commit = checked_source()
    canonical, canonical_commit = verified_source('conch_focus')
    if canonical_commit != CONCH_COMMIT:
        raise ValueError('ConVL actual native CONCH reference revision changed')
    for name in ['transformer.py', 'model_configs/conch_ViT-B-16.json']:
        if pinned_text(root, commit, FORK_ROOT+name) != pinned_text(canonical, canonical_commit, 'conch/open_clip_custom/'+name):
            raise ValueError('ConVL vendored full text architecture differs from the verified native tower')
    # This loader binds all150 actual text tensors and the exact original
    # 802235437byte checkpoint; none are frozen here.
    model = native_text_model(Path(weights))
    tree = ast.parse(pinned_text(root, commit, FORK_ROOT+'coca_model.py'))
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CoCa')
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'_encode_text', 'encode_text'}]
    if {node.name for node in functions} != {'_encode_text', 'encode_text'}:
        raise ValueError('ConVL native contextual encode_text methods are missing')
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(root/(FORK_ROOT+'coca_model.py')), 'exec'), namespace)
    for name in ['_encode_text', 'encode_text']:
        setattr(model, name, namespace[name].__get__(model, type(model)))
    return model


def author_classes(weights):
    root, commit = checked_source()
    if (root/TOKENIZER).read_text() != pinned_text(root, commit, TOKENIZER):
        raise ValueError('ConVL original local tokenizer asset changed')
    tokenizer = load_single('convl', FORK_ROOT+'custom_tokenizer.py')
    weights = Path(weights).expanduser().resolve()
    if digest_file(weights) != WEIGHTS_SHA:
        raise ValueError('ConVL requires the actual registered pretrained CONCH weights')

    def factory(name, path):
        if name != 'conch_ViT-B-16' or Path(path).resolve() != weights:
            raise ValueError('ConVL native CONCH factory parameters changed')
        return native_contextual_text_model(weights), None

    # Preserve original forward equations. The four-bin CUDA backward port
    # avoids the unsupported strict-deterministic scan; native forward bytes
    # and CPU autograd remain unchanged. Import/device routing is explicit.
    replacements = [
        ('S = torch.cumprod(1 - hazards, dim=1)', 'S = survival_cumprod(1 - hazards, dim=1)'),
        ('from external.conch.open_clip_custom import get_tokenizer, tokenize', ''),
        ('from external.conch.open_clip_custom import create_model_from_pretrained', ''),
        ("conch_ckpt = os.environ.get('CONCH_CKPT_PATH')", 'conch_ckpt = '+repr(str(weights))),
        ('clip_model = clip_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))',
         'clip_model = clip_model.to(torch.device("cpu"))')]
    return load_definitions('convl', 'models/ConVL/ConVL.py', NAMES,
        {'torch': torch, 'nn': nn, 'F': F, 'json': json, 'os': __import__('os'), 'survival_cumprod': survival_cumprod,
         'has_CONCH': lambda: True, 'create_model_from_pretrained': factory,
         'get_tokenizer': tokenizer.get_tokenizer, 'tokenize': tokenizer.tokenize}, replacements)


def original_concepts(dataset):
    """Read the actual full ordered author concepts without shortening prompts."""
    root, commit = checked_source()
    path = root/CONCEPT_PATH
    if path.read_text() != pinned_text(root, commit, CONCEPT_PATH):
        raise ValueError('ConVL original concept dictionary changed')
    key = str(dataset).upper()
    if key == 'COADREAD':
        key = 'CO&RE'
    if key not in json.loads(path.read_text()):
        raise ValueError('ConVL requires a supported explicit original concept cohort')
    return path, key


def build_original(weights, dataset='BLCA', **kwargs):
    """Full author native512->256/16dynamic-prompt model, before trainer binding."""
    concepts, key = original_concepts(dataset)
    # The released YAML declares768 for an unused fc_patch projection. Actual
    # forward uses fc_concept on native512 CONCH for both visual and text input.
    # Preserve that allocated-but-unused projection and its initialization draws.
    settings = {'dim_in': 768, 'emb_dim': 256, 'n_protos': 16, 'n_classes': 4,
        'dropout': .25, 'text_prompt_path': str(concepts), 'dataset_name': key,
        'survival': True, 'concept_weight': .5, 'w_er': .1, 'w_clst': -.5,
        'w_concept_clst': -.3, 'encoder_model': 'CONCH_V1', 'use_ot_loss': False,
        'concept_agg_mode': 'mean', 'attention_temperature': .1,
        'use_distribution_alignment': True, 'use_cross_attention': True,
        'use_instance_conditioned_prompt': True}
    unknown = set(kwargs)-set(settings)
    if unknown:
        raise ValueError('Unknown original ConVL settings: '+str(sorted(unknown)))
    settings.update(kwargs)
    if settings['dim_in'] != 768 or settings['emb_dim'] != 256 or settings['n_protos'] != 16:
        raise ValueError('ConVL full native CONCH512/embedding256/prompt16 dimensions are required')
    model = author_classes(weights).ConVL(**settings)
    prompt_length = model.prompt_learner.token_prefix.shape[1]+16+model.prompt_learner.token_suffix.shape[1]
    if (model.prompt_learner.tokenized_prompts.argmax(-1) >= prompt_length).any():
        raise ValueError('ConVL source EOT pooling exceeds contextual prompt length; explicit protocol repair required')
    return model
