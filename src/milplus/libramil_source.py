"""Complete privately loaded Libra-MIL network and original CONCH text path."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .focus_network import CONCH_COMMIT, TOKENIZER, WEIGHTS_SHA, native_text_model
from .io import digest_file
from .upstream import load_definitions, load_single, pinned_text, verified_source

COMMIT = 'f29921970bfa22d364e08ea1768b984b47e07a8d'


def checked_source():
    root, sha = verified_source('libramil')
    if sha != COMMIT:
        raise ValueError('Libra-MIL requires its complete reviewed author revision')
    return root, sha


def original_model():
    checked_source()
    return load_definitions('libramil', 'models/mil_ot.py',
        ['pairwise_cosine_distance', 'sinkhorn_ot', 'MIL_MultiPrompt_OTFusion'],
        {'torch': torch, 'nn': nn, 'F': F})


def native_conch(weights):
    """All150 real pretrained text tensors, original normalized CLS encode_text.

    The unused visual/decoder allocations are omitted. They are not used by
    Libra-MIL, whose text embeddings are computed once with no_grad. Factory
    initialization RNG differs; no cold-start seedwise factory claim is made.
    """
    root, sha = verified_source('conch_focus')
    if sha != CONCH_COMMIT:
        raise ValueError('Libra-MIL actual native CONCH dependency revision changed')
    model = native_text_model(Path(weights))
    text = pinned_text(root, sha, 'conch/open_clip_custom/coca_model.py')
    coca = next(x for x in ast.parse(text).body if isinstance(x, ast.ClassDef) and x.name == 'CoCa')
    methods = [x for x in coca.body if isinstance(x, ast.FunctionDef) and x.name in {'_encode_text', 'encode_text'}]
    if {x.name for x in methods} != {'_encode_text', 'encode_text'}:
        raise ValueError('Libra-MIL actual CONCH encode_text methods missing')
    scope = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(root/'conch/open_clip_custom/coca_model.py'), 'exec'), scope)
    for key in ['_encode_text', 'encode_text']:
        setattr(model, key, scope[key].__get__(model, type(model)))
    return model


def author_text_encoder(weights):
    checked_source()
    weights = Path(weights).resolve()
    if digest_file(weights) != WEIGHTS_SHA:
        raise ValueError('Libra-MIL requires the actual registered pretrained CONCH checkpoint')
    root, sha = verified_source('conch_focus')
    if (root/TOKENIZER).read_text() != pinned_text(root, sha, TOKENIZER):
        raise ValueError('Libra-MIL actual original tokenizer asset changed')
    tokenizer = load_single('conch_focus', 'conch/open_clip_custom/custom_tokenizer.py')

    def factory(name, filename):
        if name != 'conch_ViT-B-16' or Path(filename).resolve() != weights:
            raise ValueError('Libra-MIL original CONCH factory identity changed')
        return native_conch(weights), None

    return load_definitions('libramil', 'models/text_encoder.py', ['TextEncoder'],
        {'torch': torch, 'nn': nn, 'create_model_from_pretrained': factory,
         'get_tokenizer': tokenizer.get_tokenizer, 'tokenize': tokenizer.tokenize}).TextEncoder(512, str(weights))


def original_prompts():
    root, sha = checked_source()
    instance = root/'text_prompt/TCGA_RCC_instance_prompt.json'
    bag = root/'text_prompt/TCGA_RCC_two_scale_text_prompt.csv'
    for path in [instance, bag]:
        if path.read_text() != pinned_text(root, sha, str(path.relative_to(root))):
            raise ValueError('Libra-MIL original full ordered language descriptions changed')
    structs = json.loads(instance.read_text())
    all_bags = pd.read_csv(bag, header=None)[0].tolist()
    if (not isinstance(structs, list) or len(structs) != 46 or len(all_bags) != 6
            or any(not isinstance(x, str) or not x.strip() for x in structs+all_bags)):
        raise ValueError('Libra-MIL full released46structural/6two-scale descriptions required')
    return structs, all_bags[3:], [instance, bag]


def encode_original_prompts(weights):
    structural, bags, paths = original_prompts()
    encoder = author_text_encoder(weights)
    # Literal source main.py calls the two lists separately, with no prompt
    # shortening or ensembling. The encoder already uses original no_grad.
    structure = encoder(structural).detach().cpu()
    bag = encoder(bags).detach().cpu()
    if (structure.shape != (46, 512) or bag.shape != (3, 512)
            or not torch.isfinite(structure).all() or not torch.isfinite(bag).all()):
        raise ValueError('Libra-MIL native full text embeddings are invalid')
    return structure, bag, paths


def build_original(structural, bags, prototypes=4):
    if (structural.ndim != 2 or structural.shape[1] != 512 or not len(structural)
            or bags.ndim != 2 or bags.shape[1] != 512 or len(bags) < 2
            or not torch.isfinite(structural).all() or not torch.isfinite(bags).all()
            or type(prototypes) is not int or prototypes not in {4, 6}):
        raise ValueError('Libra-MIL requires complete original512D ordered structural/class text embeddings and native4orREADME6visual prototypes')
    return original_model().MIL_MultiPrompt_OTFusion(512, len(structural), prototypes, len(bags),
        structural, bags, pooling_type='attention', use_proj=True,
        ot_epsilon=.05, ot_iter=20, num_heads=8)
