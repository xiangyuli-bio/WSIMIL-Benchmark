"""Complete pinned MUSE architecture, all language banks and explicit ports.

This module alone is not an admitted executable profile. It retains the full
released architecture/initialization/selection equations; source-vs-paper
differences are recorded rather than silently repaired.
"""
import math
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .focus_network import CONCH_COMMIT, TOKENIZER, WEIGHTS_SHA, native_text_model
from .io import digest_file
from .upstream import load_definitions, load_single, pinned_text, verified_source

COMMIT = '0bd0e1abb402e772ed71cf208cfbd4b870175a00'
COHORTS = {
    'camelyon_all': ('task_camelyon_all_binary', ['Normal lymph node', 'Metastatic lymph node']),
    'tcga_nsclc': ('task_tcga_lung_subtyping', ['Lung adenocarcinoma', 'Lung squamous cell carcinoma']),
    'tcga_brca': ('task_tcga_brca_subtyping', ['Invasive ductal carcinoma', 'Invasive lobular carcinoma'])}
MODEL_FILE = 'models/model_text_retrevial.py'
NAMES = ['initialize_weights', 'TextEncoder', 'PromptLearner', '_no_grad_trunc_normal_',
         'trunc_normal_', 'get_prototype_text_from_chatgpt', 'get_description_about_class',
         'Expert', 'TopkRouter', 'NoisyTopkRouter', 'SparseMoE', 'Text_Agumented_Model']


def checked_source():
    root, sha = verified_source('muse')
    if sha != COMMIT:
        raise ValueError('MUSE requires its reviewed complete author source revision')
    return root, sha


def bank_assets(cohort):
    root, sha = checked_source()
    if cohort not in COHORTS:
        raise ValueError('MUSE requires an explicit released class-ordered knowledge cohort')
    paths = [root/'text_prompt'/cohort/f'generated_new_{i}.csv' for i in [0, 1]]
    descriptions = []
    for path in paths:
        if path.read_text() != pinned_text(root, sha, str(path.relative_to(root))):
            raise ValueError('MUSE complete original knowledge bank bytes changed')
        # Literal author parsing: explicit header row and index column omitted.
        values = np.array(pd.read_csv(path, header=None, index_col=False).iloc[1:, 1:]).squeeze()
        if values.shape != (300,) or any(not isinstance(x, str) or not x.strip() for x in values):
            raise ValueError('MUSE requires all300 complete descriptions per original class')
        descriptions.append(values.tolist())
    return {'cohort': cohort, 'task': COHORTS[cohort][0], 'class_prompts': COHORTS[cohort][1],
            'paths': paths, 'sha256': {str(p): digest_file(p) for p in paths},
            'descriptions': descriptions}


def author_classes(weights):
    root, _ = checked_source()
    canonical, sha = verified_source('conch_focus')
    if sha != CONCH_COMMIT:
        raise ValueError('MUSE native complete CONCH dependency revision changed')
    if (canonical/TOKENIZER).read_text() != pinned_text(canonical, sha, TOKENIZER):
        raise ValueError('MUSE actual native tokenizer bytes changed')
    weights = Path(weights).resolve()
    if digest_file(weights) != WEIGHTS_SHA:
        raise ValueError('MUSE requires the actual registered pretrained CONCH weights')
    tokenizer = load_single('conch_focus', 'conch/open_clip_custom/custom_tokenizer.py')

    def factory(name, filename):
        if name != 'conch_ViT-B-16' or Path(filename).resolve() != weights:
            raise ValueError('MUSE original native CONCH factory identity changed')
        return native_text_model(weights), None

    replacements = [
        ("conch_checkpoint_path = '/data2/dongjiajun/code/xjh_few_shot_WSI_final/CONCH/checkpoints/conch/pytorch_model.bin'",
         'conch_checkpoint_path = '+repr(str(weights))) ]
    # Bind every original absolute CSV address to its actual released Git asset.
    for cohort in COHORTS:
        bank_assets(cohort)
        for index in [0, 1]:
            old = f'/data2/dongjiajun/code/xjh_few_shot_WSI_final/text_prompt/{cohort}/generated_new_{index}.csv'
            replacements.append((old, str(root/'text_prompt'/cohort/f'generated_new_{index}.csv')))
    for name, argument in [('text_base_0', 'embedding_0'), ('text_base_1', 'embedding_1'),
                           ('text_base', 'embedding_total'), ('class_without_learnable', 'class_without_learnable')]:
        replacements.append((f"self.{name} = self.text_encoder({argument}).to('cuda').detach()",
                             f"self.register_buffer('{name}', self.text_encoder({argument}).detach())"))
    # Only the unused diagnostic function retains hard-coded CUDA transfers.
    # Device routing lets the literal diagnostic run serve as an RNG oracle on
    # CPU as well; it never authorizes reading private sample paths.
    replacements.append((".to('cuda')", '.to(next(self.parameters()).device)'))
    return load_definitions('muse', MODEL_FILE, NAMES,
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'pd': pd, 'os': os,
         'pjoin': os.path.join, 'math': math, 'warnings': warnings,
         'get_tokenizer': tokenizer.get_tokenizer, 'tokenize': tokenizer.tokenize,
         'create_model_from_pretrained': factory}, replacements)


def build_original(weights, cohort='tcga_brca', dimension=512):
    if type(dimension) is not int or dimension not in {512, 768, 1024, 1536}:
        raise ValueError('MUSE requires an explicit source-supported feature width')
    assets = bank_assets(cohort)
    # Preserve original full300+300+600 text precomputation and the recursive
    # self.apply(initialize_weights), including its reinitialization of linear
    # and normalization layers in the text tower after the fixed caches exist.
    # Neither behavior may be replaced by a small/random text stand-in.
    model = author_classes(weights).Text_Agumented_Model(dataset=assets['task'],
        input_dim=dimension, num_classes=2, k=20, num_experts=8,
        num_selected=2, topk_ratio=.2)
    if (model.text_base_0.shape != (300, 512) or model.text_base_1.shape != (300, 512)
            or model.text_base.shape != (600, 512) or model.prompt_learner.ctx.shape != (16, 768)
            or len(model.moe.experts) != 8 or model.moe.top_k != 2):
        raise ValueError('MUSE complete native language/knowledge/expert dimensions changed')
    return model


def diagnostic_rng_port(model):
    """Exact random draws of the released diagnostic, without private I/O.

    This diagnostic never updates parameters or contributes to the objective.
    Its noisy-router draws and Python seed reset nevertheless affect future
    training. Preserve each original draw shape/order, not one flattened draw.
    The independent literal diagnostic with isolated synthetic I/O is the
    verification oracle; this port is not evidence for visualization fidelity.
    """
    if model.training or model.text_base_0.shape != (300, 512) or model.text_base_1.shape != (300, 512):
        raise ValueError('MUSE diagnostic RNG replay requires original eval mode and full300-entry banks')
    random.seed(2)
    random.sample(range(0, 1000), 50)
    tensor = next(model.parameters())
    torch.randn((2, 8), device=tensor.device, dtype=tensor.dtype)
    for _ in range(300):
        torch.randn((8,), device=tensor.device, dtype=tensor.dtype)
        torch.randn((8,), device=tensor.device, dtype=tensor.dtype)
    for _ in range(50):
        torch.randn((1, 2, 8), device=tensor.device, dtype=tensor.dtype)
