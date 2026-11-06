"""Pinned complete TaxoMIL author graph, objectives, collator and epoch functions.

No author code is redistributed: definitions execute from verified Git blobs.
The decoder is the real pretrained GPT2, not a synthetic/small language model.
"""
import random
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer, GPT2Model

from .io import digest_file
from .upstream import load_definitions, verified_source

COMMIT='5e095f0c212b3a30c3f6c6c81b3ca1a18e0c676e'
GPT2_REVISION='607a30d783dfa663caf39e06633721c8d4cfcd7e'
GPT2_SHA256='248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'


def checked_source():
    root,sha=verified_source('taxomil')
    if sha!=COMMIT:raise ValueError('TaxoMIL source revision changed')
    return root,sha


def author_components(repair_alignment_keys=True):
    checked_source()
    attention=load_definitions('taxomil','aggregator/ABMIL.py',['AttentionGated'],
        {'torch':torch,'nn':nn,'F':F}).AttentionGated
    decoder=load_definitions('taxomil','decoder.py',['MultiHeadGPT2','TaxoMIL','build_model'],
        {'torch':torch,'nn':nn,'random':random,'List':List,'Optional':Optional,
         'AutoConfig':AutoConfig,'AutoTokenizer':AutoTokenizer,'GPT2Model':GPT2Model,
         'AttentionGated':attention})
    losses=load_definitions('taxomil','utils/loss.py',
        ['TextHierarchicalLoss','AlignmentLoss','ContrastiveLoss'],
        {'torch':torch,'nn':nn,'F':F,'Dict':Dict,'List':List})
    scheduler=load_definitions('taxomil','utils/loss_scheduler.py',['PeriodicLossScheduler'],
        {'math':__import__('math'),'Tuple':__import__('typing').Tuple}).PeriodicLossScheduler
    utils=load_definitions('taxomil','utils/utils.py',
        ['get_optimizer','_resize_linear_out_features','_patch_unembedding_layer',
         'patch_unembedding_to_tokenizer','calculate_accuracies_shared_dual',
         'mean_pool_label_embeddings','build_text_bank_for_losses'],
        {'torch':torch,'nn':nn,'List':List,'Dict':Dict})
    collate=load_definitions('taxomil','dataloader/dataset.py',['custom_collate_fn'],
        {'pad_sequence':pad_sequence}).custom_collate_fn
    # Logging is disabled explicitly; computation, stochastic draws and updates
    # remain the literal released train/validate definitions.
    trainer=load_definitions('taxomil','trainer/train.py',
        ['compute_ital_loss','compute_optional_losses','train_one_epoch','validate'],
        {'torch':torch,'nn':nn,'Dict':Dict,'List':List,'tqdm':lambda iterable,**kw:iterable,
         'wandb':None,'build_text_bank_for_losses':utils.build_text_bank_for_losses,
         'calculate_accuracies_shared_dual':utils.calculate_accuracies_shared_dual},
        replacements=[('criteria[f"{head_type}_ital"]',
            'criteria[{"coarse": "c_ital", "fine": "f_ital"}[head_type]]')] if repair_alignment_keys else [])
    return decoder,losses,scheduler,utils,collate,trainer


def source_identity():
    root,_=checked_source()
    return {'url':'https://github.com/QuIIL/TaxoMIL','commit':COMMIT,
        'files':{name:digest_file(root/name) for name in ['decoder.py','aggregator/ABMIL.py',
            'utils/loss.py','utils/loss_scheduler.py','utils/utils.py','trainer/train.py',
            'dataloader/dataset.py','config/dataset_labels.json','config/prompt_templates.json']}}
