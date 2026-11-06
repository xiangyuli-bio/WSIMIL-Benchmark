"""Private fixed-source DisPro factories and exact public BioBERT asset loader.

This is development infrastructure, not an admitted profile. No smaller language
model, synthetic pretrained parameters or generic fusion model is substituted.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import os
import random
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .io import digest_file, read_json
from .upstream import load_definitions, pinned_text, verified_source

COMMIT = '98473003de24e5a8f28143442d3dd48bd159aff1'
REPO = 'dmis-lab/biobert-base-cased-v1.2'
REVISION = '67c9c25b46986521ca33df05d8540da1210b3256'
ASSET_SHA = {
    'config.json': 'a51b82d83d616bffaea11e629c4c82cca1e71b71460f133d55ebb4444ee9582e',
    'vocab.txt': 'eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02',
    'pytorch_model.bin': '70681f9fcd099d9d3c17680e6b885d986b2684a07eba84673b8fbc2431b0bd25'}


def asset_root():
    root = Path(os.environ.get('MILPLUS_DISPRO_BIOBERT', Path.home()/'.cache/milplus/assets/dispro_biobert'))
    meta = read_json(root/'manifest.json')
    if meta['repo'] != REPO or meta['revision'] != REVISION:
        raise ValueError('DisPro requires the actual reviewed fixed public BioBERT-v1.2 asset')
    for name, sha in ASSET_SHA.items():
        if digest_file(root/name) != sha or meta['files'][name]['sha256'] != sha:
            raise ValueError('DisPro pretrained BioBERT bytes differ: '+name)
    return root


@lru_cache(maxsize=1)
def backbone():
    """One read-only native12-layer/768-weight object, shared without shrinking.

    All source uses refer to identical frozen pretrained weights. No BN, mutable
    running statistics or trainable language parameters are shared. Transformer
    dropout executes independently on every invocation using the actual RNG.
    Loading a constant pretrained asset is isolated from run initialization RNG.
    """
    from transformers import BertConfig, BertModel
    root = asset_root(); config = BertConfig.from_json_file(str(root/'config.json'))
    if (config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
            config.intermediate_size, config.vocab_size, config.max_position_embeddings) != (768,12,12,3072,28996,512):
        raise ValueError('DisPro BioBERT is not the native released architecture')
    config._attn_implementation = 'eager'
    with torch.random.fork_rng(devices=[]):
        model = BertModel(config, add_pooling_layer=False)
    saved = torch.load(root/'pytorch_model.bin', map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or any(not isinstance(k,str) or not torch.is_tensor(v) for k,v in saved.items()):
        raise ValueError('DisPro pretrained file must be a pure tensor state dictionary')
    weights = {k.removeprefix('bert.'):v for k,v in saved.items() if k.startswith('bert.')}
    # HF changed position/token-type constants from persisted to nonpersistent
    # buffers. They are derived constants, not pretrained missing parameters.
    for name in ['embeddings.position_ids','embeddings.token_type_ids']:
        weights.pop(name, None)
    # The public file also contains a BERT pooler. The author's
    # AutoModelForMaskedLM(...).bert has add_pooling_layer=False, and every
    # DisPro path calls .encoder directly; these two tensors are unused there.
    for name, shape in [('pooler.dense.weight',(768,768)),('pooler.dense.bias',(768,))]:
        if name in weights:
            if tuple(weights[name].shape) != shape:
                raise ValueError('Unexpected native BioBERT pooler tensor shape')
            weights.pop(name)
    model.load_state_dict(weights, strict=True)
    del saved, weights
    model.width, model.vocab_size, model.autoregressive, model.pool = 768,28996,False,'cls'
    model.requires_grad_(False); model.eval()
    return model


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import BertTokenizer
    return BertTokenizer(vocab_file=str(asset_root()/'vocab.txt'), do_lower_case=False)


def _fixed():
    root, sha = verified_source('dispro')
    if sha != COMMIT: raise ValueError('DisPro source changed from reviewed commit')
    return root, sha


def _definitions(relative, names=None, namespace=None, replacements=()):
    root, sha = _fixed()
    if names is None:
        names = [n.name for n in ast.parse(pinned_text(root, sha, relative)).body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    scope = {'torch':torch,'nn':nn,'F':F,'np':np,'OrderedDict':OrderedDict,
             'argparse':argparse,'random':random}
    scope.update(namespace or {})
    return load_definitions('dispro', relative, names, scope, replacements)


def configs(dimension=1024, omic_sizes=()):
    module = _definitions('utils/options.py', ['get_wsi_config','get_gene_config','get_prompt_config'])
    wsi = module.get_wsi_config(); wsi.in_embed_dim = dimension
    gene = module.get_gene_config(argparse.Namespace(signatures=None))
    gene.omics_size, gene.num_classes = list(omic_sizes), 4
    prompt = {k:module.get_prompt_config(k) for k in ['WSI','Omics']}
    return wsi, gene, prompt


def _register_prompt_constants(model):
    # Original .cuda() constants are otherwise omitted from state_dict and
    # model.to(device). Register their exact computed values as real buffers.
    for module in model.modules():
        for key in ['token_prefix_uncensored','token_prefix_censored',
                    'token_suffix_uncensored','token_suffix_censored','tokenized_prompts_attn',
                    'wsi_tokenized_prompts_attn','omics_tokenized_prompts_attn']:
            value = getattr(module, key, None)
            if torch.is_tensor(value) and key not in module._buffers:
                delattr(module, key); module.register_buffer(key, value.detach().clone())
    return model


def build_unimodal(modal, dimension=1024, omic_sizes=(), stage2_copy=False):
    wsi, gene, prompt = configs(dimension, omic_sizes)
    if modal not in {'WSI','Omics'}: raise ValueError('DisPro modality must be WSI or Omics')
    # Source initializes embeddings using the pretrained model in eval mode.
    # Existing trained modules later choose their own train/eval mode normally.
    pretrained = backbone(); pretrained.eval()
    relative = ('models/Incomplete/DisPro/CoOp_'+('WSI' if modal=='WSI' else 'Omics')+'.py'
                if stage2_copy else 'models/'+modal+'/'+('Coop_WSI_BioBert' if modal=='WSI' else 'Coop_PathTrans_BioBert')+'/network.py')
    utility = ('models/Incomplete/DisPro/'+('wsi_utils.py' if modal=='WSI' else 'omics_utils.py')
               if stage2_copy else 'models/'+modal+'/'+('Coop_WSI_BioBert' if modal=='WSI' else 'Coop_PathTrans_BioBert')+'/model_utils.py')
    functions = _definitions(utility, ['lock_params','tokenize'])
    pool = _definitions('models/WSI/Coop_WSI_BioBert/model_utils.py', ['topj_pooling','avg_pooling'])
    scope = {'load_pretrained_tokenizer':lambda _:tokenizer(), 'load_pretrained_bert':lambda _:pretrained,
             'lock_params':functions.lock_params,'tokenize':functions.tokenize,
             'topj_pooling':pool.topj_pooling,'avg_pooling':pool.avg_pooling}
    if modal == 'WSI':
        scope['WSI_head'] = _definitions('models/WSI/Coop_WSI_BioBert/model_utils.py',['WSI_head']).WSI_head
    else:
        gene_path = ('models/Incomplete/DisPro/omics_utils.py' if stage2_copy else
                     'models/Omics/Coop_PathTrans_BioBert/transformer.py')
        block = _definitions(gene_path,['SNN_Block']).SNN_Block
        scope['gene_encoder'] = _definitions(gene_path,['Transformer'],{'SNN_Block':block}).Transformer
    root, sha = _fixed(); text = pinned_text(root, sha, relative)
    replacements = [('extended_attention_mask.cuda()', 'extended_attention_mask.to(prompts.device)'),
        ('prompt_init_ids = torch.tensor(prompt_init_ids)',
         'prompt_init_ids = torch.tensor(prompt_init_ids, device=next(token_embedding.parameters()).device)'),
        ('tokenized_prompts_ids = torch.tensor(tokenized_prompts_ids)',
         'tokenized_prompts_ids = torch.tensor(tokenized_prompts_ids, device=next(token_embedding.parameters()).device)'),
        ('self.tokenized_prompts_attn = torch.concat((ctx_attn, tokenized_prompts_attn), dim=-1)',
         'self.tokenized_prompts_attn = torch.concat((ctx_attn, tokenized_prompts_attn), dim=-1).to(next(token_embedding.parameters()).device)')]
    replacements = [pair for pair in replacements if pair[0] in text]
    if '.cuda()' in text: replacements.append(('.cuda()',''))
    marker = 'from .model_utils import WSI_head as modal_encoder'
    if modal == 'WSI' and marker in text: replacements.append((marker,'modal_encoder = WSI_head'))
    # Definitions refer to one another through the same isolated namespace.
    module = _definitions(relative, ['PromptLearner','ClassStrEncoder','CustomCLIP','CoOp'], scope, replacements)
    with contextlib.redirect_stdout(io.StringIO()):
        model = (module.CoOp(REPO,prompt['WSI'],wsi) if modal=='WSI' else
                 module.CoOp(REPO,'PathTransMean',prompt['Omics'],gene))
    return _register_prompt_constants(model)


def build_multimodal(wsi, omics, omic_sizes, dimension=1024, repaired_mask=True):
    wsi_cfg, gene_cfg, prompts = configs(dimension, omic_sizes)
    pretrained = backbone()
    attention = _definitions('models/Incomplete/DisPro/model_utils.py',['Attn_Net_Gated']).Attn_Net_Gated
    pool = _definitions('models/WSI/Coop_WSI_BioBert/model_utils.py',['topj_pooling','avg_pooling'])
    scope = {'load_pretrained_model':lambda _:pretrained,
        'load_uni_models':lambda *args:(wsi,omics),'Attn_Net_Gated':attention,
        'topj_pooling':pool.topj_pooling,'avg_pooling':pool.avg_pooling}
    replacements = []
    if repaired_mask:
        replacements.append(('attention_mask=tokens_mask,',
            'attention_mask=(1-tokens_mask[:,None,None,:])*torch.finfo(tokens.dtype).min,'))
    module = _definitions('models/Incomplete/DisPro/network.py',['Transformer'],scope,replacements)
    settings = {'path_model_wsi':'bound_in_memory_selected_UniPro_WSI',
        'path_model_omics':'bound_in_memory_selected_UniPro_Omics', 'prompt_config':prompts,
        'gene_config':gene_cfg,'wsi_config':wsi_cfg}
    with contextlib.redirect_stdout(io.StringIO()):
        result = module.Transformer(settings,list(omic_sizes),'BioBERT',4,max_length=512,
                                   n_WSI=dimension,dim_token=768,fine_tune=False)
    return _register_prompt_constants(result)


@lru_cache(maxsize=1)
def survival_loss():
    module = _definitions('utils/loss.py',['nll_loss','NLLSurvLoss'])
    return module.NLLSurvLoss(alpha=0.)


def original_holder_optimizer(model):
    """Literal source control: .data creates distinct leaves with no gradients."""
    return torch.optim.Adam([
        {'params':model.holder_path.data,'lr':2e-4},
        {'params':model.holder_omic.data,'lr':1e-5},
        {'params':model.wsi_adapter.parameters(),'lr':2e-4},
        {'params':model.omics_adapter.parameters(),'lr':1e-5},
        {'params':model.wsi_attn_head.parameters(),'lr':2e-4},
        {'params':model.omics_attn_head.parameters(),'lr':1e-5},
        {'params':model.linear_head.parameters(),'lr':2e-4},
        {'params':model.classifier.parameters(),'lr':2e-4}])
