"""Original QPMIL-VL components; full continual trainer is a separate gate.

Private pinned Git-object loading avoids redistributing the author's ND code or
CONCH weights. No cached/random text replaces differentiable prompt encoding.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import yaml

from .io import digest_file
from .upstream import verified_source, pinned_text, load_definitions

COMMIT='3a7a7698582dec866d43eb748f8c3599f7be4391'


def original_root():
    root,commit=verified_source('qpmilvl')
    if commit!=COMMIT:raise ValueError('QPMIL-VL needs the exact reviewed original commit')
    return root,commit


@lru_cache(maxsize=1)
def definitions():
    root,commit=original_root()
    ns={'torch':torch,'nn':nn,'F':F,'OrderedDict':OrderedDict,'Callable':Callable,
        'Optional':Optional,'Sequence':Sequence,'Tuple':Tuple,'checkpoint':checkpoint}
    transformer=load_definitions('qpmilvl','models/conch/transformer.py',
        ['LayerNormFp32','LayerNorm','QuickGELU','LayerScale','AttentionalPooler','ResidualAttentionBlock','Transformer','TextTransformer','MultimodalTransformer'],ns)
    import timm
    if timm.__version__!='1.0.11':raise ValueError('QPMIL-VL full original CONCH construction requires its declared timm1.0.11 runtime')
    from timm.models.vision_transformer import VisionTransformer
    from timm.layers import Mlp,to_2tuple
    vision=load_definitions('qpmilvl','models/conch/vision_tower.py',['VisualModel'],
        {**ns,'AttentionalPooler':transformer.AttentionalPooler,'Mlp':Mlp,'to_2tuple':to_2tuple,'logging':logging})
    coca=load_definitions('qpmilvl','models/conch/coca_model.py',
        ['CoCaVisionCfg','CLIPTextCfg','_build_vision_tower','_build_text_tower','_build_text_decoder_tower','CoCa','resize_pos_embed'],
        {**ns,'Union':Union,'dataclass':dataclass,'np':np,'logging':logging,'VisualModel':vision.VisualModel,
         'VisionTransformer':VisionTransformer,'TextTransformer':transformer.TextTransformer,
         'MultimodalTransformer':transformer.MultimodalTransformer})
    from transformers import PreTrainedTokenizerFast
    tokenizer=load_definitions('qpmilvl','models/conch/custom_tokenizer.py',['get_tokenizer','tokenize'],
        {'PreTrainedTokenizerFast':PreTrainedTokenizerFast,'F':F,
         'TOKENIZER_DIR':root/'models/conch/tokenizers','DEFAULT_TOKENIZER':'conch_byte_level_bpe_uncased.json'})
    # Read a tracked blob and require identical tokenization material on disk.
    path=root/'models/conch/tokenizers/conch_byte_level_bpe_uncased.json'
    if path.read_text()!=pinned_text(root,commit,str(path.relative_to(root))):
        raise ValueError('Original CONCH local tokenizer bytes differ')
    model=load_definitions('qpmilvl','models/model_il.py',
        ['merge_parameter','TunableVLearner','CONCHPromptEncoder','PromptLearner','QPMIL_VL'],
        {**ns,'get_tokenizer':tokenizer.get_tokenizer,'tokenize':tokenizer.tokenize})
    utilities=load_definitions('qpmilvl','models/model_utils.py',['freeze_weight','set_tunable_v','EarlyStopping'],
        {'np':__import__('numpy')},replacements=[('np.Inf','np.inf')])
    return SimpleNamespace(transformer=transformer,tokenizer=tokenizer,model=model,utilities=utilities,coca=coca)


def config():
    root,commit=original_root()
    return yaml.safe_load(pinned_text(root,commit,'configs/main.yaml'))


def runtime_identity():
    import timm
    if timm.__version__!='1.0.11':raise ValueError('QPMIL-VL requires its declared isolated timm1.0.11 runtime')
    return {'timm':timm.__version__,'torch_actual':torch.__version__,'precision':'original_float32',
        'constructor':'complete_source_CoCa_including_frozen_visual_and_unused_decoder',
        'numerical_execution':{'policy':'strict_deterministic_original_Torch_operators',
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'warn_only':torch.is_deterministic_algorithms_warn_only_enabled()}}


def configure_runtime():
    """Configure the original operators; never replace the model/objective.

    Repeated prototype/prompt indices use CPU index_put accumulation in the
    original backward graph. Its default parallel reduction is nondeterministic
    even for identical inputs. Strict Torch deterministic execution selects a
    reproducible original operator implementation, retaining float32/threads.
    Component-level users share this explicit process execution setting. The
    public train/evaluate entries restore their caller's setting on exit.
    """
    runtime_identity()  # Check the declared dependency before changing flags.
    torch.use_deterministic_algorithms(True,warn_only=False)
    return runtime_identity()


@contextmanager
def reproducible_execution():
    previous=(torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        configure_runtime()
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0],warn_only=previous[1])


def load_base(weights,expected_sha256,device='cpu'):
    runtime_identity()
    path=Path(weights).expanduser().resolve()
    if len(expected_sha256)!=64 or digest_file(path)!=expected_sha256:
        raise ValueError('Actual authorized CONCH checkpoint hash mismatch')
    configure_runtime()
    d=definitions();root,commit=original_root()
    cfg=json.loads(pinned_text(root,commit,'models/conch/model_configs/conch_ViT-B-16.json'))
    cfg.pop('custom_text',None)
    # Construct the entire original CoCa, including the frozen vision tower and
    # unused decoder. Their initialization advances the source RNG before the
    # trainable key/prompt pools, even though MIL only calls the text branch.
    base=d.coca.CoCa(**cfg)
    checkpoint_values=torch.load(path,map_location='cpu',weights_only=True)
    state=checkpoint_values.get('state_dict',checkpoint_values)
    if any(k.startswith('module.') for k in state):state={k.removeprefix('module.'):v for k,v in state.items()}
    text={k[len('text.'):]:v for k,v in state.items() if k.startswith('text.')}
    base.text.load_state_dict(text,strict=True)
    if 'logit_scale' not in state or state['logit_scale'].shape!=torch.Size([]):
        raise ValueError('Original pretrained CONCH logit_scale is required')
    d.coca.resize_pos_embed(state,base)
    missing,unexpected=base.load_state_dict(state,strict=False)
    if unexpected or any(not k.startswith('text_decoder.') for k in missing):
        raise ValueError('Actual CONCH text/visual checkpoint is incomplete or incompatible')
    # Original factory allows the pretraining-only decoder to be absent. It is
    # initialized and frozen just as in the author code, and never used by MIL.
    base.original_missing_unused_decoder_keys=list(missing)
    base.to(device).eval();base.requires_grad_(False);base.dtype=base.logit_scale.dtype
    if base.dtype!=torch.float32:raise ValueError('Original full-float CONCH reference requires float32 weights')
    return base


def parameter_pools(cfg,base):
    # Literal original initialization order/distributions, including future
    # task vectors; only the current task vector becomes trainable.
    dev=base.logit_scale.device;dtype=base.dtype
    key=nn.ParameterList([nn.Parameter(.02*torch.randn(1,512,dtype=dtype,device=dev)) for _ in range(cfg['pool_size'])])
    prompt=nn.ParameterList([nn.Parameter(.02*torch.randn(1,cfg['prompt_length'],768,dtype=dtype,device=dev)) for _ in range(cfg['pool_size'])])
    vectors=nn.ParameterList([nn.Parameter(torch.zeros(n,512,dtype=dtype,device=dev)) for n in cfg['dataset_subtype_num']])
    return key,prompt,vectors


def ensemble_through(task_name):
    root,commit=original_root();x=json.loads(pinned_text(root,commit,'class_ensemble/class_ensemble.json'))['0']
    if task_name not in x['classnames']:raise ValueError('An explicit class-name/template manifest is needed for another clinical endpoint')
    result={'ensemble_classes':[],'count':[]}
    for task,classes in x['classnames'].items():
        for names in classes.values():
            result['count'].append(len(names)*len(x['templates']))
            for name in names:result['ensemble_classes'] += [t.replace('CLASSNAME',name) for t in x['templates']]
        if task==task_name:break
    return result


def model(cfg,base,pools,ensemble,train_frequency):
    definitions().utilities.set_tunable_v(pools[2],cfg['task_num'])
    return definitions().model.QPMIL_VL(cfg,base,base.logit_scale.device,*pools,ensemble,train_frequency)


class Log:
    def __init__(self):self.rows=[]
    def log(self,row):self.rows.append(dict(row))


def original_update(model,optimizer,cfg,bags,labels):
    # Call the literal original CE + lambda*matching + beta*class-similarity
    # update, including norm clipping and per-parameter zero-gradient clearing.
    log=Log()
    manager=load_definitions('qpmilvl','manager/manager.py',['Manager'],
        {'torch':torch,'wandb':log,'Tensor':torch.Tensor})
    context=SimpleNamespace(model=model,optimizer=optimizer,cfg=cfg,loss_function=nn.CrossEntropyLoss())
    result=manager.Manager._update_network(context,len(bags),bags,labels,None)
    return result,log.rows[-1]
