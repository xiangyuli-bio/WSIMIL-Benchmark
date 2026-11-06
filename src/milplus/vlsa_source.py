"""Private complete VLSA rank/VLFAN/CONCH mechanisms and actual native weights.

No smaller text model or generic hazard-NLL replacement is used. Method
admission still requires the full trainer, inputs and independent package gates.
"""
import contextlib
import io
import json
import math
import os
import random
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .focus_network import WEIGHTS_BYTES,WEIGHTS_SHA
from .io import digest_file
from .upstream import load_definitions,load_single,pinned_text,verified_source

COMMIT='915f37afcbd86cd6c33ebb30ab0caa5def0f9ab9'
TOKENIZER='model/conch/tokenizers/conch_byte_level_bpe_uncased.json'


def weights_path(value=None):
    return Path(value or os.environ.get('MILPLUS_VLSA_WEIGHTS',
        os.environ.get('MILPLUS_FOCUS_WEIGHTS',Path.home()/'.cache/milplus/assets/focus/conch.pth'))).expanduser().resolve()


def fixed():
    root,sha=verified_source('vlsa')
    if sha!=COMMIT:raise ValueError('VLSA requires the reviewed original author commit')
    return root,sha


def definitions(path,names,namespace=None,replacements=()):
    fixed()
    return load_definitions('vlsa',path,names,{'torch':torch,'nn':nn,'F':F,'np':np,
        'Optional':Optional,'Tuple':Tuple,'Union':Union,'List':List,'json':json,
        **(namespace or {})},replacements)


def actual_conch(weights):
    weights=Path(weights)
    if weights.stat().st_size!=WEIGHTS_BYTES or digest_file(weights)!=WEIGHTS_SHA:
        raise ValueError('VLSA requires the actual complete native pretrained CONCH checkpoint')
    root,sha=fixed()
    cfg=json.loads(pinned_text(root,sha,'model/conch/model_configs/conch_ViT-B-16.json'))
    expected={'context_length':128,'vocab_size':32007,'width':768,'heads':12,
              'layers':12,'embed_cls':True,'output_tokens':True}
    if cfg['text_cfg']!=expected or cfg['embed_dim']!=512:
        raise ValueError('VLSA native CONCH text architecture changed')
    block=definitions('model/conch/transformer.py',
        ['LayerNormFp32','LayerNorm','QuickGELU','LayerScale','ResidualAttentionBlock','Transformer','TextTransformer'],
        {'math':math,'random':random,'OrderedDict':OrderedDict,'Callable':Callable,
         'Sequence':Sequence,'checkpoint':checkpoint})
    builder=definitions('model/conch/coca_model.py',['CLIPTextCfg','_build_text_tower'],
                        {'dataclass':dataclass,'TextTransformer':block.TextTransformer})
    # Only unused frozen foundation initialization is RNG-isolated. All learned
    # VLFAN/rank/context/residual initialization follows the actual source.
    with torch.random.fork_rng(devices=[]):
        text=builder._build_text_tower(512,cfg['text_cfg'])
    saved=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
    state={k.removeprefix('text.'):v for k,v in saved.items() if k.startswith('text.')}
    if len(state)!=150 or 'logit_scale' not in saved or saved['logit_scale'].shape!=():
        raise ValueError('VLSA checkpoint native text/temperature tensors are incomplete')
    text.load_state_dict(state,strict=True);text.requires_grad_(False)
    model=nn.Module();model.text=text
    model.logit_scale=nn.Parameter(saved['logit_scale'].detach().clone())
    return model


@lru_cache(maxsize=1)
def tokenizer_module():
    root,sha=fixed()
    if (root/TOKENIZER).read_text()!=pinned_text(root,sha,TOKENIZER):
        raise ValueError('VLSA local tokenizer JSON differs from its fixed Git object')
    # transformers4.51 resets the ByteLevel prefix default toFalse, overriding
    # this original JSON's True. The author's4.41.2/.19.1 stack retained True.
    # Bind that explicit original setting instead of changing any text, BPE
    # weights or model mechanism. Independent old-stack token IDs verify it.
    config=json.loads((root/TOKENIZER).read_text())
    if config.get('pre_tokenizer',{}).get('add_prefix_space') is not True:
        raise ValueError('VLSA original ByteLevel prefix-space contract changed')
    from transformers import PreTrainedTokenizerFast

    def original_factory(*args,**kwargs):
        return PreTrainedTokenizerFast(*args,**kwargs,add_prefix_space=True)

    return definitions('model/conch/custom_tokenizer.py',['get_tokenizer','tokenize'],
        {'PreTrainedTokenizerFast':original_factory,'TOKENIZER_DIR':root/Path(TOKENIZER).parent,
         'DEFAULT_TOKENIZER':Path(TOKENIZER).name})


def source_components(weights):
    conch_tokenizer=tokenizer_module()
    tokenizer_cls=definitions('model/utils_vl.py',['Tokenizer'],{'conch':conch_tokenizer}).Tokenizer
    encoder_cls=definitions('model/prompt_encoder.py',['CONCHPromptEncoder'],
        {'conch':SimpleNamespace(coca_model=SimpleNamespace(CoCa=nn.Module))}).CONCHPromptEncoder
    io_functions=definitions('utils/io.py',['load_init_prompt','load_init_text'])
    plain=definitions('model/prompt_learners/plain_prompt_learner.py',['PlainPromptLearner'],
                      {'load_init_prompt':io_functions.load_init_prompt}).PlainPromptLearner
    rank=definitions('model/prompt_learners/rank_prompt_learner.py',['RankPromptLearner'],
        {'PlainPromptLearner':plain,'load_init_prompt':io_functions.load_init_prompt}).RankPromptLearner
    adapter=definitions('model/prompt_learners/prompt_adapter.py',['PromptAdapter'],
        {'load_init_prompt':io_functions.load_init_prompt,'load_init_text':io_functions.load_init_text}).PromptAdapter
    prompt_api=definitions('model/prompt_learners/__init__.py',['load_prompt_learner','load_prompt_adapter'],
        {'PlainPromptLearner':plain,'RankPromptLearner':rank,'PromptAdapter':adapter})
    mil=definitions('model/deepmil.py',['VLFAN']).VLFAN

    def native_factory(text_cfg,image_cfg,root,api,**kwargs):
        if (api!='CONCH' or text_cfg.get('name')!='mahmoodlab/conch'
                or image_cfg.get('name')!='VLFAN' or image_cfg.get('dim_in')!=512
                or image_cfg.get('query')!='Text' or image_cfg.get('query_pooling')!='mean'
                or image_cfg.get('gated_query') or image_cfg.get('use_feat_proj')):
            raise ValueError('VLSA source reference requires its complete native CONCH/VLFAN/Text-query configuration')
        model=actual_conch(weights)
        model.visual=mil(**image_cfg)
        return model

    network=definitions('model/vlsa.py',['VLSA'],{'Tokenizer':tokenizer_cls,
        'load_vl_model_to_cpu':native_factory,
        'get_prompt_encoder':lambda model,api:encoder_cls(model),
        'load_prompt_learner':prompt_api.load_prompt_learner,
        'load_prompt_adapter':prompt_api.load_prompt_adapter}).VLSA
    return network


def build_network(weights,rank_prompts,query_prompts,query_key,num_ranks):
    if type(num_ranks) is not int or num_ranks<2:
        raise ValueError('VLSA ordinal interpolation requires at least two actual time ranks')
    query_data=json.loads(Path(query_prompts).read_text())
    texts=query_data.get(query_key)
    if not isinstance(texts,list) or not texts or any(not isinstance(t,str) or not t.strip() for t in texts):
        raise ValueError('VLSA requires explicit ordered prognostic query texts')
    text_cfg={'name':'mahmoodlab/conch','frozen':True}
    image_cfg={'name':'VLFAN','frozen':False,'dim_in':512,'dim_hid':256,'use_feat_proj':False,
        'drop_rate':.25,'pred_head':'default','query':'Text','num_query':len(texts),
        'query_pooling':'mean','gated_query':False,'query_text_method':'TaskRes',
        'query_text_res_ratio':.5,'query_text_load_path':str(query_prompts),'query_text_load_idx':query_key}
    prompt_cfg={'name':'CoOp','pretrained':False,'method':'rank','num_ranks':num_ranks,
        'num_base_ranks':4,'num_tokens_per_rank':4,'num_context_tokens':8,
        'rank_tokens_position':'tail','init_prompt_path':str(rank_prompts),
        'init_prompt_rank_idx':0,'init_prompt_context_idx':0,'rank_specific_context':False,
        'frozen_context_embeds':False,'frozen_rank_embeds':False}
    with contextlib.redirect_stdout(io.StringIO()):
        model=source_components(weights)(text_cfg,image_cfg,prompt_cfg,
            vlsa_api='CONCH',path_clip_model='bound_actual_native_CONCH_checkpoint')
    model.prompt_encoder.requires_grad_(False)
    # Persist the original constructor-generated constants. The source marked
    # several nonpersistent, but checkpoint verification must bind their values.
    for module in model.modules():
        module._non_persistent_buffers_set.clear()
    encoder=model.prompt_encoder
    mask=encoder.attn_mask;delattr(encoder,'attn_mask');encoder.register_buffer('attn_mask',mask)
    scale=model.mil_encoder.coattn_logit_scale;delattr(model.mil_encoder,'coattn_logit_scale')
    model.mil_encoder.register_buffer('coattn_logit_scale',scale)
    return model


@lru_cache(maxsize=1)
def objectives():
    mle=definitions('loss/loss_surv.py',['SurvIFMLE']).SurvIFMLE
    emd=definitions('loss/loss_surv_ext.py',['cdf_loss','convert_survival_label','SurvEMD']).SurvEMD
    with contextlib.redirect_stdout(io.StringIO()):return mle(alpha=0.),emd(p=2)


def objective(logits,times,events,logit_scale):
    if (logits.ndim!=2 or len(logits)<1 or logits.shape[1]<2 or not torch.isfinite(logits).all()
            or times.shape!=(len(logits),) or events.shape!=(len(logits),)
            or times.dtype!=torch.int64 or (times<0).any() or (times>=logits.shape[1]).any()
            or not torch.isin(events,events.new_tensor([0,1])).all()):
        raise ValueError('VLSA requires finite incidence logits and aligned discrete event/time labels')
    probabilities=logits.softmax(-1)
    mle,emd=objectives();a=mle(probabilities,times,events);b=emd(probabilities,times,events,logit_scale)
    return a+b,{'ifmle':a,'emd':b,'incidence':probabilities}


@lru_cache(maxsize=1)
def evaluation_source():
    from scipy import integrate
    from scipy.interpolate import interp1d,PchipInterpolator
    curves=definitions('eval/SurvivalEVAL/Evaluations/util.py',
        ['interpolated_survival_curve','predict_mean_survival_time'],
        {'warnings':warnings,'integrate':integrate,'interp1d':interp1d,'PchipInterpolator':PchipInterpolator})
    pairs=definitions('eval/SurvivalEVAL/Evaluations/Concordance.py',
        ['concordance','_estimate_concordance_index','_get_comparable'])
    return curves,pairs


def mean_times(incidence,time_coordinates):
    tensor=torch.as_tensor(incidence).detach().cpu()
    if tensor.dtype not in {torch.float32,torch.float64}:
        raise ValueError('VLSA source curve conversion needs its original floating-point incidence dtype')
    values=tensor.numpy();coords=np.asarray(time_coordinates,dtype=np.float64)
    if (values.ndim!=2 or values.shape[1]!=len(coords) or len(coords)<2
            or not np.isfinite(values).all() or (values<0).any()
            or not np.allclose(values.sum(-1),1,atol=1e-6)
            or coords[0]!=0 or not np.isfinite(coords).all() or not (np.diff(coords)>0).all()):
        raise ValueError('VLSA mean-time prediction requires finite ordered TRAIN-derived coordinates and incidence probabilities')
    # Source first forms the curve with torch.cumsum in prediction dtype; only
    # SurvivalEVAL's check_and_convert then casts the curve to float64.
    survival=(1-tensor.cumsum(-1)).clamp_min(0).numpy().astype(np.float64)
    curves,_=evaluation_source()
    means=np.asarray([curves.predict_mean_survival_time(s,coords,interpolation='Linear') for s in survival])
    if not np.isfinite(means).all():raise ValueError('VLSA original survival-curve mean is undefined')
    return means,survival


def source_concordance(time,event,predicted_time):
    t,e,p=(np.asarray(v) for v in [time,event,predicted_time])
    if (t.ndim!=1 or len(t)<2 or e.shape!=t.shape or p.shape!=t.shape
            or not np.isfinite(t).all() or not np.isfinite(p).all() or not np.isin(e,[0,1]).all()):
        raise ValueError('VLSA concordance needs aligned finite patient arrays')
    _,pairs=evaluation_source()
    value,_,denominator=pairs.concordance(p,t,e,pair_method='Comparable',ties='All')
    if denominator<=0 or not np.isfinite(value):raise ValueError('VLSA source concordance has no comparable patient pairs')
    return float(value)


def time_bins(train_times,train_events,num_bins=None):
    """Original interval interior edges; terminal infinity removes held-out max.

    The source uses full-table max+EPS solely to close the last interval. All
    finite coordinates here retain its TRAIN-event-based linspace and auto
    ceil(sqrt(event_count)) rank count, without consulting held-out follow-up.
    """
    import pandas as pd
    times,events=np.asarray(train_times,dtype=np.float64),np.asarray(train_events)
    if (times.ndim!=1 or not len(times) or events.shape!=times.shape
            or not np.isfinite(times).all() or (times<=0).any()
            or not np.isin(events,[0,1]).all() or events.sum()<2
            or (num_bins is not None and (type(num_bins) is not int or num_bins<2))):
        raise ValueError('VLSA time bins require positive finite TRAIN times and at least two observed TRAIN events')
    converter=definitions('dataset/label_converter.py',['calculate_discrete_time_bins'],{'math':math})
    with contextlib.redirect_stdout(io.StringIO()):
        bins=converter.calculate_discrete_time_bins(pd.DataFrame({'t':times,'e':events}),
            num_bins=num_bins,use_quantiles=False,max_time=float('inf'))
    if len(bins)<3 or not (np.diff(bins)>0).all():
        raise ValueError('VLSA original time-rank intervals are invalid')
    return bins


def optimizer_groups(model,weight_decay):
    grouping=definitions('optim/optim_factory.py',['add_weight_decay']).add_weight_decay
    return grouping(model,weight_decay)
