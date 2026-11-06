"""Pinned complete KMP-MIL and explicitly reviewed official CONCH dependency."""
import ast
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Callable,List,Optional,Sequence,Tuple,Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import yaml

from .upstream import verified_source,pinned_text,load_definitions
from .io import digest_file
from .akdpmp_source import _compile

COMMIT='77137ac84a9a27b16a2c07aca414c534f7cf5226'
CONCH_COMMIT='141cc09c7d4ff33d8eda562bd75169b457f71a62'


def root(source='kmpmil'):
    path,commit=verified_source(source)
    if commit!=(COMMIT if source=='kmpmil' else CONCH_COMMIT):raise ValueError('KMP-MIL needs the exact pinned source/dependency revision')
    return path,commit


def text(relative,source='kmpmil'):
    path,commit=root(source);return pinned_text(path,commit,relative).lstrip('\ufeff')


def definitions_from(source,relative,names,namespace):
    path,commit=root(source);original=pinned_text(path,commit,relative)
    replacements=[('\ufeff','')] if original.startswith('\ufeff') else []
    return load_definitions(source,relative,names,namespace,replacements=replacements)


def runtime_identity():
    import timm
    if timm.__version__!='1.0.20':raise ValueError('KMP-MIL requires its declared isolated timm1.0.20 runtime')
    return {'timm':timm.__version__,'torch_actual':torch.__version__,'precision':'original_float32',
        'CONCH_dependency_commit':CONCH_COMMIT,'dependency_scope':'Reviewed original official CONCH; KMP authors do not supply a historical dependency pin'}


@lru_cache(maxsize=1)
def conch_definitions():
    runtime_identity();path,commit=root('conch_kmp')
    ns={'torch':torch,'nn':nn,'F':F,'np':np,'math':__import__('math'),'random':__import__('random'),
        'OrderedDict':OrderedDict,'Callable':Callable,'Optional':Optional,'Sequence':Sequence,
        'Tuple':Tuple,'Union':Union,'List':List,'checkpoint':checkpoint,'dataclass':dataclass,'logging':logging}
    transformer=definitions_from('conch_kmp','conch/open_clip_custom/transformer.py',
        ['LayerNormFp32','LayerNorm','QuickGELU','LayerScale','PatchDropout','Attention','AttentionalPooler',
         'ResidualAttentionBlock','CustomResidualAttentionBlock','Transformer','TextTransformer','MultimodalTransformer'],ns)
    from timm.models.vision_transformer import VisionTransformer
    from timm.layers import Mlp,to_2tuple
    vision=definitions_from('conch_kmp','conch/open_clip_custom/vision_tower.py',['VisualModel'],
        {**ns,'AttentionalPooler':transformer.AttentionalPooler,'Mlp':Mlp,'to_2tuple':to_2tuple})
    coca=definitions_from('conch_kmp','conch/open_clip_custom/coca_model.py',
        ['CoCaVisionCfg','CLIPTextCfg','_build_vision_tower','_build_text_tower','_build_text_decoder_tower','CoCa','resize_pos_embed'],
        {**ns,'VisualModel':vision.VisualModel,'VisionTransformer':VisionTransformer,
         'TextTransformer':transformer.TextTransformer,'MultimodalTransformer':transformer.MultimodalTransformer})
    from transformers import PreTrainedTokenizerFast
    tokenizer=definitions_from('conch_kmp','conch/open_clip_custom/custom_tokenizer.py',['get_tokenizer','tokenize'],
        {'PreTrainedTokenizerFast':PreTrainedTokenizerFast,'F':F,'TOKENIZER_DIR':path/'conch/open_clip_custom/tokenizers',
         'DEFAULT_TOKENIZER':'conch_byte_level_bpe_uncased.json'})
    actual=path/'conch/open_clip_custom/tokenizers/conch_byte_level_bpe_uncased.json'
    if actual.read_text()!=pinned_text(path,commit,str(actual.relative_to(path))):raise ValueError('Original CONCH tokenizer bytes changed')
    return SimpleNamespace(coca=coca,tokenizer=tokenizer,transformer=transformer)


def load_base(weights,expected_sha256,device='cpu'):
    runtime_identity();p=Path(weights).expanduser().resolve()
    if len(expected_sha256)!=64 or digest_file(p)!=expected_sha256:raise ValueError('Actual authorized CONCH checkpoint hash mismatch')
    dependency=conch_definitions();cfg=json.loads(text('conch/open_clip_custom/model_configs/conch_ViT-B-16.json','conch_kmp'));cfg.pop('custom_text',None)
    base=dependency.coca.CoCa(**cfg)
    checkpoint_values=torch.load(p,map_location='cpu',weights_only=True);state=checkpoint_values.get('state_dict',checkpoint_values)
    if any(k.startswith('module.') for k in state):state={k.removeprefix('module.'):v for k,v in state.items()}
    base.text.load_state_dict({k[len('text.'):]:v for k,v in state.items() if k.startswith('text.')},strict=True)
    if 'logit_scale' not in state or state['logit_scale'].shape!=torch.Size([]):raise ValueError('Original pretrained CONCH logit_scale is required')
    dependency.coca.resize_pos_embed(state,base);missing,unexpected=base.load_state_dict(state,strict=False)
    if unexpected or any(not k.startswith('text_decoder.') for k in missing):raise ValueError('Actual used CONCH text/visual weights are incomplete')
    base.original_missing_unused_decoder_keys=list(missing);base.to(device).eval();base.requires_grad_(False);base.dtype=base.logit_scale.dtype
    if base.dtype!=torch.float32:raise ValueError('Original full-float KMP CONCH reference requires actual float32 weights')
    return base


@lru_cache(maxsize=1)
def definitions():
    path,commit=root();dependency=conch_definitions()
    ns={'torch':torch,'nn':nn,'F':F,'np':np,'List':List,'Optional':Optional,'Dict':dict,'Tensor':torch.Tensor,
        'softmax':F.softmax,'time':__import__('time')}
    from sklearn.metrics import accuracy_score,average_precision_score,roc_auc_score
    ns.update(accuracy_score=accuracy_score,average_precision_score=average_precision_score,roc_auc_score=roc_auc_score)
    pfc=definitions_from('kmpmil','models/pfc.py',['PrototypeConditionedFeatureCalibration'],ns)
    tc=definitions_from('kmpmil','models/tc.py',['TextualCalibration'],ns)
    model=definitions_from('kmpmil','models/model_il.py',['merge_parameter','CONCHPromptEncoder','PromptLearner','KMPMIL'],
        {**ns,'get_tokenizer':dependency.tokenizer.get_tokenizer,'tokenize':dependency.tokenizer.tokenize,
         'PrototypeConditionedFeatureCalibration':pfc.PrototypeConditionedFeatureCalibration,'TextualCalibration':tc.TextualCalibration})
    builders=definitions_from('kmpmil','models/kmp.py',
        ['build_task_kmp_pools','_rows_to_parameter_list','build_universal_kmp_mil','build_task_kmp_mil'],{**ns,'KMPMIL':model.KMPMIL})
    utilities=definitions_from('kmpmil','models/model_utils.py',['freeze_weight','activate_current_tc_residual','EarlyStopping'],ns)
    training=definitions_from('kmpmil','manager/training.py',['TrainingMixin'],ns)
    metrics=definitions_from('kmpmil','manager/metrics_helpers.py',['MetricsMixin'],ns)
    evaluation=definitions_from('kmpmil','manager/evaluation.py',['EvaluationMixin'],ns)
    tree=ast.parse(text('manager/training.py'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TrainingMixin')
    update=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_update_network')
    body=__import__('copy').deepcopy(update.body)
    assert isinstance(body[-1],ast.Return) and isinstance(body[-1].value,ast.Tuple) and [n.id for n in body[-1].value.elts]==['bag_preds','key_indices']
    # The only AST view change is additional observational return values. Every
    # original loss, clipping/zero-gradient rule and Adam step stays literal.
    body[-1]=ast.parse("return bag_preds,key_indices,{'clf_loss':float(clf_loss.detach()),'matching_loss':float(loss_dict['matching_loss'].detach()),'routing_loss':float(loss_dict['routing_loss'].detach()),'class_sim_loss':float(loss_dict['class_sim_loss'].detach()),'pfc_reg_loss':float(pfc_reg_loss.detach()),'total_loss':float(total_loss.detach())}").body[0]
    observed_update=_compile('author_update_with_observational_return',['self','i_batch','xs','ys'],body,ns,path/'manager/training.py')
    return SimpleNamespace(model=model,builders=builders,utilities=utilities,training=training,metrics=metrics,evaluation=evaluation,observed_update=observed_update)


def config():return yaml.safe_load(text('main.yaml'))


def source_text_assets():
    return json.loads(text('textual_knowledge/class_prompts.json')),json.loads(text('textual_knowledge/task_descriptors.json'))


def class_prompts(task_name):
    assets,_=source_text_assets()
    if task_name not in assets['classnames']:raise ValueError('Declare actual clinical descriptions for another KMP task')
    result={'class_prompts':[],'count':[]}
    for task,classes in assets['classnames'].items():
        for names in classes.values():
            result['count'].append(len(names)*len(assets['templates']))
            for name in names:result['class_prompts'] += [t.replace('CLASSNAME',name) for t in assets['templates']]
        if task==task_name:break
    return result


def task_pools(cfg,base,descriptor_count):
    return definitions().builders.build_task_kmp_pools(cfg,feature_dim=512,embedding_dim=768,dtype=base.dtype,device=base.logit_scale.device,pool_len=descriptor_count)


def original_update(model,optimizer,cfg,bags,labels):
    context=SimpleNamespace(model=model,optimizer=optimizer,cfg=cfg,loss_function=nn.CrossEntropyLoss())
    return definitions().observed_update(context,len(bags),bags,labels)
