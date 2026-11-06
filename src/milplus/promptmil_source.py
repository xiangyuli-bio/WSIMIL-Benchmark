"""Private original Prompt-MIL DINO, prompt and DSMIL definitions, without substitutes."""
import argparse
from functools import partial
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .io import digest_file
from .upstream import verified_source, pinned_text, package, load_definitions

COMMIT='b7c088e983eb4e369b9fe346873e941da01ed2e6'
WEIGHTS={
    'brca':{'filename':'tcga-brca-vitt.pth','sha256':'a80864a58cae61c690ff9709b6e34c0148fa08e11112cef5de5c823d1fe88596'},
    'bright':{'filename':'bright-vitt.pth','sha256':'6f91a4182fdf3e4560d3c139aaf5b597e36bc7dc1cad9b9b3374e742a8a3dc4f'},
    'crc':{'filename':'tcga-crc-vitt-cings.pth','sha256':'8382d3007ca6884d9c088295c4842c958ca40eabcf9ab76b18098a557ab18802'},
}


def checked_source():
    root,commit=verified_source('promptmil')
    if commit!=COMMIT:raise ValueError('Prompt-MIL needs the reviewed complete original source')
    return root,commit


def author_modules():
    root,commit=checked_source();prefix='_milplus_promptmil_'+commit
    if prefix+'.dino_vit_prompt' not in sys.modules:
        package(prefix,root/'network')
        for filename in ['dino_vision_transformer','dino_vit_prompt','dsmil']:
            path='network/'+filename+'.py'
            text=pinned_text(root,commit,path)
            if (root/path).read_text()!=text:raise ValueError('Original Prompt-MIL tracked source bytes changed')
            mod=types.ModuleType(prefix+'.'+filename);mod.__file__=str(root/path);mod.__package__=prefix
            sys.modules[mod.__name__]=mod
            exec(compile(text,mod.__file__,'exec'),mod.__dict__)  # noqa:S102 -- reviewed pinned Git blob
    forward=load_definitions('promptmil','pl_model/forward_fn.py',
        ['dsmil_forward','dsmil_forward_ce'],{'torch':torch,'F':F})
    return sys.modules[prefix+'.dino_vit_prompt'],sys.modules[prefix+'.dsmil'],forward


def _legacy_scalar(dtype,data):
    if (not isinstance(dtype,np.dtype) or dtype.hasobject or not isinstance(data,bytes)
            or len(data)!=dtype.itemsize):raise ValueError('Invalid original numeric scalar encoding')
    return np.core.multiarray.scalar(dtype,data)


# NumPy2 moved this primitive to _core. Match the qualified name in the
# original author checkpoint without enabling arbitrary pickle globals.
_legacy_scalar.__module__='numpy.core.multiarray'
_legacy_scalar.__name__='scalar'


def read_teacher(path,condition):
    if condition not in WEIGHTS or digest_file(path)!=WEIGHTS[condition]['sha256']:
        raise ValueError('Prompt-MIL requires its actual original cohort DINO checkpoint, not a random or partial encoder')
    allowed=[argparse.Namespace,_legacy_scalar,np.dtype]+[type(np.dtype(x)) for x in ['float64','float32','int64','int32']]
    with torch.serialization.safe_globals(allowed):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(checkpoint,dict) or not isinstance(checkpoint.get('teacher'),dict):
        raise ValueError('Original Prompt-MIL teacher branch is missing')
    return {k.replace('module.','').replace('backbone.',''):v for k,v in checkpoint['teacher'].items()}


def build_original(weights,condition,classes,deep_prompt=False):
    if type(classes)is not int or classes<2 or type(deep_prompt)is not bool:
        raise ValueError('Declare actual classes and an explicit Prompt-MIL prompt-depth option')
    prompt,dsmil,forward=author_modules()
    config=dict(patch_size=16,embed_dim=192,depth=12,num_heads=3,use_avgpool=True,
                qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6))
    backbone=prompt.DinoPromptedTransformer(config,1,0.,project_prompt_dim=-1,deep_prompt=deep_prompt)
    values=read_teacher(weights,condition)
    expected={k:v for k,v in backbone.state_dict().items() if 'prompt' not in k}
    if not set(expected)<=set(values):raise ValueError('Original pretrained backbone keys are incomplete')
    for k,value in expected.items():
        if values[k].shape!=value.shape or values[k].dtype!=value.dtype:
            raise ValueError('Original Prompt-MIL backbone shape or dtype mismatch: '+k)
    result=backbone.load_state_dict({k:values[k] for k in expected},strict=False)
    allowed_missing={'prompt_embeddings'}|({'deep_prompt_embeddings'} if deep_prompt else set())
    if set(result.missing_keys)!=allowed_missing or result.unexpected_keys:
        raise ValueError('Only newly introduced prompt parameters may lack pretrained weights')
    for k,value in backbone.state_dict().items():
        if 'prompt'not in k and not torch.equal(value,values[k]):raise ValueError('Pretrained tensor did not load: '+k)
    for name,parameter in backbone.named_parameters():parameter.requires_grad=('prompt'in name)
    classifier=dsmil.MILNet(dsmil.FCLayer(192,classes),dsmil.BClassifier(192,classes,dropout_v=0.))
    return backbone,classifier,forward
