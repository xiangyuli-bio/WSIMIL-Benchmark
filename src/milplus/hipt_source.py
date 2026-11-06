"""Pinned HIPT definitions and the actual published pretrained teacher backbone."""
import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import warnings
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .io import atomic_torch_save, digest_file
from .upstream import load_definitions, verified_source

COMMIT='780fafaed2e5b112bc1ed6e78852af1fe6714342'
CHECKPOINT_SHA='2b0bd9e9a602a35f2bb3f76da39d2b53a91f23fc3f115dc59a63267d95ad2b7b'
TENSOR_SHA='c4a5c7b076768d6180f940070df225e43edc3ab70c85f5c1d227041d1b72793b'
PATCH_SHA='6960cd5a8657dc8bb214671aa0c6dbd3f5b698e84386884955836487ddc89e24'
PROFILES={'hipt_lgp_frozen_source':True,'hipt_lgp_finetune_source':False}


def author_definitions():
    _,sha=verified_source('hipt')
    if sha!=COMMIT:raise ValueError('HIPT source identity differs from the reviewed native model')
    namespace={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'partial':partial,'warnings':warnings}
    vit=load_definitions('hipt','HIPT_4K/vision_transformer4k.py',
        ['_no_grad_trunc_normal_','trunc_normal_','drop_path','DropPath','Mlp','Attention','Block','VisionTransformer4K','vit4k_xs'],namespace)
    attention=load_definitions('hipt','2-Weakly-Supervised-Subtyping/models/model_utils.py',
        ['Attn_Net_Gated'],namespace)
    model=load_definitions('hipt','2-Weakly-Supervised-Subtyping/models/model_hierarchical_mil.py',
        ['HIPT_LGP_FC'],{**namespace,'vit4k_xs':vit.vit4k_xs,'Attn_Net_Gated':attention.Attn_Net_Gated})
    return model.HIPT_LGP_FC,vit.vit4k_xs


def tensor_map_hash(state):
    if not isinstance(state,dict) or not state or not all(isinstance(k,str) and torch.is_tensor(v) for k,v in state.items()):
        raise ValueError('HIPT pretrained teacher must be a named tensor map')
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        if value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise ValueError('HIPT native pretrained teacher requires finite float32 tensors')
        h.update(name.encode()+b'\0')
        h.update(json.dumps(list(value.shape),separators=(',',':')).encode()+b'\0')
        h.update(value.detach().cpu().contiguous().numpy().astype('<f4',copy=False).tobytes())
    return h.hexdigest()


def _numeric_scalar(dtype,content):
    if dtype.kind not in 'biufc' or dtype.itemsize>16 or not isinstance(content,bytes) or len(content)!=dtype.itemsize:
        raise ValueError('Only fixed-width numeric legacy NumPy scalars are accepted')
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # NumPy1 keeps the original module path.
        from numpy.core.multiarray import scalar
    return scalar(dtype,content)


# Torch2.4/2.5 safe_globals maps by module/name; no NumPy module mutation.
_numeric_scalar.__name__='scalar'
_numeric_scalar.__module__='numpy.core.multiarray'


def _legacy_backbone(path):
    """Decode only the fixed original file; invoked in a separate interpreter."""
    path=Path(path)
    if path.stat().st_size!=395710078 or digest_file(path)!=CHECKPOINT_SHA:
        raise ValueError('HIPT original checkpoint identity differs')
    with torch.serialization.safe_globals([_numeric_scalar,np.dtype,type(np.dtype('float64')),argparse.Namespace]):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    teacher=checkpoint.get('teacher')
    if not isinstance(teacher,dict):raise TypeError('HIPT original teacher is absent')
    converted={k.removeprefix('module.').removeprefix('backbone.'):v for k,v in teacher.items()}
    weights={k:v for k,v in converted.items() if not k.startswith('head.')}
    if tensor_map_hash(weights)!=TENSOR_SHA:raise ValueError('HIPT original teacher tensors differ')
    return weights


def _isolated_legacy_backbone(path):
    # Torch2.5 safe_globals removes overlapping pre-existing entries on exit.
    # Never enter that global context in the caller (including concurrent loads).
    # Child imports the exact same package path, also for extracted-wheel runs.
    program="""import sys,torch
sys.path.insert(0,sys.argv[1])
torch.set_num_threads(1)
from milplus.hipt_source import _legacy_backbone
weights=_legacy_backbone(sys.argv[2])
torch.save({k:v.detach().cpu().clone() for k,v in weights.items()},sys.stdout.buffer)
"""
    result=subprocess.run([sys.executable,'-c',program,str(Path(__file__).resolve().parent.parent),
        str(Path(path).resolve())],capture_output=True,check=False)
    if result.returncode:
        raise ValueError('HIPT isolated weights-only conversion failed; no unsafe loading fallback')
    return torch.load(io.BytesIO(result.stdout),map_location='cpu',weights_only=True)


def load_backbone(path):
    """Accept the fixed public original or its bitwise verified portable tensor map.

    weights_only is always enabled. Original NumPy1 numeric metadata is decoded
    only after the entire original file matches its published Git LFS digest,
    in an isolated interpreter that cannot alter the caller's safe globals.
    """
    path=Path(path)
    if path.stat().st_size==395710078 and digest_file(path)==CHECKPOINT_SHA:
        weights=_isolated_legacy_backbone(path)
    else:
        checkpoint=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if (not isinstance(checkpoint,dict) or set(checkpoint)!={'schema_version','kind','source_commit',
                'parent_checkpoint_sha256','tensor_map_sha256','backbone'} or
                checkpoint['schema_version']!=1 or checkpoint['kind']!='hipt_teacher_backbone' or
                checkpoint['source_commit']!=COMMIT or checkpoint['parent_checkpoint_sha256']!=CHECKPOINT_SHA or
                checkpoint['tensor_map_sha256']!=TENSOR_SHA):
            raise ValueError('HIPT requires its actual verified teacher backbone; no random or equal-width substitution')
        weights=checkpoint['backbone']
    if tensor_map_hash(weights)!=TENSOR_SHA:
        raise ValueError('HIPT pretrained tensors differ from the fixed public teacher')
    return weights


def export_backbone(checkpoint,output):
    output=Path(output)
    if output.exists():raise ValueError('HIPT pretrained backbone output must be new')
    weights=load_backbone(checkpoint)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_torch_save({'schema_version':1,'kind':'hipt_teacher_backbone','source_commit':COMMIT,
        'parent_checkpoint_sha256':CHECKPOINT_SHA,'tensor_map_sha256':TENSOR_SHA,
        'backbone':{k:v.detach().cpu().contiguous().clone() for k,v in weights.items()}},output)
    return output


def original_optimizer(model,lr,weight_decay):
    function=load_definitions('hipt','2-Weakly-Supervised-Subtyping/utils/utils.py',
        ['get_optim'],{'optim':torch.optim}).get_optim
    return function(model,SimpleNamespace(opt='adam',lr=lr,reg=weight_decay))
