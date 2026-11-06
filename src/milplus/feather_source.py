"""Privately load actual pinned FEATHER definitions; never replace its weights.

Architecture-component inspection is separate from complete pretrained-model
construction. Original author files are read from their Git objects and are
not copied into this package.
"""
from abc import ABC, abstractmethod
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .upstream import load_definitions, pinned_text, verified_source

COMMIT = 'dfc3c17ca6876a8f9bd33a9162c40f2c22631d9d'
ENCODERS = {'conch_v15':768,'uni':1024,'uni_v2':1536}


def checked_source():
    root,commit = verified_source('feather')
    if commit != COMMIT:
        raise ValueError('FEATHER requires its reviewed original MIL-Lab revision')
    return root,commit


def original_configuration(encoder, num_classes):
    if encoder not in ENCODERS or type(num_classes) is not int or num_classes < 1:
        raise ValueError('Choose an original FEATHER24K native encoder and positive downstream class count')
    root,commit = checked_source()
    config = yaml.safe_load(pinned_text(root,commit,'src/model_configs/abmil/base.yaml'))
    if config != {'gate':True,'embed_dim':512,'attn_dim':384,'num_fc_layers':1,'dropout':.25}:
        raise ValueError('Original full FEATHER base architecture changed')
    tree = ast.parse(pinned_text(root,commit,'src/_global_mappings.py'))
    node = next(node for node in tree.body if isinstance(node,ast.AnnAssign)
                and isinstance(node.target,ast.Name) and node.target.id == 'ENCODER_DIM_MAPPING')
    source_encoders = ast.literal_eval(node.value)
    if any(source_encoders[name] != dim for name,dim in ENCODERS.items()):
        raise ValueError('Original FEATHER native encoder dimensions changed')
    return {**config,'in_dim':source_encoders[encoder],'num_classes':num_classes}


def architecture_classes():
    """Actual complete network definitions, not a pretrained execution profile."""
    checked_source()
    base = load_definitions('feather','src/models/mil_template.py',['MIL'],
        {'torch':torch,'nn':nn,'ABC':ABC,'abstractmethod':abstractmethod,'Optional':Optional})
    layers = load_definitions('feather','src/models/layers.py',
        ['create_mlp','GlobalAttention','GlobalGatedAttention'],{'nn':nn})
    module = load_definitions('feather','src/models/abmil.py',['ABMIL'],
        {'MIL':base.MIL,'torch':torch,'nn':nn,'F':F,
         'create_mlp':layers.create_mlp,'GlobalAttention':layers.GlobalAttention,
         'GlobalGatedAttention':layers.GlobalGatedAttention})
    return SimpleNamespace(MIL=base.MIL,ABMIL=module.ABMIL,layers=layers)


def original_model(encoder,num_classes):
    """Actual Hugging Face wrapper and complete original architecture.

    A caller must load the verified original pretrained asset before returning
    this model as FEATHER. This constructor alone never establishes admission.
    """
    import transformers
    from transformers import PretrainedConfig,PreTrainedModel
    if transformers.__version__ != '4.53.1':
        raise ValueError('Use the reviewed original FEATHER transformers4.53.1 stack')
    definitions = architecture_classes()
    module = load_definitions('feather','src/models/abmil.py',
        ['ABMILGatedBaseConfig','ABMILModel'],
        {'PretrainedConfig':PretrainedConfig,'PreTrainedModel':PreTrainedModel,
         'MODEL_TYPE':'abmil','ABMIL':definitions.ABMIL,'torch':torch,'nn':nn})
    return module.ABMILModel(module.ABMILGatedBaseConfig(**original_configuration(encoder,num_classes)))


def original_loading_helpers():
    """Actual local-checkpoint loading and classifier-reset functions.

    Remote branches are inaccessible from the candidate's verified local-file
    interface. No download or access-condition acceptance is performed here.
    """
    import safetensors.torch
    from loguru import logger
    checked_source()
    namespace = {'torch':torch,'nn':nn,'os':os,'Path':Path,'Any':Any,'Callable':Callable,
        'Dict':Dict,'Optional':Optional,'Tuple':Tuple,'Union':Union,
        'safetensors':safetensors,'_has_safetensors':True,'logger':logger}
    return load_definitions('feather','src/builder_utils.py',
        ['clean_state_dict','_resolve_pretrained_source','load_state_dict','load_pretrained'],namespace)


def build_pretrained(asset,num_classes):
    """Complete actual source construction followed by verified pretrained load.

    Every non-classifier key must be present and match the released tensor.
    Only the source's original downstream classifier is reinitialized.
    """
    from .feather_assets import OriginalFEATHERAsset
    if not isinstance(asset,OriginalFEATHERAsset):
        asset = OriginalFEATHERAsset(asset)
    asset.verify(deep=True)
    model = original_model(asset.encoder,num_classes)
    helpers = original_loading_helpers()
    checkpoint = helpers.load_state_dict(str(asset.weights),device='cpu')
    parameters = model.state_dict()
    expected = {key for key in parameters if not key.startswith('model.classifier.')}
    observed = {key for key in checkpoint if not key.startswith('model.classifier.')}
    if observed != expected:
        raise ValueError('Original FEATHER backbone checkpoint is incomplete or has unexpected keys')
    for key in expected:
        value = checkpoint[key];target = parameters[key]
        if not isinstance(value,torch.Tensor) or value.dtype != target.dtype or value.shape != target.shape:
            raise ValueError('Original FEATHER native backbone shape or dtype differs: '+key)
    model = helpers.load_pretrained(model,num_classes=num_classes,
        pretrained_cfg={'source':'file','file':str(asset.weights)},strict=False,
        from_pretrained=False,keep_classifier=False)
    current = model.state_dict()
    for key in expected:
        if not torch.equal(current[key].cpu(),checkpoint[key].cpu()):
            raise ValueError('An original FEATHER pretrained backbone tensor was replaced: '+key)
    if (model.model.classifier.in_features,model.model.classifier.out_features) != (512,num_classes):
        raise ValueError('Original FEATHER downstream classifier reset changed')
    if any(not value.requires_grad for value in model.parameters()):
        raise ValueError('Complete FEATHER finetuning must keep its actual pretrained backbone trainable')
    model._milplus_feather_evidence = {'asset_contract':asset.contract,
        'original_source_commit':COMMIT,'pretrained_backbone_keys_exact':sorted(expected),
        'all_pretrained_backbone_tensors_exact':True,'source_classifier_reset_preserved':True,
        'native_encoder':asset.encoder,'native_feature_dim':ENCODERS[asset.encoder],
        'parameters':sum(parameter.numel() for parameter in model.parameters()),
        'full_method_validated':False}
    return model
