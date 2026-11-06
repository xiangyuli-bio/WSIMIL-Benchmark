"""Original three-stage ProtoMIL sources and its reviewed actual SAE dependency.

This is component work, not an admitted or simplified ProtoMIL profile. The
unused language-model/sweep aggregate in SAE __init__ is isolated; every actual
SAE, resampler, optimizer, loss, metric and pipeline implementation stays original.
"""
import importlib
import importlib.machinery
from functools import lru_cache
from pathlib import Path
import types
import sys
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .upstream import verified_source,pinned_text,load_definitions

COMMIT='3e3745d7e4fecccb24aa486bd2af8b0dc6493711'
DTN_COMMIT='03484baef77747596712cf971e402f77b9da5a9a'


def root(source='protomil'):
    path,sha=verified_source(source)
    if sha!=(COMMIT if source=='protomil' else DTN_COMMIT):raise ValueError('Original ProtoMIL and reviewed SAE dependency commits are required')
    return path.resolve(),sha


@lru_cache(maxsize=1)
def sparse_definitions():
    path,sha=root('protomil_dtn');package=path/'sparse_autoencoder/sparse_autoencoder'
    for p in package.rglob('*.py'):
        if p.read_text()!=pinned_text(path,sha,str(p.relative_to(path))):raise ValueError('Actual original SAE dependency source bytes changed')
    # Isolate ONLY unused root imports of NLP datasets, transformer-lens and
    # hyperparameter sweep automation. No numerical submodule is substituted.
    name='sparse_autoencoder'
    if name in sys.modules:
        prior=sys.modules[name]
        if list(getattr(prior,'__path__',[]))!=[str(package)]:raise ValueError('Another unverified sparse_autoencoder is already imported')
    else:
        module=types.ModuleType(name);module.__path__=[str(package)];module.__package__=name
        module.__spec__=importlib.machinery.ModuleSpec(name,loader=None,is_package=True);sys.modules[name]=module
    exports={
        'SparseAutoencoder':('autoencoder.model','SparseAutoencoder'),
        'ActivationResampler':('activation_resampler.activation_resampler','ActivationResampler'),
        'TensorActivationStore':('activation_store.tensor_store','TensorActivationStore'),
        'AdamWithReset':('optimizer.adam_with_reset','AdamWithReset'),
        'L2ReconstructionLoss':('loss.decoded_activations_l2','L2ReconstructionLoss'),
        'LearnedActivationsL1Loss':('loss.learned_activations_l1','LearnedActivationsL1Loss'),
        'LossReducer':('loss.reducer','LossReducer'),
        'LossReductionType':('loss.abstract_loss','LossReductionType'),
        'AbstractLoss':('loss.abstract_loss','AbstractLoss'),
        'MetricResult':('metrics.abstract_metric','MetricResult'),
        'MetricLocation':('metrics.abstract_metric','MetricLocation'),
        'MetricsContainer':('metrics.metrics_container','MetricsContainer'),
        'default_metrics':('metrics.metrics_container','default_metrics'),
        'TrainMetricData':('metrics.train.abstract_train_metric','TrainMetricData'),
        'AbstractOptimizerWithReset':('optimizer.abstract_optimizer','AbstractOptimizerWithReset'),
        'AbstractActivationResampler':('activation_resampler.abstract_activation_resampler','AbstractActivationResampler'),
        'ParameterUpdateResults':('activation_resampler.abstract_activation_resampler','ParameterUpdateResults'),
        'Axis':('tensor_types','Axis')}
    values={key:getattr(importlib.import_module(name+'.'+module),attribute) for key,(module,attribute) in exports.items()}
    for key,value in values.items():setattr(sys.modules[name],key,value)
    for name,module in list(sys.modules.items()):
        if name.startswith('sparse_autoencoder.') and getattr(module,'__file__',None) and not Path(module.__file__).resolve().is_relative_to(package):raise ValueError('SAE imported an unverified external numerical module')
    return SimpleNamespace(**values)


@lru_cache(maxsize=1)
def definitions():
    path,_=root();sae=sparse_definitions()
    import argparse,random,numpy as np,math,datetime,wandb
    from jaxtyping import Int64
    from pydantic import NonNegativeInt,PositiveInt,validate_call
    from tqdm.auto import tqdm
    ns={**sae.__dict__,'torch':torch,'Tensor':torch.Tensor,'nn':nn,'F':F,'Path':Path,'TYPE_CHECKING':False,
        'quote_plus':__import__('urllib.parse',fromlist=['quote_plus']).quote_plus,'Int64':Int64,
        'NonNegativeInt':NonNegativeInt,'PositiveInt':PositiveInt,'validate_call':validate_call,
        'DataLoader':torch.utils.data.DataLoader,'tqdm':tqdm,'wandb':wandb,'time':__import__('time').time}
    pipeline=load_definitions('protomil','dncbm/custom_pipeline.py',['Pipeline'],ns)
    model=load_definitions('protomil','models/ProtoMIL.py',['Sum','Attn_Net_Gated','ProtoMIL'],ns)
    config=load_definitions('protomil','scripts/train_sae.py',['get_common_parser'],{'argparse':argparse})
    import torchmetrics
    from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_fscore_support
    mil_ns={**ns,'os':__import__('os'),'np':np,'argparse':argparse,'torchmetrics':torchmetrics,
        'roc_auc_score':roc_auc_score,'roc_curve':roc_curve,'precision_recall_fscore_support':precision_recall_fscore_support,
        'ProtoMIL':model.ProtoMIL,'statistics':__import__('statistics')}
    mil=load_definitions('protomil','scripts/train_MIL.py',
        ['ImageWiseCoverageEnergyPercent','str2bool','get_parser','prepare_model','train_model','eval_model'],mil_ns)
    return SimpleNamespace(sae=sae,Pipeline=pipeline.Pipeline,model=model,parser=config.get_common_parser,mil=mil)



def model(classes=2):return definitions().model.ProtoMIL(input_dims=2048,output_dims=classes)


def autoencoder():return sparse_definitions().SparseAutoencoder(n_input_features=512,n_learned_features=2048,n_components=1)


def loss():
    d=sparse_definitions();return d.LossReducer(d.LearnedActivationsL1Loss(l1_coefficient=3e-4),d.L2ReconstructionLoss())


def optimizer(network):
    d=sparse_definitions();return d.AdamWithReset(params=network.parameters(),named_parameters=network.named_parameters(),
        lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,has_components_dim=True)


@lru_cache(maxsize=1)
def mil_epoch_definition():
    import ast,copy
    from .akdpmp_source import _compile
    path,sha=root();tree=ast.parse(pinned_text(path,sha,'scripts/train_MIL.py'))
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_model')
    epoch=next(n for n in ast.walk(train) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='e')
    index=next(i for i,n in enumerate(epoch.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))
    body=copy.deepcopy(epoch.body[:index+1])
    assert isinstance(body[-1],ast.For) and ast.unparse(body[-1].target)=='(batch_idx, data)'
    body+=ast.parse("return {'loss':total_loss/total_batches,'classification_loss':total_classification_loss/total_batches,'classifier_L1':total_sparsity_loss/total_batches,'updates':total_batches}").body
    return _compile('literal_ProtoMIL_training_epoch',
        ['args','model','optimizer','train_loader','classification_loss_fn','sparsity_loss_fn','device'],body,
        definitions().mil.train_model.__globals__,path/'scripts/train_MIL.py')


def mil_epoch(args,model,optimizer,loader,device):
    return mil_epoch_definition()(args,model,optimizer,loader,nn.CrossEntropyLoss(),nn.L1Loss(),device)


def mil_validation(args,model,loader,device):
    # Original eval_model uses module globals for args and val_dataloader.
    # Bind them locally for this actual validation call, leaving source intact.
    original=definitions().mil.eval_model
    function=types.FunctionType(original.__code__,{**original.__globals__,'args':args,'val_dataloader':loader},
        original.__name__,original.__defaults__)
    function.__kwdefaults__=original.__kwdefaults__
    value=function(model,loader,2,'CE',args.probe_sparsity_loss_lambda,device)
    return {'loss':float(value[0]),'classification_loss':float(value[1]),'classifier_L1':float(value[2]),
        'accuracy':float(value[3]),'auroc':float(value[7]),'source_accuracy':float(value[8]),
        'source_zero_weights':{str(k):int(v) for k,v in value[6].items()}}



def runtime_identity():
    import importlib.metadata
    pins={'numpy':'1.26.3','pydantic':'2.5.3','pydantic-core':'2.14.6','jaxtyping':'0.2.25','einops':'0.7.0',
        'protobuf':'4.25.1','wandb':'0.16.1','torchmetrics':'1.3.0.post0','lightning-utilities':'0.10.1','packaging':'23.2'}
    versions={key:importlib.metadata.version(key) for key in pins}
    if versions!=pins or torch.__version__.split('+')[0]!='2.5.1':raise ValueError('ProtoMIL requires its isolated reviewed numerical/validation dependency runtime')
    return {'dependencies':versions,'torch_actual':torch.__version__,'Pillow':importlib.metadata.version('Pillow'),
        'private_SAE_dependency_commit':DTN_COMMIT,'precision':'original_float32',
        'source_environment_Torch2_1_2_reviewed_Torch2_5_port':True}
