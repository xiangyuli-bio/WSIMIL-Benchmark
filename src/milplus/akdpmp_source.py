"""Pinned AKD/PMP models, buffer and literal author training/replay blocks."""
import ast
from collections import defaultdict
from functools import lru_cache
import copy
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn,einsum
from torch.nn import functional as F
from einops import rearrange,reduce

from .upstream import verified_source,pinned_text,load_definitions

COMMIT='a1c83d2efe31bd713b0f7999715297700bfdbb80'
TOPK_COMMIT='12c1645f187e2fa0c05f47bf1fe48864d4bd2707'


def to_device(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,list):return [to_device(v,device) for v in value]
    if isinstance(value,tuple):return tuple(to_device(v,device) for v in value)
    return value


@lru_cache(maxsize=1)
def smooth_svm():
    _,sha=verified_source('smooth_topk')
    if sha!=TOPK_COMMIT:raise ValueError('Use the pinned original smooth-topk dependency')
    ns={'torch':torch,'nn':nn,'np':np,'ag':torch.autograd,'math':math}
    utilities=load_definitions('smooth_topk','topk/utils.py',['delta','detect_large'],ns)
    numeric=load_definitions('smooth_topk','topk/polynomial/sp.py',['log_sum_exp'],ns)
    functional=load_definitions('smooth_topk','topk/functional.py',['Top1_Hard_SVM','Top1_Smooth_SVM'],
        {**ns,'delta':utilities.delta,'log_sum_exp':numeric.log_sum_exp})
    original=load_definitions('smooth_topk','topk/svm.py',['_SVMLoss','SmoothTop1SVM'],
        {**ns,'F':functional,'detect_large':utilities.detect_large})

    class DeviceClosureSmoothTop1SVM(original.SmoothTop1SVM):
        # The old library only refreshes its labels-capturing functions after
        # .cuda()/.cpu(). Fabric/model.to() uses _apply instead. Rebind the exact
        # original functions to the actually moved labels, with no loss change.
        def _apply(self,fn,recurse=True):
            result=super()._apply(fn,recurse=recurse);self.get_losses();return result

    return SimpleNamespace(original=original.SmoothTop1SVM,compatible=DeviceClosureSmoothTop1SVM)


def _compile(name,args,body,namespace,path):
    node=ast.FunctionDef(name=name,args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=a) for a in args],
        vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=copy.deepcopy(body),decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    scope=dict(namespace);exec(compile(module,str(path),'exec'),scope)
    return scope[name]


@lru_cache(maxsize=1)
def definitions():
    root,sha=verified_source('akdpmp')
    if sha!=COMMIT:raise ValueError('AKD-PMP requires its exact official source commit')
    ns={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'ceil':math.ceil,'einsum':einsum,
        'rearrange':rearrange,'reduce':reduce,'random':__import__('random'),'defaultdict':defaultdict,
        'move_data_to_device':to_device}
    svm=smooth_svm()
    clam=load_definitions('akdpmp','models/clam.py',['Attn_Net','Attn_Net_Gated','CLAM_SB'],
        {**ns,'SmoothTop1SVM':svm.compatible},replacements=[('            from topk.svm import SmoothTop1SVM','            # Exact pinned loss supplied privately with device-closure compatibility')])
    trans=load_definitions('akdpmp','models/transmil.py',
        ['exists','moore_penrose_iter_pinv','NystromAttention','TransLayer','PPEG','TransMIL'],ns)
    from .engine import seed_everything
    utilities=load_definitions('akdpmp','main_cl.py',
        ['SimpleBuffer','ClsssIncrementalBuffer','EarlyStopping','kd_loss_fn','distill_slide'],
        {**ns,'seed_everything':seed_everything},replacements=[('np.Inf','np.inf')])
    code=pinned_text(root,sha,'main_cl.py');tree=ast.parse(code)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='one_fold')
    task=next(n for n in function.body if isinstance(n,ast.For) and ast.unparse(n.target)=='task')
    epoch=next(n for n in task.body if isinstance(n,ast.For) and ast.unparse(n.target)=='i')
    native_train=next(n for n in epoch.body if isinstance(n,ast.For) and ast.unparse(n.target)=='(batch_idx, batch)')
    mode=next(n for n in epoch.body if isinstance(n,ast.Expr) and ast.unparse(n)=='model.train()')
    native_val=[n for n in epoch.body if isinstance(n,ast.For) and ast.unparse(n.target)=='(batch_idx, batch)'][1]
    common={**ns,'kd_loss_fn':utilities.kd_loss_fn,'seed_everything':seed_everything}
    # These AST nodes are the unmodified current-step and memory-replay body
    # from one_fold, not a generic CE trainer or a retyped approximation.
    epoch_fn=_compile('author_train_epoch',
        ['args','model','optimizer','train_loader','buffer','old_model','seen_classes','old_seen_classes',
         'unseen_classes','task','i','fabric','train_loss_metric','logger'],[mode,native_train],common,root/'main_cl.py')
    val_fn=_compile('author_val_epoch',
        ['args','model','val_loader','seen_classes','unseen_classes','fabric','val_loss_metric','val_acc_metric'],
        [ast.parse('model.eval()').body[0],native_val],common,root/'main_cl.py')
    # The complete selected-model seed/reset, class-quota adjustment, reservoir
    # sampling and attention-driven pseudo-bag construction, without CSV logging.
    refill=next(n for n in task.body if isinstance(n,ast.If) and n.lineno==737)
    refill_body=[n for n in refill.body if n.lineno<=776]
    refill_fn=_compile('author_refill',
        ['args','model','buffer','train_loader','seen_classes','cur_classes','fabric','distill_slide'],refill_body,common,root/'main_cl.py')
    # An index-returning view of the exact source selector enables provenance;
    # its RNG, top-k, duplicate indices and integer rounding stay untouched.
    selector=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='distill_slide')
    index_body=copy.deepcopy(selector.body);assert ast.unparse(index_body[-1])=='return slide[idx]'
    index_body[-1]=ast.Return(value=ast.Name(id='idx',ctx=ast.Load()))
    index_fn=_compile('author_selection_indices',['slide','attn','size','method'],index_body,common,root/'main_cl.py')
    return SimpleNamespace(clam=clam,trans=trans,utilities=utilities,epoch=epoch_fn,validation=val_fn,refill=refill_fn,selection_indices=index_fn)


def model(kind,dimension,classes):
    if kind=='transmil':return definitions().trans.TransMIL(D_feat=dimension,D_inner=512,n_classes=classes)
    if kind=='clam':return definitions().clam.CLAM_SB(D_feat=dimension,L=512,D=256,K=1,dropout=True,
        gate=True,k_sample=8,n_classes=classes,instance_loss_name='svm',subtyping=True)
    raise ValueError('AKD-PMP requires the original TransMIL or gated CLAM-SB with smooth SVM instance loss')


def arguments(kind,preset,classes,config,tasks):
    if preset not in {'tcga','skin'}:raise ValueError('Choose an explicit source TCGA or skin recipe')
    defaults={'tcga':{'epochs':50,'lr':2e-4,'weight_decay':1e-5,'buffer_size':300,'buffer_balance':True,
        'buffer_slide_size':.1,'weighted_sample':True},
        'skin':{'epochs':[20,40,25] if kind=='transmil' else [40,50,50],
            'lr':[1e-5,1e-6,1e-6,1e-6,1e-6] if kind=='transmil' else [1e-4]*5,
            'weight_decay':0.,'buffer_size':42,'buffer_balance':False,'buffer_slide_size':-1.,'weighted_sample':False}}
    cfg=copy.deepcopy(defaults[preset])
    if set(config)-{'epochs','threads','buffer_size','distill_method'}:raise ValueError('Preserve original AKD/PMP losses, replay, sampling, optimizer and full512D model widths')
    if 'epochs' in config:cfg['epochs']=config['epochs']
    if 'buffer_size' in config:cfg['buffer_size']=config['buffer_size']
    for key in ['epochs','lr']:
        values=cfg[key] if isinstance(cfg[key],list) else [cfg[key]]*tasks
        if len(values)<tasks or any(type(v) not in {int,float} or v<=0 or not np.isfinite(v) for v in values):
            raise ValueError('AKD-PMP needs an explicit valid '+key+' value for every actual task')
        if key=='epochs' and any(type(v) is not int for v in values):raise ValueError('Every actual task epoch budget must be a positive integer')
        cfg[key]=values[:tasks]
    if type(cfg['buffer_size']) is not int or cfg['buffer_size']<classes:raise ValueError('Memory capacity must support all declared classes; never allow an empty/zero-quota replay pool')
    selector=config.get('distill_method','maxrand')
    if selector not in {'random','max','maxmin','maxminrand','maxrand'}:raise ValueError('Choose a literal original PMP selection rule')
    return SimpleNamespace(**cfg,net='transmil' if kind=='transmil' else 'clam_sb',n_classes=classes,
        batch_size=1,n_batches=-1,cl_method='prev',cl_on_current=False,wn=False,distill_method=selector,
        early_stop=True,patience=10,seed=1,opt='adam')


class Fabric:
    """Literal single-device/full-float Fabric operations; logging has no math."""
    def __init__(self,device):self.device=device;self.updates=0
    def to_device(self,value):return to_device(value,self.device)
    def backward(self,loss):
        if not torch.isfinite(loss):raise FloatingPointError('Original AKD/PMP objective became nonfinite')
        loss.backward();self.updates+=1


class Mean:
    # torchmetrics.MeanMetric default CPU float32 accumulation is part of
    # original selection, including its rounding; do not silently use float64.
    def __init__(self):self.values=[];self.total=torch.tensor(0.);self.weight=torch.tensor(0.)
    def update(self,value):
        self.values.append(float(value));self.total+=torch.tensor(float(value));self.weight+=torch.tensor(1.)
    def mean(self):return float(self.total/self.weight)


class Accuracy:
    def __init__(self):self.correct=0;self.count=0
    def update(self,pred,target):self.correct+=int((pred==target).sum());self.count+=len(target)


class Logger:
    def __init__(self):self.rows=[]
    def log_metrics(self,row):self.rows.append(dict(row))
