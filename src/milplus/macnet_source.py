"""Privately loaded full original Swin/GFK/adaptive-cluster MacNet definitions."""
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import ast,importlib,importlib.machinery,types,sys,random,math,time,copy
import torch
from torch import nn
from .upstream import verified_source,pinned_text
from .io import digest_file

COMMIT='7610d7a534cd833ba0b139136ed9af9b41f59f48'
SWIN22K_SHA256='108a66d1241f874f8328dbb2876b320e8f541618a7ad9ebe8102f7f18dd4396e'
SWIN22K_URL='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth'


def root():
    p,sha=verified_source('macnet')
    if sha!=COMMIT:raise ValueError('Original pinned MacNet source is required')
    return p.resolve()


def _definitions(file,names,ns):
    p=root();text=pinned_text(p,COMMIT,file)
    # Original Simple_MIL_model_modules.py has inconsistent tabs. Eight-column
    # lexical expansion repairs parsing without replacing numerical expressions.
    if file=='Simple_MIL_model_modules.py':text=text.expandtabs(8)
    tree=ast.parse(text);nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Actual MacNet original definitions changed')
    class CurrentDevice(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value==0:node.args=[]
            return node
    # Port only literal cuda(0) to the explicitly selected current CUDA device.
    # On source device0 every numerical operation is identical.
    nodes=[CurrentDevice().visit(n) for n in nodes]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p/file),'exec'),ns)


@lru_cache(maxsize=1)
def definitions():
    p=root();directory=p/'SwinT_models/flash_models'
    for f in directory.rglob('*.py'):
        if f.read_text()!=pinned_text(p,COMMIT,str(f.relative_to(p))):raise ValueError('Actual Swin/Triton source bytes changed')
    # Isolate the original root's wrong Models.* import path. Every numerical
    # dense/flash function is still loaded from the verified author tree; no
    # dummy attention, truncated encoder or replacement backbone is injected.
    alias='_milplus_original_macnet_flash'
    if alias not in sys.modules:
        package=types.ModuleType(alias);package.__path__=[str(directory)];package.__package__=alias;package.__spec__=importlib.machinery.ModuleSpec(alias,loader=None,is_package=True);sys.modules[alias]=package
    elif list(sys.modules[alias].__path__)!=[str(directory)]:raise ValueError('Another unverified MacNet numerical package is already loaded')
    dense=importlib.import_module(alias+'.func_swin');flash=importlib.import_module(alias+'.func_flash_swin')
    from timm.models.layers import DropPath,to_2tuple,trunc_normal_
    import torch.utils.checkpoint as checkpoint
    ns={'__name__':__name__,'torch':torch,'nn':nn,'F':torch.nn.functional,'math':math,'time':time,'random':random,
        'einops':__import__('einops'),'checkpoint':checkpoint,'DropPath':DropPath,'to_2tuple':to_2tuple,'trunc_normal_':trunc_normal_,
        'window_partition':dense.window_partition,'window_reverse':dense.window_reverse,'mha_core':dense.mha_core,'flash_swin_attn_func':flash.flash_swin_attn_func}
    _definitions('SwinT_models/flash_models/swin_transformer.py',['WindowAttention','Mlp','SwinTransformerBlock','PatchMerging','BasicLayer','PatchEmbed','SwinTransformer'],ns)
    _definitions('Utils/GFK_torch.py',['degenerated_GSVD','null_space','sqrt_SymTensor','GFK'],ns)
    _definitions('Simple_MIL_model_modules.py',['TgiClustering','Sef_Cluster','MIL_Parallel_Feature','MIL_Parallel_Head','MIL_Parallel_Head_ub'],ns)
    _definitions('MIL_main_public.py',['create_swin_base'],ns)
    return SimpleNamespace(**{k:v for k,v in ns.items() if not k.startswith('__')})


def original_base(weights,classes):
    if not torch.cuda.is_available():raise ValueError('Original MacNet requires actual CUDA; no automatic CPU substitute')
    if classes not in {2,3}:raise ValueError('Use original binary public or three-grade private MacNet tasks')
    weights=Path(weights).resolve()
    if digest_file(weights)!=SWIN22K_SHA256:raise ValueError('Verify actual official author-compatible ImageNet22K Swin-T weights, not ImageNet1K or random initialization')
    d=definitions();raw=d.create_swin_base
    args=SimpleNamespace(img_size=[96,96],class_num=classes,is_flash=False,pretrained_weights_path=str(weights))
    function=types.FunctionType(raw.__code__,{**raw.__globals__,'args':args},raw.__name__,raw.__defaults__)
    base=function()
    checkpoint=torch.load(weights,map_location='cpu',weights_only=True)['model'];actual=base.state_dict()
    allowed=lambda k:any(s in k for s in ['relative_position_index','relative_coords_table','attn_mask']) or k.startswith('head.')
    missing=[k for k in actual if not allowed(k) and k not in checkpoint];unexpected=[k for k in checkpoint if k not in actual and not allowed(k)]
    if missing or unexpected:raise ValueError('Original pretrained Swin feature keys missing/unexpected: '+repr((missing,unexpected)))
    # Independently verify every transformed feature parameter after the literal
    # author loader; classifier reinitialization is deliberately original.
    for name,value in actual.items():
        if allowed(name):continue
        expected=checkpoint[name]
        if 'relative_position_bias_table' in name and expected.shape!=value.shape:
            a,b=expected.shape;h,w=value.shape;size1=int(a**.5);size2=int(h**.5)
            expected=nn.functional.interpolate(expected.permute(1,0).view(1,b,size1,size1),size=(size2,size2),mode='bicubic').view(w,h).permute(1,0)
        if not torch.equal(value,expected):raise ValueError('Original Swin feature initialization changed: '+name)
    return base


class FullMacNet(nn.Module):
    """Complete raw Swin plus literal public/private head and retained cache."""
    def __init__(self,weights,classes=2,condition='public',adaptive='source_cached'):
        super().__init__()
        if condition not in {'public','lscc81','cscc64'} or adaptive not in {'source_cached','refresh_repaired'}:raise ValueError('Choose explicit actual original MacNet condition and adaptive-cache semantics')
        if (classes==2)!=(condition=='public'):raise ValueError('Original public/private class and guidance counts differ')
        d=definitions();base=original_base(weights,classes)
        guides={'public':5,'lscc81':81,'cscc64':64}[condition];bags={'public':2000,'lscc81':1042,'cscc64':1025}[condition]
        self.feature=d.MIL_Parallel_Feature(base_model=base,img_size=[96,96])
        head=d.MIL_Parallel_Head_ub if condition=='public' else d.MIL_Parallel_Head
        self.head=head(base_model=base,class_num=classes,batch_size=1 if condition=='public' else 2,bags_len=bags,model_stats='train',seed=0)
        self.register_buffer('initial_adaptive_diagonal',self.head.tgi_clustering_block.inst_adapt_w.detach().clone())
        self.condition=condition;self.adaptive=adaptive;self.guide_count=guides;self.original_bag_length=bags
        self.head.register_forward_pre_hook(lambda module,inputs:self._bind_adaptive())

    def _bind_adaptive(self):
        block=self.head.tgi_clustering_block;current=torch.diag(block.init_w)
        # Match the released stale diagonal values and its gradient identity,
        # while rebinding the graph in each call so a fresh checkpoint process
        # retains this exact source runtime state. The repair is an explicit
        # separate condition; it never silently replaces the source profile.
        block.inst_adapt_w=(self.initial_adaptive_diagonal+(current-current.detach())) if self.adaptive=='source_cached' else current

    def forward(self,rgb):
        return self.head(self.feature(rgb))


@lru_cache(maxsize=1)
def training_definitions():
    import numpy as np
    from sklearn.metrics import accuracy_score
    ns={'__name__':__name__,'torch':torch,'nn':nn,'np':np,'os':__import__('os'),'accuracy_score':accuracy_score,
        'autocast':torch.cuda.amp.autocast,'GradScaler':torch.cuda.amp.GradScaler}
    _definitions('Utils/train_val.py',['vit_lr_schedule','view_results','training_for_parallel'],ns)
    return SimpleNamespace(**{k:v for k,v in ns.items() if not k.startswith('__')})


@lru_cache(maxsize=1)
def literal_epoch_definition():
    # Extract the complete original epoch through its real raw training loop.
    # Additional statements only record detached losses after each update.
    from .akdpmp_source import _compile
    tree=ast.parse(pinned_text(root(),COMMIT,'Utils/train_val.py'))
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training_for_parallel')
    outer=next(n for n in train.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='i')
    index=next(i for i,n in enumerate(outer.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))
    body=copy.deepcopy(outer.body[:index+1]);body[-1].body+=ast.parse('epoch_statistics.append(float(loss_value.detach()))').body
    body+=ast.parse('return rmp_optim').body
    return _compile('literal_complete_MacNet_training_epoch',
        ['mil_feature','mil_head','mil_paras','loss_fn','scaler','train_loader','i','max_input_len','use_amp','epoch_statistics'],
        body,training_definitions().training_for_parallel.__globals__,root()/'Utils/train_val.py')


def train_epoch(model,groups,loader,epoch,scaler,use_amp=False):
    statistics=[]
    optimizer=literal_epoch_definition()(model.feature,model.head,groups,nn.CrossEntropyLoss(),scaler,loader,epoch,2000,use_amp,statistics)
    if not statistics or not all(math.isfinite(x) for x in statistics):raise FloatingPointError('Complete original MacNet raw epoch produced no finite objectives')
    return optimizer,statistics
