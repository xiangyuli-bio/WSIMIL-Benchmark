"""Compare adapter tensors to pinned author functions; requires downloaded sources.

Does not import authors' training scripts, datasets, or module initializers.
Only named definitions are evaluated from Git blobs at recorded commits.
"""
import argparse
import ast
import copy
import math
import subprocess
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import Tensor, nn
from torch.nn import functional as F

from milplus.augmentation import phenotype_partition, remix_augment
from milplus.expansion import MADMIL
from milplus.io import write_json
from milplus.prototype_models import Chowder, PrototypeMIL
from milplus.registry import resource


def definitions(source_root, source, path, names, namespace=None):
    lock = resource('sources.lock.json')[source]
    root = source_root / source
    sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if sha != lock['commit']:
        raise ValueError(f'{source} commit differs from lock')
    text = subprocess.check_output(['git','show',f'{sha}:{path}'],cwd=root,text=True)
    tree = ast.parse(text)
    selected = [node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f'{source} reference definitions missing')
    scope = {'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'copy':copy,'warnings':warnings,
             'List':list,'Tuple':tuple,'Optional':Optional,'Union':Union,'Tensor':Tensor,'cdist':cdist}
    scope.update(namespace or {})
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(root/path),'exec'),scope)  # noqa: S102 -- named definitions from fixed Git commit
    return scope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(147)
    np.random.seed(147)
    records=[]
    def verify(name,source,actual,expected,atol=2e-6):
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=atol)
        records.append({'method':name,'source_id':source,'commit':resource('sources.lock.json')[source]['commit'],
                        'max_absolute_error':float((actual-expected).abs().max().detach()),'state':'passed'})

    prefix='rl_benchmarks/models/slide_models/'
    scope=definitions(args.sources,'chowder',prefix+'utils/tile_layers.py',['MaskedLinear','TilesMLP'])
    scope.update(definitions(args.sources,'chowder',prefix+'utils/mlp.py',['MLP']))
    scope.update(definitions(args.sources,'chowder',prefix+'utils/extreme_layer.py',['ExtremeLayer']))
    original=definitions(args.sources,'chowder',prefix+'chowder.py',['Chowder'],scope)['Chowder'](
        16,2,n_top=3,n_bottom=3,tiles_mlp_hidden=[32],mlp_hidden=[32,16],metadata_cols=0)
    adapter=Chowder(16,2,hidden=32,extremes=3)
    for a,b in zip(adapter.parameters(),original.parameters()):
        a.data.copy_(b.data)
    x=torch.randn(27,16)
    verify('chowder','chowder',adapter(x)['logits'],original(x.unsqueeze(0)))

    def initialize_weights(module):
        for m in module.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    scope=definitions(args.sources,'madmil','models/model_abmil.py',
                      ['Attn_Net','Attn_Net_Gated','ABMIL_Multihead'],{'initialize_weights':initialize_weights})
    original=scope['ABMIL_Multihead'](dropout=True,n_classes=2,n=2)
    adapter=MADMIL(1024,2,dropout=.25)
    for a,b in zip(adapter.parameters(),original.parameters()):
        a.data.copy_(b.data)
    x=torch.randn(27,1024)
    # Compare stochastic training as well as inference, including attention dropout.
    for training in [False,True]:
        adapter.train(training);original.train(training)
        torch.manual_seed(91); expected=original(x)[0]
        torch.manual_seed(91); actual=adapter(x)['logits']
        verify('madmil_train' if training else 'madmil_eval','madmil',actual,expected)

    scope=definitions(args.sources,'panther','src/mil_models/PANTHER/networks.py',['mog_eval','DirNIWNet'])
    original=scope['DirNIWNet'](3,8,load_proto=False)
    adapter=PrototypeMIL(8,2,'panther',prototypes=3)
    adapter.prototypes.copy_(original.m.detach());adapter.prepared.fill_(True)
    x=torch.randn(21,8)
    pi,mu,var,_=original.map_em(x.unsqueeze(0),num_iters=3,tau=10.)
    expected=torch.cat((pi.flatten(),mu.flatten(),var.flatten())).unsqueeze(0)
    verify('panther_map_em','panther',adapter.representation(x),expected)

    original=definitions(args.sources,'psemix','utils/core.py',['PseudoBag'])['PseudoBag'](
        4,3,clustering_method='ProtoDiv',proto_method='mean',pheno_cut_method='uniform',iter_fine_tuning=8)
    x=torch.randn(85,12)
    torch.manual_seed(61); expected=original.divide(x)
    torch.manual_seed(61); actual=phenotype_partition(x,4,3,refinements=8)
    verify('psemix_partition','psemix',actual,expected,atol=0)

    original=definitions(args.sources,'remix','train_remix.py',['mix_aug'])['mix_aug']
    x,y=torch.randn(8,512),torch.randn(8,512)
    shifts=torch.randn(8,200,512)*.1
    for mode in ['replace','append','interpolate','cov','joint']:
        np.random.seed(73); expected=original(x.numpy(),y.numpy(),mode=mode,rate=.7,strength=.4,shift=shifts.numpy())
        np.random.seed(73); actual=remix_augment(x,y,shifts,.7,.4,mode)
        verify('remix_'+mode,'remix',actual,torch.from_numpy(expected))
    write_json(args.output,{'checks':records,'scope':'Numerical mechanisms on synthetic tensors, not paper reproduction'})
    print(f'{len(records)} numerical author comparisons passed')


if __name__=='__main__':
    main()
