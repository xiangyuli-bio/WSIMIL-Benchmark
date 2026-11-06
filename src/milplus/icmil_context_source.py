"""Actual released PFN/Perceiver ICMIL; distinct from iteratively coupled ICMIL."""
import sys
from pathlib import Path

from .io import digest_file
from .upstream import verified_source

COMMIT='23bb7203777891987ec3c18a1233ffe238b032b3'
ARCH={'in_features':25,'embedding_size':256,'mlp_hidden_size':1054,'num_attention_heads':4,
    'num_column_row_iterations':6,'num_final_col_row_layers':0,'num_outputs':2,
    'feature_group_size':1,'bag_chunk_size':32,'gradient_checkpointing':False}
WEIGHTS={'c5trd795':'4be2e171a8393148c531f2428c505ff20f2bd2c7abd4ab97bf92726a29947340',
    'ggwsqibd':'c52cc0ce16f1cf17dbbf5c53447eea64b2b5a982c4c9baeb90b02037e4a86ba0',
    'k337zhz1':'f9c1fbe298eee2da26e5ae46ae8d04d75dab3a898beaf074eb771993e7191ded'}


def checked_source():
    root,commit=verified_source('icmil_context')
    if commit!=COMMIT:raise ValueError('In-context ICMIL original source commit changed')
    return root.resolve()


def checkpoint(seed):
    if seed not in WEIGHTS:raise ValueError('Choose one of the three distinct actually released ICMIL pretrained seeds')
    p=checked_source()/'checkpoints'/('icmil-'+seed+'.pt')
    if digest_file(p)!=WEIGHTS[seed]:raise ValueError('Released ICMIL checkpoint bytes differ from the pinned author artifact')
    return p


def load_native(path,seed,device):
    if sys.version_info[:2]!=(3,11):raise ValueError('Use the separate original Python3.11 ICMIL environment')
    import torch
    root=checked_source();path=Path(path)
    if seed not in WEIGHTS or digest_file(path)!=WEIGHTS[seed]:raise ValueError('Actual original ICMIL pretrained seed is required; no random/small replacement')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['arch']!=ARCH or state['epoch']!=8000:raise ValueError('Original ICMIL256/1054/six-iteration/8000epoch pretrained architecture changed')
    sys.path.insert(0,str(root))
    import icmil
    if not Path(icmil.__file__).resolve().is_relative_to(root):raise ValueError('Another icmil package shadows the pinned original source')
    model=icmil.load_icmil(source=path,seed=seed,device=device)
    if sum(p.numel() for p in model.parameters())!=8257552:raise ValueError('ICMIL full original parameter graph changed')
    return model
