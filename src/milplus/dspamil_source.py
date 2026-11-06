"""Released full DSPA SAT components; no claim of a full DSPA/DSPA-U trainer.

The current author tree explicitly says incomplete. Its SAT is an available
component, not a substitute for the dual concept/teacher/student method.
"""
from functools import lru_cache
import ast,math
import torch
from torch import nn,einsum
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_normal_
from .upstream import verified_source,pinned_text,load_definitions
from .io import digest_file
COMMIT='75d95d410374771a3eba4bd84fdfadc06acf0eed'
NAMES=['Mlp','drop_path','DropPath','Attention','Block','SAT_MIL']


def checked_source():
    root,commit=verified_source('dspamil')
    if commit!=COMMIT:raise ValueError('Use the reviewed original incomplete DSPA source revision')
    return root


@lru_cache(maxsize=1)
def author_definitions():
    checked_source()
    return load_definitions('dspamil','modules/SAT_MIL/sat_mil.py',NAMES,
        {'torch':torch,'nn':nn,'einsum':einsum,'F':torch.nn.functional,'math':math,
         'rearrange':rearrange,'repeat':repeat,'Rearrange':Rearrange,'xavier_normal_':xavier_normal_})


def build_native_source_component():
    return author_definitions().SAT_MIL(num_classes=6,embed_dim=1024,depth=12,num_heads=8)


def original_training_loop():
    checked_source()
    return load_definitions('dspamil','utils/loop_utils.py',['train_loop_SAT'],
        {'torch':torch,'time':__import__('time')}).train_loop_SAT


def source_review():
    from .registry import profiles
    root=checked_source();paths=['modules/SAT_MIL/sat_mil.py','utils/loop_utils.py',
        'utils/wsi_utils.py','process/SAT_MIL/process_sat_mil.py','configs/SAT_MIL.yaml','README.md','LICENSE']
    source={p:pinned_text(root,COMMIT,p) for p in paths}
    dataset=ast.parse(source['utils/wsi_utils.py']);loop=ast.parse(source['utils/loop_utils.py'])
    declared_dataset={n.name for n in dataset.body if isinstance(n,ast.ClassDef)}
    train=next(n for n in loop.body if isinstance(n,ast.FunctionDef) and n.name=='train_loop_SAT')
    unpack=[n for n in ast.walk(train) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Name) and n.value.func.id=='model']
    assert len(unpack)==1 and isinstance(unpack[0].targets[0],ast.Tuple) and len(unpack[0].targets[0].elts)==4
    return {'source_url':'https://github.com/AlexNmSED/DSPA-MIL','commit':COMMIT,'license':'MIT',
        'source_hashes':{p:digest_file(root/p) for p in paths},'declared_dataset_classes':sorted(declared_dataset),
        'required_GS_dataset_released':'WSI_Dataset_GS' in declared_dataset,
        'released_network_return':'dictionary; actual forward verified separately',
        'released_training_unpack':'four positional outputs',
        'released_native_component':{'feature_width':1024,'transformer_layers':12,'attention_heads':8,'pattern_tokens':3,'grade_classes':6},
        'missing_full_method_mechanisms':['patch-concept similarity aggregation',
            'patient concept Teacher-Gate','pseudo-text generator','Student-Gate',
            'gating/output/text knowledge alignment','DSPA-U uncertainty regularization'],
        'scope':'Original source component and executable integration-gap evidence only',
        'literal_author_trainer_runnable':False,
        'runnable_profile_registered':any(p['family']=='dspamil' for p in profiles()),
        'formal_admission':False,'paper_reproduction':'not_run'}
