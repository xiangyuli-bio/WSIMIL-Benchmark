"""Full original200epoch phases, native RGB256/1024/512/128 widths, all3bases.

Each condition executes a continuous full loop and a genuinely interrupted,
resumed full loop. Every final weight, BN buffer, Adam moment, selected score,
mutable feature permutation and RNG is compared exactly. This is synthetic
execution evidence, not new patient data or original-paper scores.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

from milplus.engine import train,evaluate
from milplus.icmil_coupled_demo import create_demo
from milplus.icmil_coupled_driver import PROFILES
from milplus.icmil_coupled_training import load,save
from milplus.io import digest_file,read_json,write_json
from milplus.provenance import software_identity


def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--profile',action='append');a=p.parse_args()
    if not torch.cuda.is_available() or torch.device(a.device).type!='cuda':raise ValueError('Full native coupled gate requires actual allocated CUDA; no CPU fallback')
    torch.cuda.set_device(a.device);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False);results=[]
    profiles=list(PROFILES) if a.profile is None else a.profile
    if not profiles or len(set(profiles))!=len(profiles) or set(profiles)-set(PROFILES):raise ValueError('Unknown or duplicated full native condition')
    cfg={'epochs':200,'threads':2,'model':{'iterations':1,'distill_epochs':200}}
    state={'state':'running','conditions':results,'software':software_identity(),'actual_torch':torch.__version__,
        'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'strict_determinism':True,'config':cfg,
        'per_slide_patches':101,'formal_admission':False,'real_wsi':False}
    write_json(root/'validation.json',state)
    try:
        for profile in profiles:
            folder=root/profile;folder.mkdir();data=create_demo(folder/'data',a.weights,patches=101)
            before=time.monotonic();torch.cuda.reset_peak_memory_stats()
            print('Starting complete native200/200/200 condition',profile,flush=True)
            continuous=train(data,profile,folder/'continuous',seed=1173,device=a.device,config=cfg)
            resumed=train(data,profile,folder/'resumed',seed=1173,device=a.device,config=cfg,stop_after=201)
            stopped=read_json(resumed/'status.json')
            assert stopped['state']=='interrupted' and stopped['phase']=='distill_0' and stopped['phase_completed_epochs']==1
            interrupted=load(resumed/'distill_0/last.pt');assert interrupted['epoch']==1 and interrupted['updates']==3
            save(interrupted,folder/'actual_interrupted_encoder_state.pt')
            train(data,profile,resumed,seed=1173,device=a.device,config=cfg,resume=True)
            stage_rows=[]
            for stage in ['bag_0','distill_0','bag_1']:
                x=load(continuous/stage/'last.pt');y=load(resumed/stage/'last.pt');exact(x,y)
                assert x['epoch']==200 and len(x['history'])==200 and x['finished'] is True
                if stage=='distill_0':assert x['updates']==600 and all(r['patches']==202 for r in x['history'])
                else:assert x['updates']==(800 if PROFILES[profile]=='dtfd' else 400)
                stage_rows.append({'stage':stage,'epochs':200,'optimizer_updates':x['updates'],
                    'every_weight_moment_RNG_order_history_bitwise_equal':True})
            evaluate(continuous,device=a.device);prediction=digest_file(continuous/'test_predictions.csv')
            evaluate(continuous,device=a.device);assert digest_file(continuous/'test_predictions.csv')==prediction
            evaluate(resumed,device=a.device);assert digest_file(resumed/'test_predictions.csv')==prediction
            assert torch.are_deterministic_algorithms_enabled()
            results.append({'state':'passed','profile':profile,'native_widths':{'RGB':256,'features':1024,'MIL':512,'attention':128},
                'TRAIN_patches':202,'VAL_patches':202,'source_patch_batch':100,'source_extract_batch':448,
                'stages':stage_rows,'actual_interruption':stopped,'double_fresh_selected_reload':True,
                'prediction_sha256':prediction,'seconds':time.monotonic()-before,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'synthetic':True})
            write_json(root/'validation.json',state)
        state['state']='passed';write_json(root/'validation.json',state)
    except BaseException as error:
        state.update(state='failed',error=repr(error));write_json(root/'validation.json',state);raise


if __name__=='__main__':main()
