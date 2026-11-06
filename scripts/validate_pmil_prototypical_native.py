"""Full30/30 native GPU PMIL proof; no reduced-budget/width CPU substitute."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from milplus.engine import train
from milplus.icmil_coupled_training import load,checkpoint_path
from milplus.io import digest_file,read_json,write_json
from milplus.pmil_prototypical_demo import create_demo
from milplus.pmil_prototypical_driver import PROFILE
from milplus.provenance import software_identity


def equal(a,b,path=''):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    else:assert a==b,(path,a,b)


def fresh_evaluate(run,device):
    launcher="import torch,runpy,sys;torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;sys.argv=['milplus']+sys.argv[1:];runpy.run_module('milplus',run_name='__main__')"
    cmd=[sys.executable,'-c',launcher,'evaluate','--run',str(run),'--device',device]
    result=subprocess.run(cmd,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    if result.returncode:raise RuntimeError(result.stderr+'\n'+result.stdout)
    return digest_file(run/'test_predictions.csv')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weights',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    assert args.device.startswith('cuda:') and torch.cuda.is_available() and torch.version.cuda,'Actual native CUDA required'
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg={'epochs':30,'threads':2,'model':{'encoder_epochs':30}}
    report={'state':'running','software':software_identity(),'profile':PROFILE,'config':cfg,
        'per_slide_RGB256_patches':257,'native_input_dimension':512,'native_metric_projection':128,
        'native_encoder':'full pretrained ResNet34[3,4,6,3]','native_train_batch':128,'native_inference_batch':256,
        'synthetic':True,'real_WSI':False,'formal_admission':False,'conditions':[],
        'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)}
    write_json(out/'validation.json',report);begin=time.monotonic()
    try:
        data=create_demo(out/'data',args.weights,patches=257)
        continuous=train(data,PROFILE,out/'continuous',seed=2026,device=args.device,config=cfg)
        completed_encoder=load(continuous/'max_mil/last.pt')['epoch']
        assert 1<=completed_encoder<=30 and load(continuous/'pmil/last.pt')['epoch']==30
        one=fresh_evaluate(continuous,args.device);two=fresh_evaluate(continuous,args.device);assert one==two
        resumed=train(data,PROFILE,out/'resumed',seed=2026,device=args.device,config=cfg,stop_after=1)
        first=read_json(resumed/'status.json');assert first['state']=='interrupted' and first['phase']=='max_mil'
        p=checkpoint_path(resumed/'max_mil/last.pt');shutil.copyfile(p,resumed/'max_mil/interrupted_original.pt')
        # The second stop is one genuine bag update epoch after completing the
        # original-budget encoder stage, respecting actual source early stop.
        train(data,PROFILE,resumed,seed=2026,device=args.device,config=cfg,resume=True,stop_after=completed_encoder)
        second=read_json(resumed/'status.json');assert second['state']=='interrupted' and second['phase']=='pmil' and second['phase_completed_epochs']==1
        p=checkpoint_path(resumed/'pmil/last.pt');shutil.copyfile(p,resumed/'pmil/interrupted_original.pt')
        train(data,PROFILE,resumed,seed=2026,device=args.device,config=cfg,resume=True)
        for phase in ['max_mil','pmil']:
            a,b=load(continuous/phase/'last.pt'),load(resumed/phase/'last.pt')
            for k in ['model','optimizer','scheduler','rng','epoch','history','best','updates']:equal(a[k],b[k],phase+'/'+k)
            report['conditions'].append({'stage':phase,'state':'passed','completed_epochs':a['epoch'],
                'optimizer_updates':a['updates'],'every_weight_BN_moment_RNG_and_history_exact':True})
        equal(load(continuous/'prototype_discovery/prototypes.pt')['result'],load(resumed/'prototype_discovery/prototypes.pt')['result'])
        assert fresh_evaluate(resumed,args.device)==one
        report.update(state='passed',actual_interruption_points=[first,second],two_fresh_process_selected_reloads=True,
            prediction_sha256=one,seconds=time.monotonic()-begin,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(out/'validation.json',report)
    except BaseException as error:
        report.update(state='failed',error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-begin)
        write_json(out/'validation.json',report);raise


if __name__=='__main__':main()
