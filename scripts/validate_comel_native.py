"""Complete original100 epochs x5 tasks; full widths and genuine recovery."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from milplus.comel_source import PROFILES,COMMIT
from milplus.comel_demo import create_demo
from milplus.comel_driver import train,evaluate
from milplus.icmil_coupled_training import load
from milplus.io import write_json,digest_file
from milplus.provenance import software_identity


def same(a,b,path='state'):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError('CoMEL exact recovery differs at '+path)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'.'+str(i))
    else:assert a==b,(path,a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if torch.device(args.device).type!='cuda' or not torch.cuda.is_available():raise ValueError('Full native validator requires an actual allocated CUDA GPU')
    data=create_demo(out/'data',tasks=5,patches=1025,dimension=1024)
    rows=[];started=time.monotonic()
    for profile in sorted(PROFILES):
        target=out/profile;target.mkdir();before=time.monotonic();torch.cuda.reset_peak_memory_stats()
        whole=train(data,profile,target/'uninterrupted',seed=2021,device=args.device)
        baseline=load(whole/'last.pt')
        assert baseline['updates']==1000 and len(baseline['history'])==500
        assert baseline['next_task']==5 and baseline['architecture']['heads']==5
        resumed=train(data,profile,target/'resumed',seed=2021,device=args.device,stop_after=1)
        assert json.loads((resumed/'status.json').read_text())['state']=='interrupted'
        train(data,profile,resumed,seed=2021,device=args.device,resume=True,stop_after=99)
        boundary=load(resumed/'last.pt');assert boundary['next_task']==1 and boundary['next_epoch']==0
        train(data,profile,resumed,seed=2021,device=args.device,resume=True,stop_after=101)
        later=load(resumed/'last.pt');assert later['next_task']==2 and later['next_epoch']==1
        train(data,profile,resumed,seed=2021,device=args.device,resume=True)
        actual=load(resumed/'last.pt')
        for key in ['architecture','model','gradients','owner','history','validation_matrix','thresholds','updates','rng','training_modes']:
            same(baseline[key],actual[key],key)
        for left,right in zip(baseline['snapshots'],actual['snapshots']):same(load(left['path']),load(right['path']))
        expected=evaluate(whole,'test',args.device);predictions=(whole/'test_predictions.csv').read_bytes()
        # Both selected reloads execute the actual public CLI in fresh Python
        # interpreters. Installed-wheel callers set PYTHONPATH only to that
        # installed target; no development source is inserted here.
        for index in range(2):
            cmd=[sys.executable,'-m','milplus','evaluate','--run',str(whole),'--subset','test','--device',args.device]
            result=subprocess.run(cmd,capture_output=True,text=True,env=os.environ.copy())
            (target/f'fresh_cli_{index}.log').write_text(result.stdout+result.stderr)
            assert result.returncode==0,result.stderr
            assert json.loads((whole/'test_metrics.json').read_text())==expected
            assert (whole/'test_predictions.csv').read_bytes()==predictions
        rows.append({'profile':profile,'state':'passed','tasks':5,'epochs_per_task':100,'actual_epochs':500,
            'optimizer_updates':1000,'actual_feature_dimension':1024,'patches_per_slide':1025,'hidden':512,
            'gdat_blocks':2,'heads_per_block':8,'effective_rank':16,'qkv_rank':48,
            'real_interruption_points':[1,100,201],'exact_all_learned_state_and_gradients_and_rng':True,
            'fresh_selected_cli_reloads':2,'gpu_peak_bytes':torch.cuda.max_memory_allocated(),
            'elapsed_seconds':time.monotonic()-before,'synthetic':True,'paper_score_reproduction':False})
        write_json(out/'progress.json',{'state':'running','completed':rows})
    result={'state':'passed','conditions':rows,'source_commit':COMMIT,'software':software_identity(),
        'elapsed_seconds':time.monotonic()-started,'synthetic':True,'formal_admission':False,
        'timing_scope':'engineering validation on shared allocated GPU; not standalone benchmark efficiency'}
    write_json(out/'validation.json',result);print(json.dumps({'state':'passed','receipt':str(out/'validation.json')}),flush=True)


if __name__=='__main__':main()
