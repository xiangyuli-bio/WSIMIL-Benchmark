"""Full native original four-task budgets, both query choices, exact recovery."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from milplus.qpmilvl_driver import train,evaluate,PROFILES
from milplus.data import Bundle
from milplus.icmil_coupled_training import load
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
from milplus.qpmilvl_source import runtime_identity,COMMIT


def same(a,b,path='state'):
    if torch.is_tensor(a):assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'.'+str(i))
    else:assert a==b,(path,a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--data',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic();completed=[]
    try:
        if not a.device.startswith('cuda') or not torch.cuda.is_available():raise ValueError('Full native gate requires an actual allocated CUDA device')
        bundle=Bundle(a.data);assets=bundle.qpmilvl
        assert assets is not None and bundle.dim==512 and len(assets.tasks)==4 and bundle.config['synthetic'] is True
        assert all(len(assets.records(t,'train'))==32 for t in range(4))
        assert all(r['patches']==4097 for r in assets.rows.values());input_contract=assets.contract
        for profile in PROFILES:
            target=out/profile;target.mkdir();started=time.monotonic();torch.cuda.reset_peak_memory_stats()
            run=train(a.data,target/'continuous',device=a.device,method=profile);baseline=load(run/'last.pt');snapshots=baseline['snapshots']
            assert len(snapshots)==4 and all(r['maximum_epochs']==12 and 6<=r['actual_epochs']<=12 for r in snapshots)
            assert baseline['updates']==2*sum(r['actual_epochs'] for r in snapshots)
            metrics=evaluate(run,device=a.device);predictions=(run/'test_predictions.csv').read_bytes()
            boundary=snapshots[0]['actual_epochs']
            resumed=train(a.data,target/'resumed',device=a.device,method=profile,stop_after=1)
            assert read_json(resumed/'status.json')['state']=='interrupted'
            train(a.data,resumed,device=a.device,method=profile,resume=True,stop_after=boundary-1)
            mid=load(resumed/'last.pt');assert mid['next_task']==1 and mid['next_epoch']==0
            train(a.data,resumed,device=a.device,method=profile,resume=True,stop_after=1)
            mid=load(resumed/'last.pt');assert mid['next_task']==1 and mid['next_epoch']==1
            train(a.data,resumed,device=a.device,method=profile,resume=True);actual=load(resumed/'last.pt')
            for key in ['pools','gradients','next_task','next_epoch','optimizer','scheduler','early','frequency','val_frequency','history','updates','generators','rng']:same(baseline[key],actual[key],key)
            for left,right in zip(snapshots,actual['snapshots']):same(load(left['path']),load(right['path']))
            assert evaluate(resumed,device=a.device)==metrics and (resumed/'test_predictions.csv').read_bytes()==predictions
            for i in range(2):
                cmd=[sys.executable,'-m','milplus','evaluate','--run',str(run),'--device',a.device]
                with (target/f'fresh_CLI_reload_{i}.log').open('x') as log:r=subprocess.run(cmd,cwd=out,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                assert r.returncode==0,'Actual fresh CLI native selected/final model evaluation failed'
                assert read_json(run/'test_metrics.json')==metrics and (run/'test_predictions.csv').read_bytes()==predictions
            assets.verify_bytes();assert Bundle(a.data).qpmilvl.contract==input_contract
            completed.append({'profile':profile,'state':'passed','actual_tasks':4,'native_maximum_epochs':[12]*4,
                'actual_epochs':[r['actual_epochs'] for r in snapshots],'actual_optimizer_updates':baseline['updates'],
                'TRAIN_slides_per_task':32,'native_minibatch':16,'original_patches_per_bag':4097,'feature_dimension':512,
                'full_original_CoCa_constructor':True,'pretrained_text_parameters':110129664,'pool_size':20,'prompt_length':24,'match_size':5,
                'actual_interruption_points':[1,boundary,boundary+1],'all_parameters_gradients_frequency_moments_scheduler_history_rng_exact':True,
                'fresh_publicCLI_reloads':2,'input_contract':input_contract,'gpu_peak_bytes':torch.cuda.max_memory_allocated(),
                'elapsed_seconds':time.monotonic()-started,'synthetic':True,'paper_score_reproduction':False})
            write_json(out/'progress.json',{'state':'running','conditions':completed})
        write_json(out/'validation.json',{'state':'passed','conditions':completed,'source_commit':COMMIT,
            'software':software_identity(),'runtime':runtime_identity(),'input_contract':input_contract,
            'elapsed_seconds':time.monotonic()-began,'input_reuse':'Same immutable actual feature fixture; every training/recovery/evaluation is fresh',
            'timing_is_standalone_efficiency':False,'formal_admission':False,'synthetic':True})
    except BaseException as error:
        write_json(out/'validation.json',{'state':'failed','conditions':completed,'error':repr(error),'formal_admission':False});raise


if __name__=='__main__':main()
