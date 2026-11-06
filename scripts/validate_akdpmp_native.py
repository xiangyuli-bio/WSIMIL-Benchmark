"""Complete author budgets, full10K TCGA bags and exact source/wheel recovery."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from milplus.akdpmp_driver import train,evaluate,PROFILES
from milplus.akdpmp_demo import create_demo
from milplus.akdpmp_source import COMMIT,TOPK_COMMIT
from milplus.icmil_coupled_training import load
from milplus.io import write_json,read_json
from milplus.provenance import software_identity


def same(a,b,path='state'):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError('AKD-PMP exact recovery differs at '+path)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a:same(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'.'+str(i))
    else:assert a==b,(path,a,b)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if torch.device(args.device).type!='cuda' or not torch.cuda.is_available():raise ValueError('Native complete-budget validation requires an actual allocated CUDA device')
    data={preset:create_demo(out/('data_'+preset),preset=preset,patches=10001 if preset=='tcga' else 2001) for preset in ['tcga','skin']}
    rows=[];began=time.monotonic()
    for profile,spec in PROFILES.items():
        target=out/profile;target.mkdir();started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        continuous=train(data[spec['preset']],profile,target/'continuous',seed=1,device=args.device)
        baseline=load(continuous/'last.pt');snapshots=baseline['snapshots'];budgets=[50]*3 if spec['preset']=='tcga' else ([40,50,50] if spec['kind']=='clam' else [20,40,25])
        assert len(snapshots)==3 and [r['maximum_epochs'] for r in snapshots]==budgets
        assert all(r['actual_epochs']>=min(b,b//2+2) and r['actual_epochs']<=b for r,b in zip(snapshots,budgets))
        first_boundary=snapshots[0]['actual_epochs']
        resumed=train(data[spec['preset']],profile,target/'resumed',seed=1,device=args.device,stop_after=1)
        assert read_json(resumed/'status.json')['state']=='interrupted'
        train(data[spec['preset']],profile,resumed,seed=1,device=args.device,resume=True,stop_after=first_boundary-1)
        boundary=load(resumed/'last.pt');assert boundary['next_task']==1 and boundary['next_epoch']==0
        train(data[spec['preset']],profile,resumed,seed=1,device=args.device,resume=True,stop_after=1)
        later=load(resumed/'last.pt');assert later['next_task']==1 and later['next_epoch']==1
        assert later['teacher'] and later['teacher_gradients'] and later['memory']
        train(data[spec['preset']],profile,resumed,seed=1,device=args.device,resume=True)
        actual=load(resumed/'last.pt')
        for key in ['model','modes','gradients','next_task','next_epoch','optimizer','teacher','teacher_gradients','early',
                    'memory','train_generator','val_generator','history','updates','rng']:same(baseline[key],actual[key],key)
        for left,right in zip(baseline['snapshots'],actual['snapshots']):same(load(left['path']),load(right['path']))
        metrics=evaluate(continuous,'test',args.device);predictions=(continuous/'test_predictions.csv').read_bytes()
        for index in range(2):
            command=[sys.executable,'-m','milplus','evaluate','--run',str(continuous),'--subset','test','--device',args.device]
            result=subprocess.run(command,capture_output=True,text=True,env=os.environ.copy());(target/f'fresh_cli_{index}.log').write_text(result.stdout+result.stderr)
            assert result.returncode==0,result.stderr
            assert read_json(continuous/'test_metrics.json')==metrics and (continuous/'test_predictions.csv').read_bytes()==predictions
        rows.append({'profile':profile,'state':'passed','actual_tasks':3,'native_maximum_epochs':budgets,
            'actual_epochs':[r['actual_epochs'] for r in snapshots],'actual_optimizer_updates':baseline['updates'],
            'actual_patches_per_original_bag':10001 if spec['preset']=='tcga' else 2001,
            'feature_dimension':1024 if spec['preset']=='tcga' else 512,'hidden':512,
            'all_original_selected_teachers_memory_values_gradients_moments_rng_history_exact':True,
            'actual_interruption_points':[1,first_boundary,first_boundary+1],'two_fresh_publicCLI_selected_reloads':True,
            'memory_samples':snapshots[-1]['memory_samples'],'memory_patches':snapshots[-1]['memory_patches'],
            'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-started,
            'synthetic':True,'paper_score_reproduction':False})
        write_json(out/'progress.json',{'state':'running','completed':rows})
    write_json(out/'validation.json',{'state':'passed','conditions':rows,'software':software_identity(),
        'source_commit':COMMIT,'actual_smooth_SVM_commit':TOPK_COMMIT,'elapsed_seconds':time.monotonic()-began,
        'formal_admission':False,'synthetic':True,'timing_scope':'engineering validation on allocated cluster GPU; not standalone paper efficiency'})
    print(json.dumps({'state':'passed','receipt':str(out/'validation.json')}),flush=True)


if __name__=='__main__':main()
