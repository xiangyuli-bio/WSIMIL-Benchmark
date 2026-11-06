"""Original full-width four-task software lifecycle; no native12epoch claim."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from milplus.qpmilvl_demo import create_demo
from milplus.qpmilvl_driver import train,evaluate
from milplus.icmil_coupled_training import load
from milplus.io import write_json,read_json


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
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);steps=[]
    try:
        data=create_demo(out/'data',a.weights,a.sha256);cfg={'epochs':2,'threads':2}
        run=train(data,out/'continuous',config=cfg,device=a.device);baseline=load(run/'last.pt')
        assert baseline['next_task']==4 and len(baseline['history'])==8 and baseline['updates']==8
        assert all(len(row['train_order'])==16 and row['dropped_native_incomplete_tail_slides']==0 for row in baseline['history'])
        result=evaluate(run,device=a.device);pred=(run/'test_predictions.csv').read_bytes();assert len(result['matrix'])==4
        assert evaluate(run,device=a.device)==result and (run/'test_predictions.csv').read_bytes()==pred
        steps.append('full4task_realCONCH110Mtext16bag_native_objectives_and_frequency_transitions')
        for point in [1,2,3]:
            resumed=train(data,out/('resumed'+str(point)),config=cfg,device=a.device,stop_after=point)
            assert read_json(resumed/'status.json')['state']=='interrupted'
            train(data,resumed,config=cfg,device=a.device,resume=True);actual=load(resumed/'last.pt')
            for key in ['pools','gradients','next_task','next_epoch','optimizer','scheduler','early','frequency','val_frequency','history','updates','generators','rng']:
                same(baseline[key],actual[key],key)
            for left,right in zip(baseline['snapshots'],actual['snapshots']):same(load(left['path']),load(right['path']))
            assert evaluate(resumed,device=a.device)==result and (resumed/'test_predictions.csv').read_bytes()==pred
            steps.append('actual_interrupt_'+str(point)+'_all_parameters_gradients_frequency_moments_scheduler_rng_exact')
        write_json(out/'validation.json',{'state':'passed_software_lifecycle_only','checks':steps,'synthetic':True,
            'tasks':4,'software_epochs_per_task':2,'actual_updates':8,'actual_minibatch':16,'real_pretrained_full_CONCH_constructor':True,
            'full_native12epoch_budget_passed':False,'public_CLI_passed':False,'included_in_method_count':False,'formal_admission':False})
    except BaseException as error:
        write_json(out/'validation.json',{'state':'failed_software_lifecycle_only','checks':steps,'error':repr(error),'included_in_method_count':False,'formal_admission':False});raise


if __name__=='__main__':main()
