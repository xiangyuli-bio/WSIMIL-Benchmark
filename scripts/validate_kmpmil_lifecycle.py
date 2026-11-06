"""Full original four-task KMP software lifecycle; synthetic, not paper scores."""
import argparse,json,time,copy
from pathlib import Path
import numpy as np
import torch
from milplus.kmpmil_driver import train,evaluate,PROFILES
from milplus.kmpmil_demo import create_demo
from milplus.kmpmil_data import KMPInputs
from milplus.icmil_coupled_training import load
from milplus.io import write_json


def same(a,b,path='root'):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b),path
        for k,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(k))
    elif isinstance(a,float) and np.isnan(a):assert np.isnan(b),path
    else:assert a==b,(path,a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--weights',required=True);p.add_argument('--sha256',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=3);p.add_argument('--profile',action='append');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();data=create_demo(out/'data',a.weights,a.sha256)
    cfg={'epochs':a.epochs,'threads':2};conditions=[];native=KMPInputs.features;observed=[]
    def guarded(self,row):
        assert row['subset']!='test';observed.append((row['subset'],row['target']));return native(self,row)
    for profile in a.profile or list(PROFILES):
        KMPInputs.features=guarded
        whole=train(data,out/(profile+'_whole'),config=cfg,device=a.device,method=profile);baseline=load(whole/'last.pt')
        assert baseline['next_task']==4 and baseline['next_epoch']==0 and len(baseline['banks']['keys'])==4
        assert [len(v) for v in baseline['banks']['keys']]==[10]*4
        assert all(len(r['native_minibatch_sizes'])==2 and r['native_minibatch_sizes'][0]==16 and 0<r['native_minibatch_sizes'][1]<16 for r in baseline['history'])
        assert all(0<r['selected_epoch']<=r['actual_epochs'] for r in baseline['snapshots'])
        boundary=baseline['snapshots'][0]['actual_epochs'];points=sorted({1,boundary,boundary+1})
        for point in points:
            run=train(data,out/(profile+'_resume'+str(point)),config=cfg,device=a.device,method=profile,stop_after=point)
            assert json.loads((run/'status.json').read_text())['state']=='interrupted'
            train(data,run,config=cfg,device=a.device,method=profile,resume=True);actual=load(run/'last.pt')
            for key in ['model','gradients','tc','banks','next_task','next_epoch','optimizer','scheduler','early','history','updates','generators','rng']:same(baseline[key],actual[key],key)
            for left,right in zip(baseline['snapshots'],actual['snapshots']):same(load(left['path']),load(right['path']),'selected')
        KMPInputs.features=native
        first=evaluate(whole,device=a.device);pred=(whole/'test_predictions.csv').read_bytes();task_pred=(whole/'test_task_cl_predictions.csv').read_bytes()
        second=evaluate(whole,device=a.device);same(first,second,'metrics');assert pred==(whole/'test_predictions.csv').read_bytes() and task_pred==(whole/'test_task_cl_predictions.csv').read_bytes()
        conditions.append({'profile':profile,'state':'passed','epochs':len(baseline['history']),'updates':baseline['updates'],'actual_interruption_points':points,
            'all_learned_parameters_gradients_banks_TC_optimizer_scheduler_history_generators_RNG_exact':True,
            'full_original_text_parameters':110129664,'class_CL_and_task_CL_reloads':2,'synthetic':True,'paper_scores_reproduced':False})
        write_json(out/'progress.json',{'conditions':conditions,'state':'running'})
    assert observed
    write_json(out/'validation.json',{'state':'passed_complete_lifecycle','conditions':conditions,'elapsed_seconds':time.monotonic()-start,'formal_admission':False,'included_in_count':False})

if __name__=='__main__':main()
