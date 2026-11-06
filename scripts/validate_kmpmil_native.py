"""Full original native four-task KMP; actual4097patches/33TRAIN/20epoch budgets."""
import argparse,json,time,copy,subprocess,sys,os,hashlib
from milplus import kmpmil_driver as driver
from pathlib import Path
import numpy as np
import torch
from milplus.kmpmil_driver import train,evaluate,PROFILES
from kmpmil_native_fixture import create_demo
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


def exact_digest(value):
    """Hash every byte, tensor dtype/shape and exact scalar; no truncated repr."""
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            x=v.detach().cpu().contiguous();h.update(b'tensor'+str(x.dtype).encode()+repr(tuple(x.shape)).encode()+x.numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(b'array'+str(v.dtype).encode()+repr(v.shape).encode()+v.tobytes())
        elif isinstance(v,dict):
            h.update(b'dict'+str(len(v)).encode())
            for k in sorted(v,key=lambda k:(type(k).__name__,str(k))):visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode()+str(len(v)).encode())
            for x in v:visit(x)
        else:h.update(type(v).__name__.encode()+repr(v).encode())
    visit(value);return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--weights',required=True);p.add_argument('--sha256',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=20);p.add_argument('--profile',action='append');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();assert a.epochs==20,'Native original per-task maximum is20; no reduced native budget'
    data=create_demo(out/'data',a.weights,a.sha256,patches=4097,train_slides_per_task=33,val_per_class=3)
    cfg={'epochs':a.epochs,'threads':2};conditions=[];native=KMPInputs.features;observed=[]
    def guarded(self,row):
        assert row['subset']!='test';observed.append((row['subset'],row['target']));return native(self,row)
    native_save=driver.save
    checkpoint_keys=['model','gradients','tc','banks','next_task','next_epoch','optimizer','scheduler','early','history','updates','generators','rng']
    for profile in a.profile or list(PROFILES):
        observed_epoch_states={}
        def observed_save(value,path):
            if Path(path).name=='last.pt':observed_epoch_states[len(value['history'])]={key:exact_digest(value[key]) for key in checkpoint_keys}
            return native_save(value,path)
        driver.save=observed_save
        KMPInputs.features=guarded
        whole=train(data,out/(profile+'_whole'),config=cfg,device=a.device,method=profile);baseline=load(whole/'last.pt');baseline_epoch_states=copy.deepcopy(observed_epoch_states)
        assert baseline['next_task']==4 and baseline['next_epoch']==0 and len(baseline['banks']['keys'])==4
        assert [len(v) for v in baseline['banks']['keys']]==[10]*4
        assert all(r['native_minibatch_sizes']==[16,16,1] and len(r['train_order'])==33 for r in baseline['history'])
        assert all(11<=r['actual_epochs']<=20 for r in baseline['snapshots'])
        assert all(0<r['selected_epoch']<=r['actual_epochs'] for r in baseline['snapshots'])
        boundary=baseline['snapshots'][0]['actual_epochs'];points=sorted({1,boundary,boundary+1})
        for point in points:
            observed_epoch_states.clear()
            run=train(data,out/(profile+'_resume'+str(point)),config=cfg,device=a.device,method=profile,stop_after=point)
            assert json.loads((run/'status.json').read_text())['state']=='interrupted'
            train(data,run,config=cfg,device=a.device,method=profile,resume=True);actual=load(run/'last.pt')
            for key in ['model','gradients','tc','banks','next_task','next_epoch','optimizer','scheduler','early','history','updates','generators','rng']:same(baseline[key],actual[key],key)
            for left,right in zip(baseline['snapshots'],actual['snapshots']):same(load(left['path']),load(right['path']),'selected')
            same(baseline_epoch_states,observed_epoch_states,'every_actual_epoch_checkpoint_including_nonterminal_Adam_moments_scheduler_and_RNG')
        KMPInputs.features=native
        first=evaluate(whole,device=a.device);pred=(whole/'test_predictions.csv').read_bytes();task_pred=(whole/'test_task_cl_predictions.csv').read_bytes()
        for reload in range(2):
            proc=subprocess.run([sys.executable,'-m','milplus','evaluate','--run',str(whole),'--device',a.device],capture_output=True,text=True,env=os.environ.copy())
            assert proc.returncode==0,proc.stdout+proc.stderr
            same(first,json.loads((whole/'test_metrics.json').read_text()),'fresh_CLI_metrics')
            assert pred==(whole/'test_predictions.csv').read_bytes() and task_pred==(whole/'test_task_cl_predictions.csv').read_bytes()
        conditions.append({'profile':profile,'state':'passed','epochs':len(baseline['history']),'updates':baseline['updates'],'actual_interruption_points':points,
            'all_learned_parameters_gradients_banks_TC_optimizer_scheduler_history_generators_RNG_exact':True,
            'every_actual_epoch_state_verified_including_nonterminal_Adam_moments':True,'epoch_states':baseline_epoch_states,
            'full_original_text_parameters':110129664,'native_maximum_epochs':[20]*4,'actual_epochs':[r['actual_epochs'] for r in baseline['snapshots']],
            'selected_epochs':[r['selected_epoch'] for r in baseline['snapshots']],'native_TRAIN_per_task':33,'native_patches_per_bag':4097,'native_update_windows':[16,16,1],'class_CL_and_task_CL_reloads':2,'synthetic':True,'paper_scores_reproduced':False})
        write_json(out/'progress.json',{'conditions':conditions,'state':'running'})
    driver.save=native_save
    assert observed
    write_json(out/'validation.json',{'state':'passed_full_native','conditions':conditions,'elapsed_seconds':time.monotonic()-start,'formal_admission':False,'included_in_count':False})

if __name__=='__main__':main()
