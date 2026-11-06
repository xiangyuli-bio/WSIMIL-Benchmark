"""Unreduced100epoch raw MacNet trajectories, actual1/50/75epoch recovery."""
from pathlib import Path
import argparse,os,sys,json,subprocess,time,hashlib
import torch
from milplus import macnet_driver as driver
from milplus.macnet_data import CONDITIONS
from milplus.macnet_source import SWIN22K_SHA256
from milplus.data import Bundle
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
from milplus.icmil_coupled_training import load,checkpoint_path
from validate_protomil_native import fingerprint
from validate_protomil_pipeline import same


def audited_train(data,method,out,dev,expected=None,resume=False,stop_after=None):
    original=driver.save;events=[]
    def observe(value,path):
        answer=original(value,path)
        if Path(path).name=='last.pt':
            epoch=value['next_epoch'];sha=fingerprint(value)
            if expected is not None:assert expected[epoch]==sha,('all_state_at_actual_epoch',epoch,expected[epoch],sha)
            row={'epoch':epoch,'full_state_sha256':sha,'actual_updates':value['updates']};events.append(row)
            with (Path(path).parent/'every_state_audit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return answer
    driver.save=observe
    try:return driver.train(data,method,out,seed=0,device_name=dev,resume=resume,stop_after=stop_after),events
    finally:driver.save=original


def fresh_evaluate(run,device):
    result=subprocess.run([sys.executable,'-m','milplus','evaluate','--run',str(run),'--device',device],capture_output=True,text=True,env=os.environ.copy())
    assert result.returncode==0,result.stdout+result.stderr
    return (run/'test_predictions.csv').read_bytes(),read_json(run/'test_metrics.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--profile',required=True,choices=list(driver.PROFILES));p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if not a.device.startswith('cuda') or not torch.cuda.is_available():raise ValueError('Actual full original raw CUDA training is required')
    torch.cuda.set_device(torch.device(a.device));probe=torch.empty(1,device=a.device);del probe;torch.cuda.reset_peak_memory_stats(a.device)
    assets=Bundle(a.data).macnet;condition=driver.PROFILES[a.profile]['condition'];rules=CONDITIONS[condition]
    assert assets is not None and assets.condition==condition and assets.value['guidance_scope']=='train_bank'
    native_count={'public':2000,'lscc81':1042,'cscc64':1025}[condition]
    assert all(r['count']==native_count and len(r['guides'])==rules['guides'] and r['shape'][1:]==[96,96,3] for r in assets.rows.values())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);began=time.monotonic();continuous=out/'continuous';resumed=out/'resumed'
    original_tensor=type(assets).tensor;observed=[]
    def no_TEST_fit(self,row):
        assert row['subset']!='test';observed.append((row['slide_id'],row['subset']));return original_tensor(self,row)
    type(assets).tensor=no_TEST_fit
    continuous,events=audited_train(a.data,a.profile,continuous,a.device)
    assert read_json(continuous/'status.json')['state']=='completed'
    expected={v['epoch']:v['full_state_sha256'] for v in events};assert len(expected)==100 and set(expected)==set(range(1,101))
    baseline=load(continuous/'last.pt');expected_updates=(4 if condition=='public' else 3)*100
    assert baseline['next_epoch']==100 and baseline['updates']==expected_updates and baseline['finished']
    assert all(r['actual_group_lrs']==[1e-5,1e-5] for r in baseline['history'])
    resumed,recovered=audited_train(a.data,a.profile,resumed,a.device,expected,stop_after=1)
    assert read_json(resumed/'status.json')['state']=='interrupted' and load(resumed/'last.pt')['next_epoch']==1
    for steps,epoch in [(49,50),(25,75)]:
        _,more=audited_train(a.data,a.profile,resumed,a.device,expected,resume=True,stop_after=steps);recovered+=more
        assert read_json(resumed/'status.json')['state']=='interrupted' and load(resumed/'last.pt')['next_epoch']==epoch
    _,more=audited_train(a.data,a.profile,resumed,a.device,expected,resume=True);recovered+=more
    assert read_json(resumed/'status.json')['state']=='completed' and len(recovered)==100
    same(baseline,load(resumed/'last.pt'),'all_final_source_raw_native_state')
    selected1=read_json(continuous/'selected.json');selected2=read_json(resumed/'selected.json');same(load(selected1['model']),load(selected2['model']),'selected_native_model_and_cached_matrix')
    feature_count=sum(v.numel() for k,v in baseline['model'].items() if k.startswith('feature.') and not k.endswith(('relative_position_index','attn_mask')))
    assert feature_count==27499482 and baseline['model']['initial_adaptive_diagonal'].shape==(768,768)
    type(assets).tensor=original_tensor
    metrics=driver.evaluate(continuous,device_name=a.device);prediction=(continuous/'test_predictions.csv').read_bytes();same(metrics,driver.evaluate(resumed,device_name=a.device),'native_restored_TEST_metrics');assert prediction==(resumed/'test_predictions.csv').read_bytes()
    for _ in range(2):
        actual,score=fresh_evaluate(resumed,a.device);assert actual==prediction;same(score,metrics,'fresh_selected_public_CLI_inference')
    evidence={str(f):digest_file(f) for run in [continuous,resumed] for f in run.rglob('*') if f.is_file()};assert observed
    report={'state':'passed_complete_full_native','profile':a.profile,'actual_epochs':100,'actual_updates':expected_updates,'full_Swin_feature_parameters':feature_count,
        'full_feature_dimension':768,'pretrained_weight_sha256':SWIN22K_SHA256,'actual_guide_count':rules['guides'],'actual_RGB_patches_per_WSI':native_count,
        'all_nonterminal_model_gradients_RMS_scaler_orders_RNG_exact':True,'actual_epoch_interruption':True,'actual_interruption_epochs':[1,50,75],
        'audited_actual_epochs_each_trajectory':100,'fresh_selected_public_CLI_reloads':2,'device_type':'cuda','device_name':torch.cuda.get_device_name(a.device),
        'cuda_peak_bytes':torch.cuda.max_memory_allocated(a.device),'continuous_run':str(continuous),'resumed_run':str(resumed),'evidence_files':evidence,
        'validator_sha256':digest_file(__file__),'software':software_identity(),'synthetic':True,'formal_admission':False,'paper_score_reproduction':False,
        'timing_is_standalone_efficiency':False,'elapsed_seconds':time.monotonic()-began}
    write_json(out/'validation.json',report)

if __name__=='__main__':main()
