"""Complete original-sized components through all stages, actual review and resume."""
import argparse,copy,json,time
from pathlib import Path
import torch,numpy as np
from milplus.protomil_demo import create_demo
from milplus.protomil_driver import train,evaluate,PROFILES
from milplus.icmil_coupled_training import load
from milplus.data import Bundle
from milplus.protomil_data import ProtoMILInputs
from milplus.io import read_json,write_json


def same(a,b,path='state'):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
    elif hasattr(a,'__dict__'):same(a.__dict__,b.__dict__,path)
    else:assert a==b,(path,a,b)


def supply_synthetic_review(run):
    request=read_json(run/'review_request.json');bank=read_json(run/'concepts/concept_bank.json');concept=bank['active_concepts'][0]
    value={'schema_version':1,'sae_state_sha256':request['sae_state_sha256'],'concept_bank_sha256':request['concept_bank_sha256'],
        'reviewer':'synthetic_validation_only','provenance':'Explicit test decision only; no clinician/pathologist review or clinical interpretation claimed',
        'removed_concepts':[concept],'concept_names':{str(concept):'synthetic test removal, no clinical meaning'}}
    write_json(request['review_file'],value)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--profile',action='append');a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);began=time.monotonic();data=create_demo(root/'input');conditions=[]
    cfg={'sae_epochs':1,'mil_epochs':2,'probe_per_class':5,'threads':2,'cache_root':str(root/'shared_lossless_cache'),'verify_existing_cache':True}
    native=ProtoMILInputs.features;observed=[]
    def guarded(self,row):
        assert row['subset']!='test';observed.append((row['slide_id'],row['subset']));return native(self,row)
    for profile in a.profile or list(PROFILES):
        ProtoMILInputs.features=guarded
        out=train(data,root/(profile+'_continuous'),config=cfg,device=a.device,method=profile)
        if read_json(out/'status.json')['state']=='awaiting_concept_review':
            assert PROFILES[profile]['intervened'] and not (out/'MIL').exists();supply_synthetic_review(out)
            train(data,out,config=cfg,device=a.device,method=profile,resume=True)
        assert read_json(out/'status.json')['state']=='completed'
        baseline_sae=load(out/'SAE/last.pt');baseline_mil=load(out/'MIL/last.pt')
        actual_pieces=baseline_sae['snapshot']['state']['next_piece'];assert actual_pieces==6 and baseline_mil['updates']==12
        resumed=train(data,root/(profile+'_resumed'),config=cfg,device=a.device,method=profile,stop_after=1)
        assert read_json(resumed/'status.json')['state']=='interrupted'
        train(data,resumed,config=cfg,device=a.device,method=profile,resume=True,stop_after=actual_pieces-1)
        assert read_json(resumed/'status.json')['state']=='interrupted' and load(resumed/'SAE/last.pt')['finished']
        train(data,resumed,config=cfg,device=a.device,method=profile,resume=True,stop_after=2)
        if read_json(resumed/'status.json')['state']=='awaiting_concept_review':
            supply_synthetic_review(resumed);train(data,resumed,config=cfg,device=a.device,method=profile,resume=True,stop_after=1)
        assert read_json(resumed/'status.json')['state']=='interrupted' and load(resumed/'MIL/last.pt')['next_epoch']==1
        train(data,resumed,config=cfg,device=a.device,method=profile,resume=True)
        assert read_json(resumed/'status.json')['state']=='completed'
        same(baseline_sae,load(resumed/'SAE/last.pt'),'complete_original_SAE_phase_state')
        same(baseline_mil,load(resumed/'MIL/last.pt'),'complete_original_MIL_phase_state')
        s1=read_json(out/'selected.json');s2=read_json(resumed/'selected.json')
        same(load(s1['SAE']),load(s2['SAE']),'selected_full_SAE');same(load(s1['MIL']),load(s2['MIL']),'selected_full_MIL')
        ProtoMILInputs.features=native
        metrics=evaluate(out,device=a.device);prediction=(out/'test_predictions.csv').read_bytes()
        same(metrics,evaluate(resumed,device=a.device),'heldout_metrics');assert prediction==(resumed/'test_predictions.csv').read_bytes()
        same(metrics,evaluate(out,device=a.device),'reloaded_metrics');assert prediction==(out/'test_predictions.csv').read_bytes()
        conditions.append({'profile':profile,'state':'passed','actual_SAE_pieces':actual_pieces,'actual_MIL_updates':12,
            'source_input512_concepts2048_batch4096':True,'actual_normal_tumor_probes_per_class':5,'actual_three_interruptions':['SAE_piece1','SAE_end','MIL_epoch1'],
            'full_original_weights_gradients_optimizer_resampler_metrics_orders_RNG_exact':True,'selected_models_and_reloaded_predictions_exact':True,
            'actual_bound_review_mask_applied':PROFILES[profile]['intervened'],'synthetic':True,'full_native_epoch_or_probe_budget_claimed':False})
        write_json(root/'progress.json',{'state':'running','conditions':conditions})
    assert observed;write_json(root/'validation.json',{'state':'passed_complete_pipeline_software','conditions':conditions,'elapsed_seconds':time.monotonic()-began,
        'included_in_method_count':False,'formal_admission':False,'paper_score_reproduction':False})

if __name__=='__main__':main()
