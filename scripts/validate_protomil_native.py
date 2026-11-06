"""Unreduced author/paper budgets, every-state audit, source/wheel native lifecycle.

This uses synthetic512D numeric features and explicit synthetic region RGB/masks.
It validates the complete original-sized implementation, never clinical scores.
"""
import argparse,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from milplus.protomil_driver import PROFILES,Progress,train,evaluate
from milplus.protomil_data import ProtoMILInputs
from milplus.protomil_stages import read_review
from milplus.icmil_coupled_training import load,checkpoint_path
from milplus.io import read_json,write_json,digest_file
from milplus.data import Bundle
from milplus.provenance import software_identity
from validate_protomil_pipeline import same,supply_synthetic_review


def fingerprint(value):
    h=hashlib.sha256()
    def visit(x):
        h.update((type(x).__module__+'.'+type(x).__qualname__+':').encode())
        if isinstance(x,torch.Tensor):
            v=x.detach().cpu().contiguous();h.update(str(v.dtype).encode());h.update(repr(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str(x.dtype).encode());h.update(repr(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k,v in x.items():visit(k);visit(v)
        elif isinstance(x,(tuple,list)):
            h.update(str(len(x)).encode())
            for v in x:visit(v)
        elif isinstance(x,(str,int,float,bool,type(None))):h.update(repr(x).encode())
        elif isinstance(x,np.generic):visit(x.item())
        elif isinstance(x,torch.dtype):h.update(str(x).encode())
        elif isinstance(x,Path):h.update(str(x).encode())
        elif hasattr(x,'__dict__'):visit(x.__dict__)
        else:raise TypeError('Native all-state digest refuses unsupported value '+str(type(x)))
        h.update(b';')
    visit(value);return h.hexdigest()


def audit_training(data,out,profile,cache,device,baseline,stop_after=None,resume=False):
    original=Progress.record;seen=[]
    def record(self,phase,completed,finished):
        if phase in {'SAE','MIL'}:
            state=load(self.root/phase/'last.pt');sha=fingerprint(state);key=(phase,completed)
            if baseline is not None:assert baseline[key]==sha,('nonterminal_full_state',key,baseline[key],sha)
            item={'phase':phase,'completed':completed,'finished':finished,'all_state_sha256':sha};seen.append(item)
            with (self.root/'every_state_audit.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
        return original(self,phase,completed,finished)
    Progress.record=record
    try:
        return train(data,out,method=profile,device=device,config={'cache_root':str(cache),'verify_existing_cache':True},stop_after=stop_after,resume=resume),seen
    finally:Progress.record=original


def fresh_evaluate(out,device):
    p=subprocess.run([sys.executable,'-m','milplus','evaluate','--run',str(out),'--device',device],capture_output=True,text=True,env=os.environ.copy())
    assert p.returncode==0,p.stdout+p.stderr
    return (out/'test_predictions.csv').read_bytes(),read_json(out/'test_metrics.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--data',required=True);p.add_argument('--cache-root',required=True)
    p.add_argument('--profile',action='append',choices=list(PROFILES));p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if not a.device.startswith('cuda') or not torch.cuda.is_available():raise ValueError('Actual CUDA required; no CPU/precision/size fallback')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);began=time.monotonic();cache=Path(a.cache_root);conditions=[]
    inputs=Bundle(a.data).protomil;assert inputs is not None
    assert len(inputs.population)==8 and all(r['patches']==4097 for r in inputs.population)
    assert len(inputs.probes)==20000 and all(sum(r['region_label']==c for r in inputs.probes)==10000 for c in [0,1])
    original_features=ProtoMILInputs.features;observed=[]
    def no_heldout(self,row):
        assert row['subset']!='test';observed.append((row['slide_id'],row['subset']));return original_features(self,row)
    for profile in a.profile or list(PROFILES):
        native=PROFILES[profile];baseline_out=out/(profile+'_continuous');resumed=out/(profile+'_resumed');torch.cuda.reset_peak_memory_stats(a.device)
        ProtoMILInputs.features=no_heldout
        baseline_out,events=audit_training(a.data,baseline_out,profile,cache,a.device,None)
        if read_json(baseline_out/'status.json')['state']=='awaiting_concept_review':
            assert native['intervened'] and not (baseline_out/'MIL').exists();supply_synthetic_review(baseline_out)
            _,more=audit_training(a.data,baseline_out,profile,cache,a.device,None,resume=True);events+=more
        assert read_json(baseline_out/'status.json')['state']=='completed'
        expected={(v['phase'],v['completed']):v['all_state_sha256'] for v in events}
        sae_pieces=6*native['sae_epochs'];mil_epochs=native['mil_epochs']
        assert len(expected)==sae_pieces+mil_epochs and set(expected)=={('SAE',i) for i in range(1,sae_pieces+1)}|{('MIL',i) for i in range(1,mil_epochs+1)}
        baseline_sae=load(baseline_out/'SAE/last.pt');baseline_mil=load(baseline_out/'MIL/last.pt')
        assert baseline_sae['finished'] and baseline_sae['snapshot']['reported_activations']==sae_pieces*8192
        assert baseline_sae['snapshot']['state']['next_piece']==sae_pieces and baseline_mil['next_epoch']==mil_epochs
        assert baseline_mil['updates']==6*mil_epochs
        bank=read_json(baseline_out/'concepts/concept_bank.json');assert bank['per_class']==10000 and len(bank['probes'])==20000
        assert all(len(v)==10 for v in bank['top10_indices'].values())
        resumed,steps=audit_training(a.data,resumed,profile,cache,a.device,expected,stop_after=1)
        assert read_json(resumed/'status.json')['state']=='interrupted'
        _,more=audit_training(a.data,resumed,profile,cache,a.device,expected,resume=True,stop_after=sae_pieces-1);steps+=more
        assert read_json(resumed/'status.json')['state']=='interrupted' and load(resumed/'SAE/last.pt')['finished']
        # One concept-stage event plus one real MIL epoch. In the released
        # default1epoch condition this is its genuine final epoch, not a
        # fabricated additional optimizer step; paper200 has a nonterminal one.
        _,more=audit_training(a.data,resumed,profile,cache,a.device,expected,resume=True,stop_after=2);steps+=more
        if read_json(resumed/'status.json')['state']=='awaiting_concept_review':
            supply_synthetic_review(resumed)
            _,more=audit_training(a.data,resumed,profile,cache,a.device,expected,resume=True,stop_after=1);steps+=more
        assert read_json(resumed/'status.json')['state']=='interrupted' and load(resumed/'MIL/last.pt')['next_epoch']==1
        _,more=audit_training(a.data,resumed,profile,cache,a.device,expected,resume=True);steps+=more
        assert read_json(resumed/'status.json')['state']=='completed'
        assert len(steps)==len(events) and {(v['phase'],v['completed']) for v in steps}==set(expected)
        same(baseline_sae,load(resumed/'SAE/last.pt'),'full_native_final_SAE');same(baseline_mil,load(resumed/'MIL/last.pt'),'full_native_final_MIL')
        # Verify every actual original periodic checkpoint payload, not just
        # its number or terminal model. No native event is omitted/pruned.
        checkpoints={f.name:digest_file(f) for f in (baseline_out/'SAE/source_checkpoints').glob('*.pt')}
        restored={f.name:digest_file(f) for f in (resumed/'SAE/source_checkpoints').glob('*.pt')}
        assert checkpoints==restored and len(checkpoints)>1
        ProtoMILInputs.features=original_features
        direct=evaluate(baseline_out,device=a.device);prediction=(baseline_out/'test_predictions.csv').read_bytes()
        same(direct,evaluate(resumed,device=a.device),'full_native_restored_TEST_metrics');assert (resumed/'test_predictions.csv').read_bytes()==prediction
        for _ in range(2):
            fresh,metrics=fresh_evaluate(resumed,a.device);assert fresh==prediction;same(direct,metrics,'fresh_process_selected_metrics')
        condition={'profile':profile,'state':'passed','actual_native_SAE_epochs':native['sae_epochs'],'actual_SAE_pieces':sae_pieces,
            'actual_SAE_optimizer_updates':sae_pieces*2,'actual_MIL_epochs':mil_epochs,'actual_MIL_updates':6*mil_epochs,
            'normal_tumor_probes_each':10000,'patches_per_WSI':4097,'input_dimension':512,'concept_dimension':2048,'SAE_batch_size':4096,
            'all_nonterminal_model_gradients_optimizer_resampler_metrics_orders_RNG_exact':True,'audited_states_each_run':len(expected),
            'actual_interruptions':['SAE_piece1','SAE_finished','MIL_epoch1'],'original_periodic_SAE_checkpoints_each_run':checkpoints,
            'fresh_selected_public_CLI_reloads':2,'actual_concept_review_mask_applied':native['intervened'],'selection':native['selection'],
            'cuda_peak_bytes':torch.cuda.max_memory_allocated(a.device),'synthetic':True,'paper_score_reproduction':False,'formal_admission':False}
        conditions.append(condition);write_json(out/'progress.json',{'state':'running','conditions':conditions})
    assert observed
    write_json(out/'validation.json',{'state':'passed_complete_full_native','conditions':conditions,'software':software_identity(),
        'actual_device':torch.cuda.get_device_name(a.device),'elapsed_seconds':time.monotonic()-began,'full_native_budgets_preserved':True,
        'formal_admission':False,'paper_score_reproduction':False,'synthetic':True})

if __name__=='__main__':main()
