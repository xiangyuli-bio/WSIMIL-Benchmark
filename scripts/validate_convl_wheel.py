"""Four complete actual CUDA ConVL precision/cohort source-budget references."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROGRAM = r'''
import gc,sys,time,hashlib
from pathlib import Path
target,out=map(Path,sys.argv[1:3]);case=int(sys.argv[3]);device=sys.argv[4];weights=Path(sys.argv[5])
sys.path.insert(0,str(target))
import torch,milplus
from milplus.convl_demo import create_convl_demo
from milplus.data import Bundle
from milplus.models import build_bundle_model
from milplus.engine import train,evaluate,seed_everything
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
from milplus.convl_precision import validate_scaler_history
assert milplus.__version__=='0.88.0' and torch.cuda.is_available()
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
torch.cuda.set_device(device);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
profile,cohort=[('convl_yaml_fp16_source_repaired','BRCA'),('convl_yaml_fp32_reference','BRCA'),
 ('convl_yaml_fp16_source_repaired','COADREAD'),('convl_yaml_fp32_reference','COADREAD')][case]
out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
data=create_convl_demo(out/'data',weights,patients=20,cohort=cohort,patches=257)
bundle=Bundle(data)
assert bundle.dim==512 and len(bundle.rows.global_patient_id.unique())==20
assert len(bundle.rows.loc[bundle.rows.subset=='train'].global_patient_id.unique())==12
assert set(record['patches'] for record in bundle.convl.meta['records'].values())=={257}
seed_everything(2024);model=build_bundle_model(profile,bundle)
parameters=112353976 if cohort=='BRCA' else 112353984
assert sum(p.numel() for p in model.parameters())==parameters
assert len(model.core.text_encoder.transformer.resblocks)==12
assert model.core.prompt_learner.ctx.shape==(16,768)
assert model.core.fc_concept[0].weight.shape==(256,512) and model.core.attention_net.mha.num_heads==4
markers=['core.prompt_learner.ctx','core.prompt_learner.meta_net.fc1.weight','core.prompt_learner.meta_net.fc2.weight',
 'core.clip_model.text.transformer.resblocks.0.attn.in_proj_weight','core.clip_model.text.transformer.resblocks.11.attn.in_proj_weight',
 'core.fc_concept.0.weight','core.attention_net.mha.in_proj_weight','core.attention_net.attention_c.weight',
 'core.classifier.weight','core.concept_pred_classifier.weight']
initial=model.state_dict();before={name:initial[name].clone() for name in markers}
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
unused=['core.fc_patch.0.weight','core.fc_patch.0.bias','core.clip_model.text.token_embedding.weight','core.clip_model.text.cls_emb']
unchanged={name:tensor_sha(initial[name]) for name in unused}
active_tensors=len(model._active_parameters());all_tensors=len(list(model.core.parameters()))
del model,initial;gc.collect()
run=train(data,profile,out/'run',seed=2024,device=device,config={'threads':2})
assert not (run/'test_predictions.csv').exists()
cfg=read_json(run/'request.json')['config']
assert cfg['epochs']==200 and cfg['patience']==8 and cfg['lr']==.0002 and cfg['weight_decay']==.00001
assert cfg['grad_clip'] is None and cfg['max_instances'] is None and cfg['selection_metric']=='convl_val_nll'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
state=last['model'];extra=state['_extra_state'];history=last['history'];epochs=last['epoch']+1
assert len(history)==extra['completed_epochs']==epochs and 9<=epochs<=200 and extra['optimizer_updates']>0
assert len(extra['train_orders'])==len(extra['update_traces'])==epochs
for order,traces in zip(extra['train_orders'],extra['update_traces']):
 assert len(order)==len(set(order))==12 and len(traces)==6
 assert [trace['patients'] for trace in traces]==[order[i:i+2] for i in range(0,12,2)]
 assert all(trace['normalization']==2 for trace in traces)
validate_scaler_history(profile.endswith('fp16_source_repaired'),extra['update_traces'],extra['scaler'])
assert all(not torch.equal(before[name],state[name]) for name in markers)
assert all(tensor_sha(state[name])==digest for name,digest in unchanged.items())
opt=last['optimizers'][0];assert set(opt)=={'fast_state','slow_state','param_groups','rectification_buffer'}
assert len(opt['fast_state'])==active_tensors and len(opt['slow_state'])==all_tensors and len(opt['rectification_buffer'])==10
assert all(s['step']==extra['optimizer_updates'] for s in opt['fast_state'].values())
assert [g['weight_decay'] for g in opt['param_groups']]==[0.,.00001]
assert all(g['counter']==extra['optimizer_updates']%5 and g['lr']==.0002 for g in opt['param_groups'])
curve=[row['validation']['convl_val_nll'] for row in history]
assert best['epoch']==min(range(epochs),key=lambda i:curve[i])
if epochs<200:assert last['bad']==8
if profile.endswith('fp32_reference'):assert extra['optimizer_updates']==epochs*6 and extra['scaler']=={}
protocol=read_json(run/'survival_protocol.json')
assert protocol['completed_epochs']==epochs and protocol['optimizer_updates']==extra['optimizer_updates']
completed_updates=extra['optimizer_updates']
del last,best,state,extra,opt;gc.collect();torch.cuda.empty_cache()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
gc.collect();torch.cuda.empty_cache()
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
import pandas as pd
frame=pd.read_csv(run/'test_predictions.csv');assert len(frame)==4 and set(frame.convl_patient_patches)<={257,514}
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
write_json(out/'validation.json',{'state':'passed','case':case,'profile':profile,'cohort':cohort,
 'max_epochs':200,'patience':8,'epochs':epochs,'patients':20,'TRAIN_patients':12,'VAL_patients':4,'TEST_patients':4,
 'patches_per_slide':257,'feature_dimension':512,'parameters':parameters,'active_parameters':87574968+(8 if cohort=='COADREAD' else 0),
 'text_layers':12,'context_tokens':16,'gradient_accumulation':2,'optimizer_updates':completed_updates,
 'all_native_mechanism_markers_updated':True,'source_unused_projection_token_embedding_and_CLS_unchanged':True,
 'source_full_fast_slow_and_scaler_state_verified':True,'source_NLL_only_selection_and_stopping_verified':True,
 'selected_checkpoint_reloaded_twice':True,'prediction_sha256':prediction,'metrics':metrics,'software':software_identity(),
 'device_type':'cuda','device':torch.cuda.get_device_name(device),'synthetic':True,'formal_admission':False,
 'wall_seconds':time.monotonic()-started,'scope':'Full original112M/native512/12layer/16prompt/CGA/3objectives/LookaheadRAdam/200max/patience8 original FP16 or explicit FP32, all patient slides/257patches retained; no source budget or width reduction'})
'''


def retain_completed_weights(case):
    """Explicit verification-only mode, after actual full training and double reload."""
    import hashlib
    case = Path(case).resolve(); run = case/'run'
    receipt = json.loads((case/'validation.json').read_text())
    assert receipt['state'] == 'passed' and receipt['selected_checkpoint_reloaded_twice']
    assert not run.is_symlink() and run.parent == case
    request = json.loads((run/'request.json').read_text()); status = json.loads((run/'status.json').read_text())
    assert request['synthetic'] is True and status['state'] == 'completed'
    assert (run/'test_predictions.csv').is_file() and (run/'test_metrics.json').is_file()
    files = []
    for path in sorted(run.rglob('*.pt')):
        assert not path.is_symlink()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4*1024**2), b''):
                digest.update(block)
        files.append({'path': str(path.relative_to(case)), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size})
    (case/'weight_receipts.json').write_text(json.dumps({'mode':'metadata_only_after_full_budget_and_double_reload',
        'files':files,'synthetic':True,'native_inputs_untouched':True,
        'limitation':'Synthetic verification weights intentionally not retained after actual double reload; reproduce the frozen native run for further inference'},indent=2)+'\n')
    for item in files:
        (case/item['path']).unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['wheel', 'source-tree', 'output', 'weights']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--devices', default='cuda:0')
    parser.add_argument('--workers-per-device', type=int, choices=[1, 2], default=1)
    parser.add_argument('--metadata-only', action='store_true', help='Only after real full-budget train and two actual reloads, retain complete weight hashes instead of temporary synthetic weights')
    args = parser.parse_args(); devices = args.devices.split(',')
    if len(devices) != len(set(devices)) or any(not re.fullmatch(r'cuda:\d+', device) for device in devices):
        raise ValueError('Choose unique actual allocated CUDA devices')
    wheel, source, out, weights = map(lambda p: Path(p).resolve(), [args.wheel, args.source_tree, args.output, args.weights])
    out.mkdir(parents=True, exist_ok=False); target = out/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        if any(not (target/name).resolve().is_relative_to(target) for name in archive.namelist()):
            raise ValueError('Unsafe wheel member')
        archive.extractall(target)
    cases = [{'case': i, 'state': 'pending'} for i in range(4)]
    slots = [device for device in devices for _ in range(args.workers_per_device)]
    active, handles, cursor = {}, [], 0
    process_state = {'state': 'running', 'cases': cases, 'formal_admission': False}
    def save():
        temporary = out/'process.tmp'
        temporary.write_text(json.dumps(process_state, indent=2)+'\n')
        temporary.replace(out/'process.json')
    try:
        while cursor < 4 or active:
            for slot, device in enumerate(slots):
                if slot in active or cursor >= 4:
                    continue
                i = cursor; cursor += 1; log = (out/f'case_{i}.log').open('x'); handles.append(log)
                child = subprocess.Popen([sys.executable,'-c',PROGRAM,str(target),str(out/f'case_{i}'),str(i),device,str(weights)],
                    cwd=source,env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT)
                cases[i].update(state='running',pid=child.pid,device=device); active[slot]=(i,child); save()
            for slot, (i, child) in list(active.items()):
                code = child.poll()
                if code is not None:
                    cases[i].update(state='passed' if code==0 else 'failed',returncode=code); del active[slot]; save()
                    if code == 0 and args.metadata_only:
                        retain_completed_weights(out/f'case_{i}')
            time.sleep(1)
        if any(row['state'] != 'passed' for row in cases):
            raise RuntimeError('ConVL full native reference failed; preserve actual case and budget')
        reports = [json.loads((out/f'case_{i}/validation.json').read_text()) for i in range(4)]
        assert all(row['state']=='passed' and row['software']==reports[0]['software'] for row in reports)
        (out/'validation.json').write_text(json.dumps({'state':'passed','profiles':reports,'software':reports[0]['software'],
            'synthetic':True,'formal_admission':False,'weight_retention':'metadata_only_after_actual_full_budget_and_double_reload' if args.metadata_only else 'all',
            'scope':'All four full native CUDA source-budget conditions; current whole-library and final release gates remain separate'},indent=2)+'\n')
        process_state['state'] = 'passed'
        save()
    except BaseException as error:
        process_state.update(state='failed', error=repr(error))
        for i, child in active.values():
            if child.poll() is None:
                child.terminate()
        for i, child in active.values():
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            cases[i].update(state='stopped_after_validator_failure', returncode=child.returncode)
        save()
        raise
    finally:
        for stream in handles:
            stream.close()


if __name__ == '__main__':
    main()
