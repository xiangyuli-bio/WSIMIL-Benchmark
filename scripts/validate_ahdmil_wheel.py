"""Eight independent-wheel complete DMIN200 + AHDMIL-joint200-max budgets.

Device workers only parallelize the eight fixed cases. Every case, the complete
source budget, matched synthetic64-node/RGB16 inputs and all losses
remain mandatory. These are synthetic engineering references, not paper scores.
"""
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
import gc,json,sys,time
from pathlib import Path
target,out=map(Path,sys.argv[1:3]);case=int(sys.argv[3]);device=sys.argv[4]
sys.path.insert(0,str(target))
import torch,milplus
import pandas as pd
from milplus.data import Bundle
from milplus.engine import train,evaluate,seed_everything
from milplus.models import build_bundle_model
from milplus.hdmil import tensor_state_hash
from milplus.ahdmil_source import RECIPES
from milplus.hdmil_demo import create_hdmil_demo
from milplus.io import write_json,read_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.73.0' and torch.cuda.is_available()
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
torch.cuda.set_device(device);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
profile=list(RECIPES)[case//2];dimension=[1024,1536][case%2];classes=RECIPES[profile]['classes']
out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
data=create_hdmil_demo(out/'data',patients=10*classes,dimension=dimension,classes=classes,patches=64)
bundle=Bundle(data);n=len(bundle.records('train'));assert bundle.config['synthetic'] is True
assert all(bundle.hdmil.inventory(row)['nodes']==64 for row in bundle.rows.to_dict('records'))
seed_everything(2026);model=build_bundle_model(profile,bundle)
assert model.pretrain.attention_net[0].weight.shape==(512,dimension)
assert model.dmin.attention_net[-1].attention_a[0].out_features==256
assert model.dmin.classifiers[0].degree==RECIPES[profile]['degree'] and len(model.lipn.extractors)==2
names=[[name for name,_ in module.named_parameters()] for module in [model.pretrain,model.lipn,model.dmin]]
initial=model.pretrain.state_dict();before={k:initial[k].clone() for k in ['attention_net.0.weight','classifiers.0.act_fun.0.cheby_coeffs']}
del model,initial;gc.collect()
write_json(out/'progress.json',dict(state='running_complete_pretraining_and_joint_native_budgets',profile=profile,dimension=dimension))
run=train(data,profile,out/'run',device=device,seed=2026,config={'threads':2})
status=read_json(run/'status.json');request=read_json(run/'request.json');cfg=request['config']
assert status['state']=='completed' and cfg['epochs']==400 and cfg['patience']==20
assert cfg['lr']==RECIPES[profile]['lr'] and cfg['weight_decay']==1e-5 and cfg['model']=={}
assert cfg['max_instances'] is None and cfg['grad_clip'] is None and cfg['selection_metric']=='ahdmil_source_ce'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False);best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
extra=last['model']['_extra_state'];trace=extra['selection_history'];history=last['history']
assert extra['completed_epochs']==len(history)==len(trace) and extra['stage']=='joint' and extra['training_done']
assert not extra['transfer_pending'] and extra['stage_seed']==2026 and len(last['optimizers'])==3
train_ids={row['slide_id'] for row in bundle.records('train')}
stage_epochs={stage:sum(r['stage']==stage for r in trace) for stage in ['dmin','joint']}
assert 50<=stage_epochs['dmin']<=200 and 1<=stage_epochs['joint']<=200
assert extra['updates']==dict(pretrain=stage_epochs['dmin']*n,lipn=stage_epochs['joint']*n,dmin=stage_epochs['joint']*n)
assert extra['ema_calls']==stage_epochs['joint']+1
for stage,patience in [('dmin',20),('joint',5)]:
 rows=[r for r in trace if r['stage']==stage]
 assert len(rows)==200 or extra['stage_bad'][stage]>=patience
 assert all(r['stage_epoch']==i+1 for i,r in enumerate(rows))
 orders=[r for r in extra['train_orders'] if r['stage']==stage]
 assert len(orders)==len(rows) and all(len(r['slides'])==n and set(r['slides'])<=train_ids for r in orders)
 if stage=='joint':
  assert all(len(r['manners'])==n and set(r['manners'])<={'hard_train','attn_train'} for r in orders)
  if RECIPES[profile]['attn_ratio']==0:assert all(set(r['manners'])=={'hard_train'} for r in orders)
 labels={row['slide_id']:row['target'] for row in bundle.records('train')}
for index,optimizer in enumerate(last['optimizers']):
 for pid,state in optimizer['state'].items():
  name=names[index][pid];expected=extra['updates']['pretrain'] if index==0 else extra['generation_updates']
  if index==0 and not RECIPES[profile]['subtyping'] and name.startswith('instance_classifiers.'):
   cls=int(name.split('.')[1]);expected=sum(labels[sid]==cls for order in extra['train_orders'] if order['stage']=='dmin' for sid in order['slides'])
  assert int(state['step'])==expected,(index,name,int(state['step']),expected)
 assert all(group['lr']==(cfg['lr'] if index==0 else 1e-5) and group['weight_decay']==1e-5 for group in optimizer['param_groups'])
assert all(not torch.equal(value,last['model']['pretrain.'+key]) for key,value in before.items())
assert extra['selected_dmin_hash']==tensor_state_hash(extra['selected_dmin'])
selected_pretrain=min((r for r in trace if r['stage']=='dmin' and r['eligible']),key=lambda r:r['value'])
assert extra['selected_dmin_epoch']==selected_pretrain['stage_epoch']
selected=min((r for r in trace if r['stage']=='joint' and r['eligible']),key=lambda r:r['value'])
assert best['epoch']==selected['epoch'] and selected['lower_before']<=selected['rate']<=1
assert extra['joint_generation']==len(extra['reinitializations'])
for reset in extra['reinitializations']:
 assert trace[reset['epoch']]['reset'] and trace[reset['epoch']]['best'] is None
 assert reset['lower_after']==(reset['lower_before']-.05 if classes==3 and reset['joint_epoch']>50 else reset['lower_before'])
for row,record in zip(history,trace):
 terms=['classification','kl','representation','retention'] if record['stage']=='dmin' else ['lipn_loss','dmin_ce','distillation','retention','mask_rate','score_rate']
 assert all(k in row['train'] and torch.isfinite(torch.tensor(row['train'][k])) for k in terms)
protocol=read_json(run/'method_protocol.json')
assert protocol['stages']==dict(dmin_epochs=200,joint_epochs=200,dmin_selection_start=50,dmin_patience=20,joint_patience=5,joint_lr=1e-5)
assert protocol['actual_optimizer_updates']==extra['updates'] and protocol['cross_ema_calls']==extra['ema_calls']
report=dict(state='passed',case=case,profile=profile,dimension=dimension,classes=classes,software=software_identity(),
 source='2f38354386165167815eb2998424ad63792a65bf',stage_max_epochs=[200,200],actual_stage_epochs=stage_epochs,
 actual_optimizer_updates=extra['updates'],cross_ema_calls=extra['ema_calls'],reset_generations=extra['joint_generation'],
 actual_true_patches_per_slide=64,native_rgb_pixels=16,native_high_patch_pixels=256,native_hidden=512,native_attention=256,native_experts=2,
 degree=RECIPES[profile]['degree'],train_slides=n,selected_pretrained_sha256=extra['selected_dmin_hash'],selected_pretrained_epoch=extra['selected_dmin_epoch'],selected_epoch=selected['epoch'],
 source_pretraining_and_joint_budgets_verified=True,source_eligible_selection_and_reset_verified=True,synthetic=True,formal_admission=False)
del last,best,extra;gc.collect();torch.cuda.empty_cache()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
frame=pd.read_csv(run/'test_predictions.csv');assert frame.slide_id.is_unique and len(frame)==len(bundle.records('test'))
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
report.update(metrics=metrics,selected_checkpoint_reloaded_twice=True,prediction_sha256=prediction,device=torch.cuda.get_device_name(device),
 wall_seconds=time.monotonic()-started,scope='Complete selected DMIN pretraining then complete asymmetric source joint recipe, two native experts; no natural WSI or historical paper-score claim')
write_json(out/'validation.json',report)
print(json.dumps({'state':'passed','case':case,'epochs':report['actual_stage_epochs']}),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--devices', default='cuda:0', help='Unique native CUDA workers; all eight cases remain mandatory')
    parser.add_argument('--workers-per-device', type=int, default=1,
                        help='Independent verification processes per GPU; concurrent wall time is not an efficiency benchmark')
    args = parser.parse_args()
    devices = args.devices.split(',')
    if (not devices or len(set(devices)) != len(devices) or any(not re.fullmatch(r'cuda:[0-9]+', device) for device in devices)
            or not 1 <= args.workers_per_device <= 4):
        raise ValueError('Provide distinct explicit CUDA devices')
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    target = out/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith('milplus/') and not name.endswith('/'):
                assert archive.read(name) == (Path(args.source_tree)/'src'/name).read_bytes(), name
        archive.extractall(target)
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join(path for path in environment.get('PYTHONPATH', '').split(os.pathsep)
        if path and Path(path).resolve() != (Path(args.source_tree)/'src').resolve())
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    cases = [{'case': index, 'state': 'pending'} for index in range(8)]
    active, handles, next_case = {}, [], 0
    slots = [(device, worker) for device in devices for worker in range(args.workers_per_device)]
    started = time.monotonic()
    def save():
        value = {'state': 'running', 'profiles': cases, 'wall_seconds': time.monotonic()-started,
                 'all_eight_complete_source_budgets_required': True}
        temporary = out/'progress.tmp'; temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(out/'progress.json')
    try:
        while next_case < 8 or active:
            for slot in slots:
                device, worker = slot
                if next_case < 8 and slot not in active:
                    index = next_case; next_case += 1
                    log = (out/f'case_{index}.log').open('x'); handles.append(log)
                    command = [sys.executable, '-c', PROGRAM, str(target), str(out/f'case_{index}'), str(index), device]
                    child = subprocess.Popen(command, cwd=out, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    active[slot] = (index, child)
                    cases[index].update(state='running', pid=child.pid, device=device, independent_worker_slot=worker)
                    save()
            for slot, (index, child) in list(active.items()):
                code = child.poll()
                if code is not None:
                    cases[index].update(state='passed' if code == 0 else 'failed', returncode=code)
                    del active[slot]; save()
            time.sleep(1)
        if any(row['state'] != 'passed' for row in cases):
            raise RuntimeError('One or more complete native AHDMIL cases failed; preserve and inspect all case logs')
        reports = [json.loads((out/f'case_{index}/validation.json').read_text()) for index in range(8)]
        assert {(row['profile'],row['dimension']) for row in reports} == {(p,d) for p in ['ahdmil_camelyon16_source_repaired','ahdmil_nsclc_source_repaired','ahdmil_brca_source_repaired','ahdmil_rcc_source_repaired'] for d in [1024,1536]}
        assert all(row['state'] == 'passed' and row['software'] == reports[0]['software'] for row in reports)
        report = {'state': 'passed', 'profiles': reports, 'software': reports[0]['software'],
                  'wall_seconds': time.monotonic()-started, 'formal_admission': False,
                  'independent_workers_per_device': args.workers_per_device,
                  'efficiency_benchmark': False,
                  'scope': 'All eight independent-wheel complete native two-stage DMIN/AHDMIL joint recipes; broad CPU suite remains a separate gate'}
        (out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'state': 'passed', 'profiles': 8}), flush=True)
    finally:
        for log in handles:
            log.close()


if __name__ == '__main__':
    main()
