"""Eight independent-wheel complete DMIN200 + LIPN200-max budgets.

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
from milplus.hdmil_source import RECIPES
from milplus.hdmil_demo import create_hdmil_demo
from milplus.io import write_json,read_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.72.0' and torch.cuda.is_available()
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
torch.cuda.set_device(device);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
profile=list(RECIPES)[case//2];dimension=[1024,1536][case%2];classes=RECIPES[profile]['classes']
out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
data=create_hdmil_demo(out/'data',patients=10*classes,dimension=dimension,classes=classes,patches=64)
bundle=Bundle(data);n=len(bundle.records('train'));assert bundle.config['synthetic'] is True
assert all(bundle.hdmil.inventory(row)['nodes']==64 for row in bundle.rows.to_dict('records'))
seed_everything(2026);model=build_bundle_model(profile,bundle)
assert model.dmin.attention_net[0].weight.shape==(512,dimension)
assert model.dmin.attention_net[-1].attention_a[0].out_features==256
assert model.dmin.classifiers[0].degree==RECIPES[profile]['degree']
parameter_names={'dmin':[name for name,_ in model.dmin.named_parameters()],'lipn':[name for name,_ in model.processor.named_parameters()]}
initial=model.dmin.state_dict();before={key:initial[key].clone() for key in ['attention_net.0.weight','classifiers.0.act_fun.0.cheby_coeffs','instance_classifiers.0.act_fun.0.cheby_coeffs']}
del model,initial;gc.collect()
write_json(out/'progress.json',dict(state='running_complete_two_stage_native_budget',profile=profile,dimension=dimension))
run=train(data,profile,out/'run',device=device,seed=2026,config={'threads':2})
status=read_json(run/'status.json');request=read_json(run/'request.json');cfg=request['config']
assert status['state']=='completed' and cfg['epochs']==400 and cfg['patience']==20
assert cfg['lr']==RECIPES[profile]['lr'] and cfg['weight_decay']==1e-5 and cfg['model']=={}
assert cfg['max_instances'] is None and cfg['grad_clip'] is None and cfg['selection_metric']=='hdmil_source_ce'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False);best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
extra=last['model']['_extra_state'];history=last['history'];trace=extra['selection_history']
assert extra['completed_epochs']==len(history)==len(trace) and extra['stage']=='lipn' and extra['training_done']
assert not extra['transfer_pending'] and extra['stage_seed']==2026
train_ids={row['slide_id'] for row in bundle.records('train')}
for stage,minimum,patience in [('dmin',50,20),('lipn',1,5)]:
 records=[r for r in trace if r['stage']==stage];epochs=len(records)
 assert minimum<=epochs<=200 and extra['updates'][stage]==epochs*n
 assert epochs==200 or extra['stage_bad'][stage]>=patience
 assert all(r['stage_epoch']==i+1 for i,r in enumerate(records))
 orders=[r['slides'] for r in extra['train_orders'] if r['stage']==stage]
 assert len(orders)==epochs and all(len(order)==n and set(order)<=train_ids for order in orders)
 labels={row['slide_id']:row['target'] for row in bundle.records('train')}
 for pid,state in last['optimizers'][int(stage=='lipn')]['state'].items():
  name=parameter_names[stage][pid]
  expected=extra['updates'][stage]
  if stage=='dmin' and not RECIPES[profile]['subtyping'] and name.startswith('instance_classifiers.'):
   cls=int(name.split('.')[1]);expected=sum(labels[sid]==cls for order in orders for sid in order)
  assert int(state['step'])==expected,(stage,name,int(state['step']),expected)
 assert all(group['lr']==(cfg['lr'] if stage=='dmin' else .0001) and group['weight_decay']==1e-5 for group in last['optimizers'][int(stage=='lipn')]['param_groups'])
 assert all(any(abs(row['train'][key])>0 for row,r in zip(history,trace) if r['stage']==stage) for key in (['classification','kl','representation','retention'] if stage=='dmin' else ['distillation','retention']))
assert all(not torch.equal(value,last['model']['dmin.'+key]) for key,value in before.items())
assert extra['best_dmin_hash']==tensor_state_hash(extra['best_dmin'])==tensor_state_hash({k[len('teacher.'):]:v for k,v in last['model'].items() if k.startswith('teacher.')})
selected_teacher=min((r for r in trace if r['stage']=='dmin' and r['eligible']),key=lambda r:r['value'])
assert extra['best_dmin_epoch']==selected_teacher['stage_epoch']
selected=min((r for r in trace if r['stage']=='lipn'),key=lambda r:r['value'])
assert best['epoch']==selected['epoch']
protocol=read_json(run/'method_protocol.json');assert protocol['stages']==dict(dmin_epochs=200,lipn_epochs=200,dmin_selection_start=50,dmin_patience=20,lipn_patience=5,lipn_lr=.0001)
assert protocol['actual_optimizer_updates']==extra['updates']
report=dict(state='passed',case=case,profile=profile,dimension=dimension,classes=classes,software=software_identity(),
 source='ef3d4681054970208e94f8205bd96b3180c00926',stage_max_epochs=[200,200],actual_stage_epochs={stage:sum(r['stage']==stage for r in trace) for stage in ['dmin','lipn']},
 actual_optimizer_updates=extra['updates'],actual_true_patches_per_slide=64,native_rgb_pixels=16,native_high_patch_pixels=256,native_hidden=512,native_attention=256,
 degree=RECIPES[profile]['degree'],train_slides=n,selected_dmin_sha256=extra['best_dmin_hash'],selected_dmin_epoch=extra['best_dmin_epoch'],selected_epoch=selected['epoch'],
 selected_teacher_transferred_verified=True,source_stage_early_stopping_verified=True,synthetic=True,formal_admission=False)
del last,best,extra;gc.collect();torch.cuda.empty_cache()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
frame=pd.read_csv(run/'test_predictions.csv');assert frame.slide_id.is_unique and len(frame)==len(bundle.records('test'))
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
report.update(metrics=metrics,selected_checkpoint_reloaded_twice=True,prediction_sha256=prediction,device=torch.cuda.get_device_name(device),
 wall_seconds=time.monotonic()-started,scope='Complete released cached-feature and matched-RGB two-stage native budgets; no online FM throughput, natural WSI or paper-score claim')
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
            raise RuntimeError('One or more complete native HDMIL cases failed; preserve and inspect all case logs')
        reports = [json.loads((out/f'case_{index}/validation.json').read_text()) for index in range(8)]
        assert {(row['profile'],row['dimension']) for row in reports} == {(p,d) for p in ['hdmil_camelyon16_source','hdmil_nsclc_source','hdmil_brca_source','hdmil_rcc_source'] for d in [1024,1536]}
        assert all(row['state'] == 'passed' and row['software'] == reports[0]['software'] for row in reports)
        report = {'state': 'passed', 'profiles': reports, 'software': reports[0]['software'],
                  'wall_seconds': time.monotonic()-started, 'formal_admission': False,
                  'independent_workers_per_device': args.workers_per_device,
                  'efficiency_benchmark': False,
                  'scope': 'All eight independent-wheel complete native two-stage DMIN/LIPN recipes; broad CPU suite remains a separate gate'}
        (out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'state': 'passed', 'profiles': 8}), flush=True)
    finally:
        for log in handles:
            log.close()


if __name__ == '__main__':
    main()
