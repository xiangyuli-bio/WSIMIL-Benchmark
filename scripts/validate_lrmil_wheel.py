"""Four independent-wheel full LRMIL native encoder/teacher/student declared budgets.

Device workers only parallelize the four fixed cases. Every case, the complete
declared budget, source patch batch64, measured-in-fixture RGB8/HR128 native768-layer inputs and all losses
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

import gc,json,sys,time,shutil
from pathlib import Path
target,out=map(Path,sys.argv[1:3]);case=int(sys.argv[3]);device=sys.argv[4]
sys.path.insert(0,str(target))
import torch,milplus
import pandas as pd
from milplus.data import Bundle
from milplus.engine import train,evaluate,seed_everything
from milplus.models import build_bundle_model
from milplus.lrmil_demo import create_lrmil_demo
from milplus.lrmil import state_hash
from milplus.io import write_json,read_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.75.0' and torch.cuda.is_available()
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
torch.cuda.set_device(device);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
classes=[2,3,4,7][case];patients=10*classes
out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
data,_=create_lrmil_demo(out/'data',patients=patients,classes=classes,low_patches=8)
bundle=Bundle(data);n=len(bundle.records('train'));assert n>0 and bundle.config['synthetic'] is True
assert all(bundle.lrmil.inventory(row)['low_nodes']==8 and bundle.lrmil.inventory(row)['high_nodes']==128 for row in bundle.records('train'))
seed_everything(2026);model=build_bundle_model('lrmil_source_repaired',bundle)
identity=model.encoder._milplus_lrmil_identity
assert identity['visual_width']==768 and identity['visual_layers']==12 and identity['visual_heads']==12 and identity['visual_parameter_count']==86250240
markers=['teacher.cls.weight','teacher.mil.attention.0.weight',
 'encoder.visual.patch_embed.proj.weight','encoder.visual.blocks.0.attn.qkv.weight',
 'encoder.visual.blocks.10.attn.qkv.weight','encoder.visual.blocks.11.attn.qkv.weight',
 'student.cls.weight','student.mil.attention.0.weight','student.inst_classifier.weight']
initial=model.state_dict();before={key:initial[key].clone() for key in markers}
del initial,model;gc.collect()
write_json(out/'progress.json',{'state':'running_all_three_complete_declared_native_budgets','case':case,'classes':classes})
run=train(data,'lrmil_source_repaired',out/'run',device=device,seed=2026,config={'threads':2})
status=read_json(run/'status.json');request=read_json(run/'request.json');cfg=request['config']
assert status['state']=='completed' and cfg['epochs']==600 and cfg['patience']==20
assert cfg['lr']==.0001 and cfg['weight_decay']==0 and cfg['model']=={}
assert cfg['max_instances'] is None and cfg['grad_clip'] is None and cfg['selection_metric']=='lrmil_stage_selection'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False);best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
state=last['model'];extra=state['_extra_state'];history=last['history']
assert extra['phase']=='student' and extra['training_done'] and len(last['optimizers'])==3
assert extra['completed_epochs']==len(history)==len(extra['train_orders'])==len(extra['selection_history'])
phase_rows={phase:[r for r in extra['selection_history'] if r['phase']==phase] for phase in ['teacher','patch','student']}
assert len(phase_rows['patch'])==200
for phase in ['teacher','student']:
 rows=phase_rows[phase];assert 1<=len(rows)<=200
 assert len(rows)==200 or rows[-1]['done'] and rows[-1]['bad']>=20
expected_updates={'teacher':len(phase_rows['teacher'])*n,'patch':200*((8*n+63)//64),'student':len(phase_rows['student'])*n}
assert extra['updates']==expected_updates and extra['student_instance_updates']>0
allowed={row['slide_id'] for row in bundle.records('train')}
patches={row['slide_id']+':'+str(index) for row in bundle.records('train') for index in range(8)}
for record,order in zip(extra['selection_history'],extra['train_orders']):
 actual=patches if record['phase']=='patch' else allowed
 assert len(order)==len(actual) and set(order)==actual
for phase,optimizer in zip(['teacher','patch','student'],last['optimizers']):
 assert len(optimizer['param_groups'])==1
 assert optimizer['param_groups'][0]['lr']==.0001 and optimizer['param_groups'][0]['weight_decay']==0
 expected=expected_updates[phase]
 instance_parameters=set(optimizer['param_groups'][0]['params'][-2:]) if phase=='student' else set()
 assert all(int(item['step'])==(extra['student_instance_updates'] if parameter in instance_parameters else expected)
            for parameter,item in optimizer['state'].items())
for row,selection in zip(history,extra['selection_history']):
 terms=row['train'];assert all(torch.isfinite(torch.tensor(value)) for value in terms.values())
 if selection['phase']=='teacher':assert abs(terms['loss']-terms['teacher_ce'])<1e-6
 elif selection['phase']=='patch':
  assert {'layer_mse','attention_weighted_mse'}<=set(terms) and terms['layer_mse']>=0
  assert abs(terms['loss']-terms['layer_mse']-terms['attention_weighted_mse'])<1e-5
 else:
  assert {'ce','bag_kd','soft_attention_kd','hard_instance_ce'}<=set(terms)
  assert abs(terms['loss']-terms['ce']-.5*terms['bag_kd']-.5*terms['soft_attention_kd']-.3*terms['hard_instance_ce'])<1e-5
assert all(not torch.equal(before[key],state[key]) for key in markers)
assert state['encoder.visual.blocks.11.attn.qkv.weight'].shape==(2304,768)
assert state['encoder.visual.patch_embed.proj.weight'].shape==(768,3,16,16)
assert state['teacher.cls.weight'].shape==state['student.cls.weight'].shape==(classes,768)
assert state['student.inst_classifier.weight'].shape==(2,768)
assert state_hash(extra['best_encoder'])==extra['best_encoder_hash']
assert state_hash({key.removeprefix('encoder.visual.'):value for key,value in state.items() if key.startswith('encoder.visual.')})==extra['best_encoder_hash']
assert state_hash({key.removeprefix('teacher.'):value for key,value in state.items() if key.startswith('teacher.')})==extra['best_teacher_hash']
for phase in ['teacher','student']:
 rows=phase_rows[phase];maximum=max(row['value'] for row in rows)
 assert extra['selected_epochs'][phase]==next(row['phase_epoch'] for row in rows if row['value']==maximum)
patch_history=[history[row['epoch']] for row in phase_rows['patch']]
assert extra['selected_epochs']['patch']==next(row['phase_epoch'] for row in phase_rows['patch'] if row['value']==min(record['value'] for record in phase_rows['patch']))
selected=next(row['epoch'] for row in phase_rows['student'] if row['value']==max(record['value'] for record in phase_rows['student']))
assert best['epoch']==selected
protocol=read_json(run/'method_protocol.json')
assert protocol['phases']=={'teacher':200,'patch':200,'student':200} and protocol['options']['patch_batch_size']==64 and protocol['options']['topk']==4
assert protocol['actual_optimizer_updates']==expected_updates
del state,last,best,extra;gc.collect();torch.cuda.empty_cache()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
frame=pd.read_csv(run/'test_predictions.csv');assert frame.slide_id.is_unique and len(frame)==len(bundle.records('test')) and set(frame.target)==set(range(classes))
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
report={'state':'passed','case':case,'profile':'lrmil_source_repaired','software':software_identity(),
 'classes':classes,'dimension':768,'patients':patients,'train_slides':n,'native_visual_width':768,'native_visual_depth':12,'native_visual_heads':12,'native_visual_parameters':86250240,
 'true_synthetic_low_patches':8,'true_synthetic_high_patches':128,'patch_batch_size':64,'topk':4,
 'stage_max_epochs':[200,200,200],'completed_phase_epochs':{key:len(rows) for key,rows in phase_rows.items()},
 'actual_optimizer_updates':expected_updates,'all_nine_module_markers_updated':True,
 'all_source_objectives_active':True,'all_three_actual_Adam_states_verified':True,
 'actual_selected_teacher_encoder_transfers_verified':True,'source_first_optimum_selection_verified':True,
 'selected_checkpoint_reloaded_twice':True,'selected_global_epoch':selected,
 'prediction_sha256':prediction,'metrics':metrics,'device':torch.cuda.get_device_name(device),
 'deterministic_CUDA_algorithms':True,'wall_seconds':time.monotonic()-started,'synthetic':True,'formal_admission':False,
 'scope':'Complete native full pretrained CONCH and all original source stages/objectives; explicit200/200/200 engineering caps where published CLI lacks defaults, not claimed historical paper caps. Full classification C2/C3/C4/C7 source heads; no realWSI/original-paper-score claim.'}
write_json(out/'validation.json',report)
print(json.dumps({'state':'passed','case':case,'classes':classes,'phase_epochs':report['completed_phase_epochs']}),flush=True)

'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--devices', default='cuda:0', help='Unique native CUDA workers; all four cases remain mandatory')
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
    cases = [{'case': index, 'state': 'pending'} for index in range(4)]
    active, handles, next_case = {}, [], 0
    slots = [(device, worker) for device in devices for worker in range(args.workers_per_device)]
    started = time.monotonic()
    def save():
        value = {'state': 'running', 'profiles': cases, 'wall_seconds': time.monotonic()-started,
                 'all_four_complete_source_budgets_required': True}
        temporary = out/'progress.tmp'; temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(out/'progress.json')
    try:
        while next_case < 4 or active:
            for slot in slots:
                device, worker = slot
                if next_case < 4 and slot not in active:
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
            raise RuntimeError('One or more complete native LRMIL cases failed; preserve and inspect all case logs')
        reports = [json.loads((out/f'case_{index}/validation.json').read_text()) for index in range(4)]
        assert {(row['dimension'], row['classes']) for row in reports} == {(768, 2), (768, 3), (768, 4), (768, 7)}
        assert all(row['state'] == 'passed' and row['software'] == reports[0]['software'] for row in reports)
        report = {'state': 'passed', 'profiles': reports, 'software': reports[0]['software'],
                  'wall_seconds': time.monotonic()-started, 'formal_admission': False,
                  'independent_workers_per_device': args.workers_per_device,
                  'efficiency_benchmark': False,
                  'scope': 'All four independent-wheel full native pretrained CONCH/HRteacher/patchKD/LRstudent recipes at declared200/200/200 caps and C2/C3/C4/C7; broad CPU suite remains a separate gate'}
        (out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'state': 'passed', 'profiles': 4}), flush=True)
    finally:
        for log in handles:
            log.close()


if __name__ == '__main__':
    main()

