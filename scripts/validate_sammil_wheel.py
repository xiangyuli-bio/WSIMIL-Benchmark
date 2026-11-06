"""Four complete native-width200max SAM-MIL installed-wheel CPU references.

Each run keeps the released512 projection/128attention, all five pseudo-bags,
original consistency kernel, original Adam/cosine/early-stop recipe and every
ordinary/group token. Synthetic software validation, not clinical performance.
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

PROGRAM = r'''
import gc,math,sys,time
from pathlib import Path
target,out=map(Path,sys.argv[1:3]);case=int(sys.argv[3]);sys.path.insert(0,str(target))
import torch,milplus
from milplus.data import Bundle
from milplus.engine import train,evaluate,seed_everything
from milplus.models import build_bundle_model
from milplus.sammil_demo import create_sammil_demo
from milplus.sammil import weights_digest
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.76.0'
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
torch.set_num_threads(2)
profile,dimension,classes=[('sammil_attn_source_control',1024,2),
 ('sammil_attn_objective_repaired',1024,2),
 ('sammil_attn_objective_repaired',1536,3),
 ('sammil_attn_objective_repaired',512,7)][case]
out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
data=create_sammil_demo(out/'data',patients=10*classes,dimension=dimension,classes=classes,patches=128)
bundle=Bundle(data);n=len(bundle.records('train'));assert n>=18
assert all(len(bundle.sammil.load(row)['features'])==131 for row in bundle.rows.to_dict('records'))
seed_everything(2026);model=build_bundle_model(profile,bundle)
teacher_sha=weights_digest(model.teacher)
markers=['patch_to_emb.0.weight','online_encoder.attn_model.attention.attention.0.weight','predictor.weight']
before={name:model.student.state_dict()[name].clone() for name in markers}
del model;gc.collect()
run=train(data,profile,out/'run',seed=2026,config={'threads':2})
assert not (run/'test_predictions.csv').exists()
cfg=read_json(run/'request.json')['config'];assert cfg['epochs']==200 and cfg['patience']==20
assert cfg['lr']==2e-4 and cfg['weight_decay']==1e-5 and cfg['model']=={}
assert cfg['max_instances'] is None and cfg['grad_clip'] is None and cfg['selection_metric']=='sammil_val_auc'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
extra=last['model']['_extra_state'];epochs=extra['completed_epochs'];history=last['history']
assert epochs==len(history) and last['epoch']==epochs-1 and extra['optimizer_updates']==epochs*n
assert epochs==200 or (epochs>=72 and extra['training_done'] and extra['early_bad']>=20 and last['epoch']>70)
allowed={row['slide_id'] for row in bundle.records('train')}
assert all(len(order)==n and set(order)==allowed for order in extra['train_orders'])
assert extra['scheduler']['last_epoch']==epochs and extra['scheduler']['T_max']==200
assert extra['initial_teacher_sha256']==teacher_sha
assert bool(extra['teacher_gradients'])==profile.endswith('control')
assert last['model']['student.patch_to_emb.0.weight'].shape==(512,dimension)
assert last['model']['student.online_encoder.attn_model.attention.attention.0.weight'].shape==(128,512)
assert last['model']['student.predictor.weight'].shape==(classes,512)
assert all(not torch.equal(before[name],last['model']['student.'+name]) for name in markers)
for key in ['global_ce','group_ce','consistency']:
 assert all(math.isfinite(row['train'][key]) for row in history)
 assert any(row['train'][key]>0 for row in history)
assert all(row['train']['patch_count']==131 for row in history)
assert all(3<=row['train']['kept']<=131 for row in history)
auc=[row['validation']['sammil_val_auc'] for row in history];selected=auc.index(max(auc))
assert best['epoch']==selected
optimizer=last['optimizers'][0];assert len(optimizer['state'])==len([key for key in last['model'] if key.startswith('student.')])
assert all(int(state['step'])==epochs*n for state in optimizer['state'].values())
assert optimizer['param_groups'][0]['weight_decay']==1e-5
assert optimizer['param_groups'][0]['lr']==extra['scheduler']['_last_lr'][0]
del last,best,extra,optimizer;gc.collect()
metrics=evaluate(run);sha=digest_file(run/'test_predictions.csv')
assert evaluate(run)==metrics and digest_file(run/'test_predictions.csv')==sha
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
write_json(out/'validation.json',{'state':'passed','case':case,'profile':profile,'dimension':dimension,
 'classes':classes,'projection':512,'attention':128,'max_epochs':200,'completed_epochs':epochs,
 'batch_size':1,'ordinary_patches':128,'SAM_group_tokens':3,'pseudo_bags':5,
 'source_weights':[1.,0.,.5,1000.],'actual_optimizer_updates':epochs*n,
 'selected_epoch':selected,'source_first_best_auc_and_latest_tie_stopping':True,
 'complete_teacher_and_cosine_state_validated':True,'all_source_objectives_computed':True,
 'selected_checkpoint_reloaded_twice':True,'prediction_sha256':sha,'metrics':metrics,
 'software':software_identity(),'wall_seconds':time.monotonic()-started,
 'synthetic':True,'formal_admission':False,'device':'cpu',
 'scope':'Complete native attention source/control or explicit repaired recipe; no real SAM segmentation, FM extraction or clinical/paper-score claim'})
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    wheel, source, output = [Path(value).resolve() for value in [args.wheel, args.source_tree, args.output]]
    if output.exists():
        raise ValueError('Preserve existing native attempts; use a new output root')
    output.mkdir(parents=True); target = output/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not (target/name).resolve().is_relative_to(target):
                raise ValueError('Unsafe installed-wheel archive member')
        archive.extractall(target)
    assert (target/'milplus/__init__.py').is_file()
    reports = []
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'}
    for case in range(4):
        with (output/f'case_{case}.log').open('x') as log:
            subprocess.run([sys.executable, '-c', PROGRAM, str(target), str(output/f'case_{case}'), str(case)],
                cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((output/f'case_{case}/validation.json').read_text())
        assert report['state'] == 'passed' and report['software'] == reports[0]['software'] if reports else report['state'] == 'passed'
        reports.append(report)
        (output/'progress.json').write_text(json.dumps({'state': 'running', 'profiles': reports}, indent=2)+'\n')
    assert len(reports) == 4
    (output/'validation.json').write_text(json.dumps({'state': 'passed', 'software': reports[0]['software'],
        'profiles': reports, 'formal_admission': False, 'synthetic': True,
        'scope': 'All four independent-wheel full200max/native512projection/128attention recipes; broad regression remains separate'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
