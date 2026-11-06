"""Complete source-budget AugDiff recipes from an isolated wheel on CUDA.

The source paper/YAML never specifies DAE epoch count. The required explicit
--dae-epochs is a declared engineering budget, not a claim of historical paper
reproduction. The released private epoch368 example motivates the caller's369.
All six MIL conditions retain the source200max/patience10 and native512/C4.
"""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['wheel', 'output', 'source-tree']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--dae-epochs', required=True, type=int)
    args = parser.parse_args()
    if args.dae_epochs < 1:
        parser.error('DAE budget must be explicitly positive')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    target = out/'wheel_import'
    target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith('milplus/') and not name.endswith('/'):
                assert archive.read(name) == (Path(args.source_tree)/'src'/name).read_bytes(), name
        archive.extractall(target)
    program = r'''
import gc,sys,json,time
from collections import Counter
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import numpy as np
import pandas as pd
import torch,yaml,milplus
from milplus.augdiff import PROFILES
from milplus.augdiff_assets import export_trained_dae
from milplus.augdiff_data import AugDiffInputs
from milplus.augdiff_demo import create_augdiff_demo
from milplus.augdiff_pretrain import DiffusionTrainer
from milplus.data import Bundle
from milplus.engine import train,evaluate
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity

target,out=map(Path,sys.argv[1:3]);dae_epochs=int(sys.argv[3])
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
assert milplus.__version__=='0.68.0' and torch.cuda.is_available()
torch.set_num_threads(2)
started=time.monotonic()
report={'state':'running','formal_admission':False,'software':software_identity(),
 'device':torch.cuda.get_device_name(0),'profiles':[],
 'historical_architecture_reproduced':False,'paper_reproduction':'not_run',
 'budget_scope':'DAE caller-declared369epoch checkpoint-reference engineering budget, not paper-stated universal epoch count; original MIL200max/p10, full512/C4, native full UNet256/512/1024, single-device LR scaling. Synthetic software reference, no natural WSI score.'}
def save():
 report['wall_seconds']=time.monotonic()-started
 write_json(out/'progress.json',report)
save()
root=create_augdiff_demo(out/'native_data',patients=36,dimension=512,instances=6,classes=4)
bundle=Bundle(root);assets=AugDiffInputs(bundle,'augdiff_views.json')
assert bundle.dim==512 and len(bundle.classes)==4
# At least one actually executed full1200patch batch, not just configured
# batch1200 with a tiny final-only batch. Every remaining row is also used.
train_n=len(assets.patches('train'));val_n=len(assets.patches('val'))
assert train_n>1200 and train_n%1200 and val_n>0
report['DAE_input']={'training_patches':train_n,'validation_patches':val_n,
 'batch_sizes_each_epoch':[1200]*(train_n//1200)+[train_n%1200],
 'supervised_train_slides':len(bundle.records('train')),'classes':4,'dimension':512,
 'view_pairing_policy':assets.pairing,'synthetic':True}
save()
trainer=DiffusionTrainer(assets,epochs=dae_epochs,device='cuda',batch_size=1200,precision='fp16',seed=2026)
protocol=trainer.fit(out/'dae_pretrain')
assert protocol['completed_epochs']==dae_epochs==protocol['max_epochs']
assert protocol['full_dae_parameters']==122055680 and protocol['diffusion_timesteps']==1000
assert protocol['batch_size']==1200 and protocol['scaled_lr']==5e-8*1200
assert protocol['precision']=='fp16' and protocol['device_type']=='cuda'
assert protocol['optimizer_updates']>0 and protocol['optimizer_updates']==sum(h['optimizer_updates'] for h in trainer.history)
assert all(sorted(order)==list(range(train_n)) for order in trainer.orders)
assert len(trainer.orders)==dae_epochs
assert int(trainer.model.model_ema.num_updates)==dae_epochs*((train_n+1199)//1200)
assert all(h['validation']['patches']==val_n and np.isfinite(h['train_mse']) for h in trainer.history)
report['DAE_training']={'state':'passed','protocol':protocol,
 'attempted_updates':int(trainer.model.model_ema.num_updates),
 'all_registered_rows_consumed_each_epoch':True,'selected_by':'strict minimum stochastic EMA validation MSE',
 'MIL_weights':'selected checkpoint RAW, not EMA'}
save();del trainer;gc.collect();torch.cuda.empty_cache()
manifest=export_trained_dae(bundle,'augdiff_views.json',out/'dae_pretrain',root/'augdiff_model.json')
task_path=root/'task.yaml';task=yaml.safe_load(task_path.read_text())
task['encoders']['toy']['augdiff']={'manifest':manifest}
task_path.write_text(yaml.safe_dump(task,sort_keys=False))
bundle=Bundle(root)
assert bundle.augdiff.metadata['pretrain_protocol']['completed_epochs']==dae_epochs
assert len(PROFILES)==6
training_rows=bundle.records('train');validation_rows=bundle.records('val')
for profile in PROFILES:
 run=out/'runs'/profile
 # Default full method config: no2epoch shortcut or patience override here.
 train(root,profile,run,device='cuda',seed=2026)
 assert read_json(run/'status.json')['state']=='completed'
 assert not (run/'test_predictions.csv').exists()
 request=read_json(run/'request.json');cfg=request['config']
 assert cfg['epochs']==200 and cfg['patience']==10 and cfg['lr']==2e-4 and cfg['weight_decay']==1e-5
 assert cfg['grad_clip'] is None and cfg['max_instances'] is None and cfg['selection_metric']=='augdiff_source_ce'
 settings=read_json(run/'method_protocol.json')['settings']
 assert settings['precision']=='auto' and settings['steps']==20 and settings['nu']==.4
 last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
 best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
 extra=last['model']['_extra_state'];epochs=extra['completed_epochs']
 assert len(last['history'])==epochs and 11<=epochs<=200
 assert epochs==200 or last['bad']>=10
 assert extra['optimizer_updates']>0 and extra['scaler']
 expected=Counter(r['slide_id'] for r in training_rows*7)
 assert all(Counter(order)==expected for order in extra['train_orders'])
 assert len(extra['train_orders'])==epochs and len(extra['window_history'])==epochs
 assert all(len(w)==(len(training_rows)*7+1)//2 for w in extra['window_history'])
 assert extra['optimizer_updates']==sum(int(window['stepped']) for w in extra['window_history'] for window in w)
 assert all(window['divisor']==2 for w in extra['window_history'] for window in w)
 assert best['epoch']==min(last['history'],key=lambda row:row['validation']['augdiff_source_ce'])['epoch']
 assert not any(key.startswith('_dae') for key in last['model'])
 del last,best;gc.collect();torch.cuda.empty_cache()
 metrics=evaluate(run,device='cuda');prediction=digest_file(run/'test_predictions.csv')
 assert evaluate(run,device='cuda')==metrics and digest_file(run/'test_predictions.csv')==prediction
 frame=pd.read_csv(run/'test_predictions.csv')
 assert len(frame)==len(bundle.records('test')) and frame.slide_id.is_unique
 assert (frame.augdiff_draws==1).all()
 report['profiles'].append({'state':'passed','id':profile,'epochs':epochs,'max_epochs':200,
  'patience':10,'full_feature_dimension':512,'classes':4,'actual_optimizer_updates':extra['optimizer_updates'],
  'amp_scaler':extra['scaler'],'TRAIN_repetitions':7,'VAL_repetitions':7,
  'TRAIN_draws_each_epoch':len(training_rows)*7,'VAL_draws_each_epoch':len(validation_rows)*7,
  'clean_TEST_once':True,'selected_checkpoint_reloaded_twice':True,'prediction_sha256':prediction,'metrics':metrics})
 save();del extra;gc.collect();torch.cuda.empty_cache()
assert len(report['profiles'])==6 and report['software']==software_identity()
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):
  assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
report['state']='passed';save();write_json(out/'validation.json',report)
print(json.dumps({'state':'passed','DAE_epochs':dae_epochs,'profiles':len(report['profiles']),
 'MIL_completed_epochs':sum(row['epochs'] for row in report['profiles'])}),flush=True)
'''
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(p for p in env.get('PYTHONPATH', '').split(os.pathsep)
                                     if p and Path(p).resolve() != (Path(args.source_tree)/'src').resolve())
    with (out/'program.log').open('w') as log:
        subprocess.run([sys.executable, '-c', program, str(target), str(out), str(args.dae_epochs)],
                       cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
