"""Independent-wheel complete100epoch three-scale source recipe on two FM widths.

Each of two public-config routes uses its actual native magnification ratios,
K, full512/256 graph,100Gaussian samples,8 readers,FP32,Adam and plateau.
Inputs are synthetic software references, not natural WSI/paper-score evidence.
"""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    target = out/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith('milplus/') and not name.endswith('/'):
                assert archive.read(name) == (Path(args.source_tree)/'src'/name).read_bytes(), name
        archive.extractall(target)
    program = r'''
import gc,sys,json,time,hashlib,shutil
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import numpy as np
import pandas as pd
import torch,milplus
from milplus.data import Bundle
from milplus.engine import train,evaluate,seed_everything
from milplus.models import build_bundle_model
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
from milplus.zoommil_demo import create_zoommil_demo

target,out=map(Path,sys.argv[1:]);assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
assert milplus.__version__=='0.69.0' and torch.cuda.is_available()
torch.set_num_threads(2)
report={'state':'running','formal_admission':False,'software':software_identity(),'profiles':[],
 'device':torch.cuda.get_device_name(0),'shared_memory_bytes':shutil.disk_usage('/dev/shm').total,
 'scope':'Four actual independent-wheel complete100epoch source3level/native1024 and full1536 FM-width reference runs, BRIGHT1:4:64/K12 and CRC1:4:16/K16. Full512/256/100Gaussian samples, weighted replacement sampler,8readers,Adam/plateau,complete CPU-CUDA RNG/selected reload. Explicit saver/device repairs. Synthetic only, no naturalWSI,paper score,two-levelEff or onlineFM total-cost claim.'}
assert report['shared_memory_bytes']>=512*1024**2,'Native8-reader prefetch needs adequate shared memory, no reader-count fallback'
started=time.monotonic()
def save():
 report['wall_seconds']=time.monotonic()-started
 write_json(out/'progress.json',report)
save()
for dimension in [1024,1536]:
 for geometry,profile,k,ratios in [
  ('bright','zoommil_bright_source_repaired',12,[1,4,64]),
  ('crc','zoommil_crc_source_repaired',16,[1,4,16])]:
  root=create_zoommil_demo(out/('data_'+geometry+'_'+str(dimension)),patients=27,
   dimension=dimension,classes=3,roots=k+4,geometry=geometry)
  bundle=Bundle(root)
  assert bundle.dim==dimension and len(bundle.classes)==3
  expected=[(k+4)*ratio for ratio in ratios]
  assert all(bundle.zoommil.inventory(row)['nodes_per_level']==expected for row in bundle.records('train'))
  seed_everything(2026)
  initial=build_bundle_model(profile,bundle).state_dict()
  selectors=['network.aux_ga_low_mag.attention_c.weight','network.aux_ga_mid_mag.attention_c.weight']
  before={name:initial[name].clone() for name in selectors};del initial
  run=out/'runs'/(profile+'_'+str(dimension))
  # No short epoch/patience/width/reader override can satisfy this native gate.
  train(root,profile,run,device='cuda',seed=2026,config={'threads':2})
  assert read_json(run/'status.json')['state']=='completed'
  request=read_json(run/'request.json');cfg=request['config']
  assert cfg['epochs']==100 and cfg['patience']==100 and cfg['lr']==.0001 and cfg['weight_decay']==0.
  assert cfg['max_instances'] is None and cfg['grad_clip'] is None
  protocol=read_json(run/'method_protocol.json')
  assert protocol['completed_epochs']==100 and protocol['input_dimension']==dimension
  assert protocol['settings']['k']==k and protocol['settings']['sigma']==.002
  assert protocol['settings']['dropout']==.25 and protocol['settings']['num_workers']==8
  assert protocol['settings']['scheduler_factor']==.8 and protocol['settings']['scheduler_patience']==5
  last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
  best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
  extra=last['model']['_extra_state'];n=len(bundle.records('train'))
  for level in ['low','mid','high']:
   assert last['model']['network.fc_'+level+'_mag.0.weight'].shape==(512,dimension)
   assert last['model']['network.ga_'+level+'_mag.attention_a.0.weight'].shape==(256,512)
  assert last['model']['network.classify_head.weight'].shape==(3,512)
  assert len(last['history'])==100 and extra['completed_epochs']==100 and extra['optimizer_updates']==100*n
  assert len(extra['train_orders'])==100 and all(len(order)==n for order in extra['train_orders'])
  assert all(set(order)<=set(row['slide_id'] for row in bundle.records('train')) for order in extra['train_orders'])
  assert extra['scheduler']['last_epoch']==100 and len(extra['lr_history'])==100
  assert all(torch.isfinite(value).all() and value.dtype==torch.float32 for name,value in last['model'].items() if torch.is_tensor(value))
  assert all(not torch.equal(before[name],last['model'][name]) for name in selectors)
  metric='zoommil_weighted_f1' if geometry=='bright' else 'zoommil_source_ce'
  select=max if geometry=='bright' else min
  assert best['epoch']==select(last['history'],key=lambda row:row['validation'][metric])['epoch']
  assert extra['recipe']['selection_metric']==metric
  updates=extra['optimizer_updates'];selected=best['epoch'];scheduler=extra['scheduler']
  lr_history=extra['lr_history'];orders_sha=protocol['train_orders_sha256']
  del last,best,extra;gc.collect();torch.cuda.empty_cache()
  metrics=evaluate(run,device='cuda');prediction=digest_file(run/'test_predictions.csv')
  assert evaluate(run,device='cuda')==metrics and digest_file(run/'test_predictions.csv')==prediction
  frame=pd.read_csv(run/'test_predictions.csv')
  assert frame.slide_id.is_unique and len(frame)==len(bundle.records('test'))
  report['profiles'].append({'state':'passed','profile':profile,'feature_dimension':dimension,'classes':3,
   'geometry':geometry,'nodes_per_level':expected,'K':k,'hidden':512,'attention_hidden':256,
   'Gaussian_samples':100,'num_workers':8,'epochs':100,'batch_size':1,'train_slides':n,
   'actual_optimizer_updates':updates,'auxiliary_selection_heads_both_updated':True,
   'optimizer':'source Adam','scheduler':scheduler,'learning_rate_history':lr_history,'train_orders_sha256':orders_sha,
   'selection_metric':metric,'selected_epoch':selected,'selected_checkpoint_reloaded_twice':True,
   'prediction_sha256':prediction,'metrics':metrics})
  save()
assert len(report['profiles'])==4 and report['software']==software_identity()
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):
  assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
report['state']='passed';save();write_json(out/'validation.json',report)
print(json.dumps({'state':'passed','profiles':4,'complete_native_epochs':400}),flush=True)
'''
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(item for item in env.get('PYTHONPATH', '').split(os.pathsep)
                                     if item and Path(item).resolve() != (Path(args.source_tree)/'src').resolve())
    with (out/'program.log').open('w') as log:
        subprocess.run([sys.executable, '-c', program, str(target), str(out)], cwd=out,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
