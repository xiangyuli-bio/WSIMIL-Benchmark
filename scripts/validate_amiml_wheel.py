"""Actual independent-wheel full AMIML four-cluster/nativeR100/batch64 budgets."""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--source-tree', required=True); parser.add_argument('--expected-version', default='0.64.0')
    args = parser.parse_args(); out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    target = out/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(args.wheel) as wheel:
        for name in wheel.namelist():
            if name.startswith('milplus/') and not name.endswith('/'):
                assert wheel.read(name) == (Path(args.source_tree)/'src'/name).read_bytes(), name
        wheel.extractall(target)
    program = r'''
import sys,math
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch,milplus
from milplus.amiml_demo import create_amiml_demo
from milplus.amiml_data import patient_bags
from milplus.amiml_network import PAPER,SOURCE
from milplus.data import Bundle
from milplus.engine import train,evaluate
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
target,out=map(Path,sys.argv[1:3]);assert milplus.__version__==sys.argv[3]
assert Path(milplus.__file__).is_relative_to(target)
torch.set_num_threads(1);identity=software_identity();results=[]
for dimension,classes in [(256,2),(1024,3),(1536,2)]:
 data=create_amiml_demo(out/f'data_{dimension}',patients=120,dimension=dimension,classes=classes,per_cluster=104)
 bundle=Bundle(data);groups=patient_bags(bundle,'train');assert len(groups)>=64
 for profile in ([PAPER,SOURCE] if dimension==256 else [PAPER]):
  run=out/(profile+'_'+str(dimension));train(data,profile,run,seed=2024,config={'threads':1})
  assert not (run/'test_predictions.csv').exists()
  cfg=read_json(run/'request.json')['config']
  assert cfg['model']=={} and cfg['epochs']==300 and cfg['patience']==30
  assert cfg['lr']==(.01 if profile==PAPER else .001)
  assert cfg['weight_decay']==0 and cfg['grad_clip'] is None and cfg['max_instances'] is None
  assert cfg['selection_metric']=='amiml_validation_generation'
  last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
  best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
  state,history=last['model'],last['history'];extra=state['_extra_state'];count=len(history)
  assert 6<=count<=300 and last['epoch']==count-1==best['epoch'] and extra['completed_epochs']==count
  assert bool(state['prepared']) and state['centers'].shape==(4,dimension)
  assert extra['cluster_fit']['patients']==len(groups) and extra['cluster_fit']['patches']>len(groups)*400
  assert len(extra['best_meta'])==len(extra['branch_rng'])==len(extra['schedulers'])==4
  allowed={g[0]['global_patient_id'] for g in groups}
  for c in range(4):
   epochs=sum(bool(orders[c]) for orders in extra['train_orders'])
   assert all(not order[c] or len(order[c])==len(allowed) and set(order[c])==allowed for order in extra['train_orders'])
   assert extra['optimizer_steps'][c]==epochs*math.ceil(len(groups)/64)
   assert extra['best_meta'][c]['epoch']>=5
   if epochs<300:assert extra['stopped'][c] and extra['bad'][c]>=30
   assert extra['schedulers'][c]['last_epoch']==epochs
   assert extra['schedulers'][c]['patience']==(10 if profile==PAPER else 5)
   prefix=f'networks.{c}.'
   assert state[prefix+'conv1.weight'].shape==(128,dimension,1)
   assert state[prefix+'conv2.weight'].shape==(64,128,1) and state[prefix+'conv3.weight'].shape==(32,64,1)
   assert state[prefix+'conv4.weight'].shape==(8,32,1) and state[prefix+'conv5.weight'].shape==(1,8,1)
   assert state[prefix+'conv6.weight'].shape==(1,8,1) and state[prefix+'query.weight'].shape==(8,8)
   assert state[prefix+'value.weight'].shape==(classes,8)
   assert len(last['optimizers'])==4
   assert all(int(v['step'])==extra['optimizer_steps'][c] for v in last['optimizers'][c]['state'].values())
  selected=max(range(4),key=lambda c:(extra['best_meta'][c]['auc'],-c))
  assert selected==extra['selected_cluster']
  p=read_json(run/'method_protocol.json');assert p['settings']['R']==100 and p['settings']['batch_size']==64
  assert p['settings']['latent']==8 and p['settings']['alpha']==.1
  assert p['settings']['morphology_clusters']==4
  del last,best,state
  metrics=evaluate(run);sha=digest_file(run/'test_predictions.csv')
  assert evaluate(run)==metrics and digest_file(run/'test_predictions.csv')==sha
  assert evaluate(run,'val')['amiml_selected_cluster']==selected
  results.append({'state':'passed','profile':profile,'feature_dimension':dimension,'classes':classes,
   'R':100,'latent':8,'batch_size':64,'morphology_clusters':4,'epochs':count,
   'train_patients':len(groups),'branch_optimizer_steps':extra['optimizer_steps'],
   'branch_best_epochs':[m['epoch'] for m in extra['best_meta']],'selected_cluster':selected,
   'prediction_sha256':sha,'metrics':metrics})
  write_json(out/'progress.json',{'state':'running','profiles':results})
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).is_relative_to(target),name
write_json(out/'validation.json',{'state':'passed','version':milplus.__version__,'software':identity,'profiles':results,
 'synthetic':True,'device':'cpu','scope':'Four complete installed-wheel/default300max/nativeR100/batch64/128-64-32-8 AMIML budgets:256binary paper+source,1024/C3 and1536/C2 paper. Every budget trains all four TRAIN-only morphology clusters, independent Adam/schedules/stopping/seed-streams, full patient bags, VAL-only branch/checkpoint selection and repeated selected reload. Public source broken/loader absent; named output/input/training repairs, no naturalWSI/CUDA/historical-score claim.'})
'''
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(p for p in env.get('PYTHONPATH', '').split(os.pathsep)
        if p and Path(p).resolve() != (Path(args.source_tree)/'src').resolve())
    with (out/'program.log').open('w') as log:
        subprocess.run([sys.executable, '-c', program, str(target), str(out), args.expected_version],
                       cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__': main()
