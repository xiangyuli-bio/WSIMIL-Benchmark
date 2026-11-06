"""Actual independent-wheel PAMIL native512/300epoch complete training budgets."""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-tree', required=True)
    parser.add_argument('--expected-version', default='0.62.0')
    args = parser.parse_args()
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    target = out/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith('milplus/') and not name.endswith('/'):
                assert archive.read(name) == (Path(args.source_tree)/'src'/name).read_bytes(), name
        archive.extractall(target)
    code = r'''
import sys, math
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch, milplus
from milplus.pamil_policy_demo import create_pamil_policy_demo
from milplus.pamil_policy_network import PAPER,SOURCE
from milplus.data import Bundle
from milplus.engine import train,evaluate
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
target,out=map(Path,sys.argv[1:3]);assert Path(milplus.__file__).is_relative_to(target)
assert milplus.__version__==sys.argv[3]
identity=software_identity();results=[];torch.set_num_threads(1)
for dimension,classes in [(1024,2),(1536,3)]:
 data=create_pamil_policy_demo(out/f'data_{dimension}',patients=10*classes,dimension=dimension,classes=classes,rows_per_slide=1536)
 bundle=Bundle(data);count=len(bundle.records('train'))
 for profile in [PAPER,SOURCE]:
  run=out/(profile+'_'+str(dimension))
  train(data,profile,run,seed=2024,config={'threads':1})
  assert not (run/'test_predictions.csv').exists()
  cfg=read_json(run/'request.json')['config'];assert cfg['model']=={}
  assert cfg['epochs']==300 and cfg['lr']==1e-4 and cfg['weight_decay']==1e-5
  assert cfg['grad_clip'] is None and cfg['max_instances'] is None and cfg['selection_metric']=='macro_auc'
  last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
  best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
  state,history=last['model'],last['history'];extra=state['_extra_state']
  assert len(history)==300 and last['epoch']==299 and extra['completed_epochs']==300 and extra['total_epochs']==300
  assert extra['main_updates']==count*300 and extra['policy_updates']==count*300*3 and extra['total_rollout_steps']==count*300*3
  allowed={r['slide_id'] for r in bundle.records('train')}
  assert all(len(order)==count and set(order)==allowed for order in extra['train_orders'])
  assert state['core.base.fc1.weight'].shape==(512,dimension)
  assert state['core.base.basedlayers.1.GroupsAttenLayer.AttenLayer.attn.to_qkv.weight'].shape==(1536,512)
  assert state['core.head.fc2.weight'].shape==(classes,512)
  assert state['core.policy.gru.weight_ih_l0'].shape==(1536,512)
  assert state['core.policy.actor.0.weight'].shape==(512,512)
  assert state['core.policy.action_var'].shape==(512,)
  for name,value in state.items():
   if name.startswith('core.policy.'):
    assert torch.equal(value,state[name.replace('core.policy.','core.policy_old.',1)])
  assert len(last['optimizers'])==2
  assert all(int(s['step'])==count*300*3 for s in last['optimizers'][1]['state'].values())
  assert all(g['lr']==1e-4 and g['weight_decay']==1e-5 for g in last['optimizers'][0]['param_groups'])
  protocol=read_json(run/'method_protocol.json');settings=protocol['settings']
  assert settings['threshold_schedule']=='train_bag_iteration'
  assert settings['bag_size']==512 and settings['layers']==2 and protocol['native_hidden']==512 and protocol['policy_hidden']==512
  assert settings['policy_epochs']==3 and settings['policy_gamma']==.1 and settings['policy_lr']==1e-5
  assert settings['action_std']==.5 and settings['eps_clip']==.2 and settings['scheme']=='liis'
  for audit in extra['last_epoch_audits']:
   assert audit['original_rows']==1536 and audit['sampled_rows']==1536 and audit['steps']==3
   assert 0<audit['unique_rows']<=1536
   if profile==PAPER:assert audit['unique_rows']==1536
  for row in history:
   epoch=row['epoch'];alpha=.5*(1+math.cos(math.pi*epoch/300));tr=row['train']
   assert all(k in tr for k in ['loss','wsl','stl','sia'])
   assert math.isclose(tr['loss'],tr['wsl']+.1*(1-alpha)*tr['stl']+.5*alpha*tr['sia'],rel_tol=1e-5,abs_tol=1e-5)
  selected=max(history,key=lambda r:r['validation']['macro_auc'])['epoch'];assert best['epoch']==selected
  del last,best,state
  metrics=evaluate(run);sha=digest_file(run/'test_predictions.csv')
  assert evaluate(run)==metrics and digest_file(run/'test_predictions.csv')==sha
  results.append({'profile':profile,'feature_dimension':dimension,'classes':classes,'hidden':512,'attention_layers':2,
   'policy_hidden':512,'action_size':512,'bag_size':512,'original_rows_per_slide':1536,'epochs':300,'train_slides':count,
   'main_updates':extra['main_updates'],'policy_optimizer_updates':extra['policy_updates'],
   'rollout_steps':extra['total_rollout_steps'],'selected_epoch':selected,'prediction_sha256':sha,'metrics':metrics,'state':'passed'})
  write_json(out/'progress.json',{'state':'running','profiles':results})
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).is_relative_to(target),name
write_json(out/'validation.json',{'state':'passed','version':milplus.__version__,'software':identity,'profiles':results,
 'synthetic':True,'device':'cpu','scope':'Four actual independent-wheel complete300epoch/AdaMax/native512two-layer/512group/GRU512/PPOK3 runs: full1024/C2 and1536/C3, paperreference and releasedsamplingreference-training controls. Complete WSL/STL/stopgradSIA/actualTRAINreward and both optimizers, no truncatedepochs. Actual selected-checkpoint reload and repeated TEST prediction. Full unpublished training choices declared; not original numerical paper reproduction, natural WSI or CUDA proof.'})
'''
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(p for p in env.get('PYTHONPATH', '').split(os.pathsep)
        if p and Path(p).resolve() != (Path(args.source_tree)/'src').resolve())
    with (out/'program.log').open('w') as log:
        subprocess.run([sys.executable, '-c', code, str(target), str(out), args.expected_version],
            cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
