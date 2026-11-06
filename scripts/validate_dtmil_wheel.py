"""Actual independent-wheel full DT-MIL50epoch/native60map/100query budgets."""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--source-tree',required=True);parser.add_argument('--expected-version',default='0.63.0')
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    target=out/'wheel_import';target.mkdir()
    with zipfile.ZipFile(args.wheel) as package:
        for name in package.namelist():
            if name.startswith('milplus/') and not name.endswith('/'):
                assert package.read(name)==(Path(args.source_tree)/'src'/name).read_bytes(),name
        package.extractall(target)
    program=r'''
import sys,math
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch,milplus
from milplus.dtmil_demo import create_dtmil_demo
from milplus.dtmil_network import SOURCE,REPAIRED
from milplus.data import Bundle
from milplus.engine import train,evaluate
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
target,out=map(Path,sys.argv[1:3]);assert milplus.__version__==sys.argv[3]
assert Path(milplus.__file__).is_relative_to(target)
torch.set_num_threads(1);identity=software_identity();results=[]
for d,c in [(1280,2),(1024,3),(1536,3)]:
 data=create_dtmil_demo(out/f'data_{d}',patients=20*c,dimension=d,classes=c,grid_shape=(60,60),cached_views=False)
 bundle=Bundle(data);train_count=len(bundle.records('train'));assert train_count>=20
 for profile in ([SOURCE,REPAIRED] if d==1280 else [REPAIRED]):
  run=out/(profile+'_'+str(d));train(data,profile,run,seed=2024,config={'threads':1})
  assert not (run/'test_predictions.csv').exists()
  cfg=read_json(run/'request.json')['config']
  assert cfg['model']=={} and cfg['epochs']==50 and cfg['patience']==50
  assert cfg['lr']==2e-4 and cfg['weight_decay']==1e-4 and cfg['grad_clip']==.1
  assert cfg['max_instances'] is None and cfg['selection_metric']=='log_loss'
  last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
  best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
  state,history=last['model'],last['history'];extra=state['_extra_state']
  assert len(history)==50 and last['epoch']==49 and extra['completed_epochs']==50
  assert extra['optimizer_steps']==50*math.ceil(train_count/20)
  allowed={r['slide_id'] for r in bundle.records('train')}
  assert len(extra['train_orders'])==50 and all(len(order)==train_count and set(order)==allowed for order in extra['train_orders'])
  assert extra['scheduler']['last_epoch']==50 and extra['scheduler']['step_size']==40 and extra['scheduler']['gamma']==.1
  assert state['network.backbone.0.conv.1.weight'].shape==(256,d,1,1)
  assert state['network.query_embed.weight'].shape==(100,256)
  assert state['network.class_embed.0.weight'].shape==(c,100*256)
  assert state['network.transformer.encoder.layers.5.self_attn.sampling_offsets.weight'].shape==(8*4*2,256)
  assert state['network.transformer.decoder.layers.5.cross_attn.in_proj_weight'].shape==(768,256)
  protocol=read_json(run/'method_protocol.json');settings=protocol['settings']
  assert settings['map_size']==60 and settings['hidden']==256 and settings['queries']==100
  assert settings['encoder_layers']==settings['decoder_layers']==6 and settings['batch_size']==20
  assert settings['heads']==8 and settings['points']==4 and settings['feedforward']==1024
  assert settings['operator_backend']=='exact_author_torch_reference'
  for audit in extra['last_epoch_audits']:
   assert audit['original_rows']==3600 and audit['retained_rows']==3600
  assert best['epoch']==min(history,key=lambda row:row['validation']['log_loss'])['epoch']
  optimizer=last['optimizers'][0];assert len(optimizer['param_groups'])==3
  assert [g['group_name'] for g in optimizer['param_groups']]==['main','backbone','projection']
  rates=[2e-4,2e-5,2e-4*.1]
  assert all(math.isclose(g['lr'],r*.1,rel_tol=1e-15,abs_tol=0) for g,r in zip(optimizer['param_groups'],rates))
  assert all(int(v['step'])==50*math.ceil(train_count/20) for v in optimizer['state'].values())
  del last,best,state,optimizer
  metrics=evaluate(run);sha=digest_file(run/'test_predictions.csv')
  assert evaluate(run)==metrics and digest_file(run/'test_predictions.csv')==sha
  results.append({'state':'passed','profile':profile,'feature_dimension':d,'classes':c,'map_size':60,'hidden':256,
   'heads':8,'points':4,'queries':100,'encoder_layers':6,'decoder_layers':6,'feedforward':1024,
   'batch_size':20,'epochs':50,'train_slides':train_count,'optimizer_updates':extra['optimizer_steps'],
   'selected_epoch':min(history,key=lambda row:row['validation']['log_loss'])['epoch'],
   'prediction_sha256':sha,'metrics':metrics})
  write_json(out/'progress.json',{'state':'running','profiles':results})
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).is_relative_to(target),name
write_json(out/'validation.json',{'state':'passed','version':milplus.__version__,'software':identity,'profiles':results,
 'synthetic':True,'device':'cpu','scope':'Four actual independent-wheel full50epoch/native60x60/256hidden/6encoder6decoder/100query/8head/4point/batch20 reference runs; full1280/C2 sourcecrop and corrected,1024/C3 corrected,1536/C3 corrected. Complete AdamW LRgroups/clipping/StepLR/physical augmentation/RNG and firstVAL selected reload. Public trainer missing, reference decisions explicit; no naturalWSI/CUDA/original paper-score claim.'})
'''
    env=dict(os.environ)
    env['PYTHONPATH']=os.pathsep.join(p for p in env.get('PYTHONPATH','').split(os.pathsep)
        if p and Path(p).resolve()!=(Path(args.source_tree)/'src').resolve())
    with (out/'program.log').open('w') as log:
        subprocess.run([sys.executable,'-c',program,str(target),str(out),args.expected_version],
                       cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
