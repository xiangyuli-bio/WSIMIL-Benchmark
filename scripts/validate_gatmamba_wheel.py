"""Actual independent-wheel CPU training at full GAT-Mamba source budgets."""
import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-tree', help='Optionally byte-verify all wheel runtime members against this checkout')
    parser.add_argument('--expected-version')
    args = parser.parse_args()
    output = Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    target = output/'wheel_import';target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        members = [n for n in archive.namelist() if n.startswith('milplus/') and not n.endswith('/')]
        if args.source_tree:
            for name in members:
                if archive.read(name) != (Path(args.source_tree)/'src'/name).read_bytes():
                    raise ValueError('Wheel/source mismatch: '+name)
        archive.extractall(target)
    child = r'''
import sys,math
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import milplus,torch
from milplus.data import Bundle
from milplus.engine import train,evaluate
from milplus.gatmamba_demo import create_gatmamba_demo
from milplus.gatmamba import patients
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
target,output=map(Path,sys.argv[1:3]);version=sys.argv[3]
assert Path(milplus.__file__).is_relative_to(target)
if version:assert milplus.__version__==version
identity=software_identity();results=[]
for dimension in [1024,1536]:
 data=create_gatmamba_demo(output/('data_'+str(dimension)),dimension=dimension,patients=20)
 bundle=Bundle(data)
 for profile in ['gatmamba_source','gatmamba_edge_conditioned']:
  run=output/(profile+'_'+str(dimension))
  train(data,profile,run,config={'threads':1})
  assert not (run/'test_predictions.csv').exists()
  request=read_json(run/'request.json');cfg=request['config']
  assert cfg['epochs']==201 and cfg['patience']==5 and cfg['model']=={} and cfg['grad_clip'] is None
  assert cfg['lr']==5e-5 and cfg['weight_decay']==1e-4
  last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
  best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
  extra=last['model']['_extra_state'];epochs=extra['completed_epochs']
  assert epochs==len(last['history']) and 7<=epochs<=201
  assert extra['optimizer_steps']==epochs*math.ceil(len(patients(bundle,'train'))/16)
  assert extra['scheduler']['last_epoch']==epochs and extra['scheduler']['patience']==5
  assert last['model']['network.uni_feature_linear_transform.weight'].shape==(64,dimension)
  assert last['model']['network.layers.0.self_attn.in_proj.weight'].shape==(160,80)
  assert last['model']['network.layers.0.self_attn.conv1d.weight'].shape==(80,1,4)
  values=[row['validation']['gatmamba_val_loss'] for row in last['history']]
  selected=max(i for i,value in enumerate(values) if value==min(values));assert best['epoch']==selected
  protocol=read_json(run/'survival_protocol.json')
  assert protocol['source_inclusive_epoch_argument']==200 and protocol['settings']['backend']['name']=='torch_reference'
  assert protocol['settings']['batch_size']==16 and protocol['settings']['weight_regu']==1e-5
  metrics=evaluate(run);sha=digest_file(run/'test_predictions.csv')
  assert metrics==evaluate(run) and sha==digest_file(run/'test_predictions.csv')
  results.append({'profile':profile,'feature_dimension':dimension,'uni_projection':64,'position_dimension':16,
   'GAT_Mamba_channels':80,'state_dimension':16,'blocks':1,'maximum_epochs':201,
   'completed_epochs':epochs,'optimizer_updates':extra['optimizer_steps'],'selected_epoch':selected,
   'prediction_sha256':sha,'metrics':metrics,'state':'passed'})
  write_json(output/'progress.json',results)
  del last,best
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).is_relative_to(target),name
assert identity==software_identity()
write_json(output/'validation.json',{'state':'passed','version':milplus.__version__,'software':identity,
 'profiles':results,'device':'cpu','synthetic':True,'import_root':str(target),
 'scope':'Independent extracted wheel. Four native64projection/16position/80channel/state16/oneblock complete GAT-Mamba source and named-edge-corrected training runs across1024/1536FM widths. Default201inclusive maximum,16patient batches,Cox+L1,Adam/plateau/last-tie selection and6bad-epoch stopping retained. Selected checkpoints actually reloaded and predictions repeated. Explicit same-version author CPU scan reference; no native CUDA, natural WSI or paper-number claim.'})
'''
    environment = {**os.environ, 'PYTHONPATH': str(target), 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
    with (output/'program.log').open('w') as log:
        result = subprocess.run([sys.executable,'-c',child,str(target),str(output),args.expected_version or ''],
            cwd=output,env=environment,stdout=log,stderr=subprocess.STDOUT,check=False)
    print(json.dumps({'returncode':result.returncode,'output':str(output),'runtime_members':len(members)}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
