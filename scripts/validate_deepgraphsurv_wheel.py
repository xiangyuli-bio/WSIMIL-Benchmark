"""Independent installed-wheel AGCN, native64batch/500maxnode/100epoch budget."""
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
    parser.add_argument('--source-tree')
    parser.add_argument('--expected-version')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    target = output/'wheel_import'
    target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        members = [n for n in archive.namelist() if n.startswith('milplus/') and not n.endswith('/')]
        if args.source_tree:
            for name in members:
                assert archive.read(name) == (Path(args.source_tree)/'src'/name).read_bytes(), name
        archive.extractall(target)
    child = r'''
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch,milplus
from milplus.deepgraphsurv_demo import create_deepgraphsurv_demo
from milplus.data import Bundle
from milplus.engine import train,evaluate
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
target,output=map(Path,sys.argv[1:3]);expected=sys.argv[3]
assert Path(milplus.__file__).is_relative_to(target)
if expected:assert milplus.__version__==expected
identity=software_identity();runs=[]
for dimension in [1024,1536]:
 data=create_deepgraphsurv_demo(output/('data_'+str(dimension)),patients=400,dimension=dimension)
 bundle=Bundle(data);run=output/('deepgraphsurv_agcn_'+str(dimension))
 assert [len(bundle.deepgraphsurv.order[s]) for s in ['train','val','test']]==[240,80,80]
 train(data,'DeepGraphSurv',run,seed=2018,config={'threads':1})
 assert not (run/'test_predictions.csv').exists()
 request=read_json(run/'request.json');cfg=request['config']
 assert cfg['epochs']==100 and cfg['model']=={} and cfg['grad_clip'] is None
 assert cfg['lr']==.0002 and cfg['weight_decay']==0 and cfg['selection_metric']=='deepgraphsurv_val_patient_ci'
 last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
 best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
 extra=last['model']['_extra_state']
 assert extra['completed_epochs']==100 and extra['optimizer_steps']==300 and extra['training_done']
 assert len(last['history'])==100 and last['optimizers'][0]['param_groups'][0]['updates']==300
 assert len(last['optimizers'][0]['state'])==30
 assert last['model']['network.attention1.metric'].shape==(dimension,dimension)
 assert last['model']['network.backbone1.weight'].shape==(3*dimension,256)
 assert last['model']['network.backbone2.weight'].shape==(3*256,256)
 assert last['model']['network.backbone3.weight'].shape==(3*256,128)
 assert last['model']['network.fc1.weight'].shape==(128,64) and last['model']['network.fc2.weight'].shape==(64,1)
 scheduled=[r for r in last['history'] if r['validation'].get('validation_performed') is not False]
 assert [r['epoch'] for r in scheduled]==list(range(5,100,5))
 selected=max(scheduled,key=lambda r:r['validation']['deepgraphsurv_val_patient_ci'])['epoch']
 assert best['epoch']==selected
 protocol=read_json(run/'method_protocol.json')
 assert protocol['settings']['batch_size']==64 and protocol['settings']['max_nodes']==500
 assert protocol['settings']['backbone_channels']==[256,256,128]
 metrics=evaluate(run);sha=digest_file(run/'test_predictions.csv')
 assert evaluate(run)==metrics and digest_file(run/'test_predictions.csv')==sha
 runs.append({'profile':'deepgraphsurv_agcn','feature_dimension':dimension,'completed_epochs':100,
  'optimizer_updates':300,'train_patients':240,'validation_patients':80,'test_patients':80,
  'batch_size':64,'max_nodes':500,'actual_synthetic_nodes':[10,12],'selected_epoch':selected,
  'backbone_channels':[256,256,128],'attention_channels':[32,1],
  'prediction_sha256':sha,'metrics':metrics,'state':'passed'})
 write_json(output/'progress.json',runs)
 del last,best,bundle
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).is_relative_to(target),name
assert identity==software_identity()
write_json(output/'validation.json',{'state':'passed','version':milplus.__version__,'software':identity,
 'profiles':runs,'synthetic':True,'device':'cpu','import_root':str(target),
 'scope':'Two independent-wheel native adaptive graph networks with full1024/1536FM channels. Original64graph minibatches,500padding/selection cap,100epochs and300actualTF1Adam steps each. Actual small synthetic10..12node graphs, not500fake nodes. Original window/loss/LR/BN/every5epoch selection retained; explicit VAL-only/full-checkpoint and reproducible always-on-dropout evaluation corrections. No natural WSI/CUDA/paper-score claim.'})
'''
    env = {**os.environ, 'PYTHONPATH': str(target), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (output/'program.log').open('w') as log:
        result = subprocess.run([sys.executable, '-c', child, str(target), str(output), args.expected_version or ''],
            cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(json.dumps({'returncode': result.returncode, 'output': str(output), 'runtime_members': len(members)}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
