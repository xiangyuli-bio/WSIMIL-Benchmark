"""Train and reload independent C²MIL wheels at original100epoch/native-width budgets."""
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
    output = Path(args.output).resolve();output.mkdir(parents=True, exist_ok=False)
    target = output/'wheel_import';target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        members = [n for n in archive.namelist() if n.startswith('milplus/') and not n.endswith('/')]
        if args.source_tree:
            for name in members:
                if archive.read(name) != (Path(args.source_tree)/'src'/name).read_bytes():
                    raise ValueError('Wheel/source mismatch: '+name)
        archive.extractall(target)
    child = r'''
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import milplus,torch
from milplus.c2mil import source_batch_boundaries
from milplus.c2mil_demo import create_c2mil_demo
from milplus.data import Bundle
from milplus.engine import train,evaluate
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
target,output=map(Path,sys.argv[1:3]);version=sys.argv[3]
assert Path(milplus.__file__).is_relative_to(target)
if version:assert milplus.__version__==version
identity=software_identity();runs=[]
for dimension in [1024,1536]:
 data=create_c2mil_demo(output/('data_'+str(dimension)),patients=30,dimension=dimension)
 bundle=Bundle(data);run=output/('c2mil_source_'+str(dimension))
 train(data,'C2MIL',run,seed=8,config={'threads':1})
 assert not (run/'test_predictions.csv').exists()
 request=read_json(run/'request.json');cfg=request['config']
 assert cfg['epochs']==100 and cfg['model']=={} and cfg['grad_clip'] is None
 assert cfg['lr']==1e-4 and cfg['weight_decay']==5e-4 and cfg['selection_metric']=='c2mil_val_c_index'
 last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
 best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
 extra=last['model']['_extra_state'];epochs=extra['completed_epochs']
 assert epochs==100 and len(last['history'])==100
 steps=100*len(source_batch_boundaries(len(bundle.c2mil.order['train']),16))
 assert extra['optimizer_steps']==steps
 assert extra['current_lr']==2.5e-5 and last['optimizers'][0]['param_groups'][0]['lr']==2.5e-5
 assert set(extra['training_order'])==set(bundle.c2mil.order['train'])
 assert last['model']['network.graphcausal.fc.0.weight'].shape==(128,dimension)
 assert last['model']['network.biasremoval.thumb_encoder.fc1.weight'].shape==(512,dimension)
 assert last['model']['network.biasremoval.thumb_encoder.fc2.weight'].shape==(1024,512)
 assert extra['centers'].shape==(extra['best_k'],1024) and extra['biases'].shape==(extra['best_k'],dimension)
 values=[row['validation']['c2mil_val_c_index'] for row in last['history']]
 selected=max(i for i,value in enumerate(values) if i>=2 and value==max(values[2:]))
 assert best['epoch']==selected
 assert all(all(k in row['train'] for k in ['full_cox','causal_cox','causal_contrastive',
     'causal_ratio','old_new_alignment']) for row in last['history'])
 protocol=read_json(run/'survival_protocol.json')
 assert protocol['settings']['batch_size']==16 and protocol['settings']['reference_patches']==1500
 assert protocol['settings']['cluster_max_iterations']==400 and protocol['settings']['gcn_layers']==4
 metrics=evaluate(run);sha=digest_file(run/'test_predictions.csv')
 assert metrics==evaluate(run) and sha==digest_file(run/'test_predictions.csv')
 runs.append({'profile':'c2mil_source','feature_dimension':dimension,'graph_hidden':128,
  'gcn_layers':4,'thumbnail_hidden':512,'cluster_dimension':1024,'reference_patches':1500,
  'maximum_epochs':100,'completed_epochs':epochs,'optimizer_updates':steps,
  'selected_epoch':selected,'prediction_sha256':sha,'metrics':metrics,'state':'passed'})
 write_json(output/'progress.json',runs)
 del last,best
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).is_relative_to(target),name
assert identity==software_identity()
write_json(output/'validation.json',{'state':'passed','version':milplus.__version__,'software':identity,
 'profiles':runs,'device':'cpu','synthetic':True,'import_root':str(target),
 'scope':'Independent extracted wheel. Original100epoch,128hidden/fourGCN and512/1024thumbnail/clustering networks, real1500row TRAIN-reference sampler, all five objectives, batch16 final-merge/Adam/40epoch halvings/third-epoch last-tie VAL selection. Selected dynamic-cache checkpoints actually reloaded and predictions repeated across1024/1536FM widths. Synthetic CPU only; no naturalWSI/CUDA/paper-number claim.'})
'''
    environment = {**os.environ, 'PYTHONPATH': str(target), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (output/'program.log').open('w') as log:
        result = subprocess.run([sys.executable, '-c', child, str(target), str(output), args.expected_version or ''],
            cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(json.dumps({'returncode': result.returncode, 'output': str(output), 'runtime_members': len(members)}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
