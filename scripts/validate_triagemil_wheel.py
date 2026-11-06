"""Actual independent-wheel TRIAGE-MIL native networks and original training budgets."""
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
import sys,math
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import milplus,torch
from milplus.triagemil_demo import create_triagemil_demo
from milplus.data import Bundle
from milplus.engine import train,evaluate
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
target,output=map(Path,sys.argv[1:3]);version=sys.argv[3]
assert Path(milplus.__file__).is_relative_to(target)
if version:assert milplus.__version__==version
identity=software_identity();runs=[]
for dimension in [1024,1536]:
 data=create_triagemil_demo(output/('data_'+str(dimension)),patients=20,dimension=dimension)
 bundle=Bundle(data);run=output/('triagemil_source_'+str(dimension))
 train(data,'TRIAGE-MIL',run,seed=35,config={'threads':1})
 assert not (run/'test_predictions.csv').exists()
 request=read_json(run/'request.json');cfg=request['config']
 assert cfg['epochs']==200 and cfg['patience']==40 and cfg['model']=={} and cfg['grad_clip'] is None
 assert cfg['lr']==2e-4 and cfg['weight_decay']==1e-4 and cfg['selection_metric']=='triagemil_selection'
 last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
 best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
 extra=last['model']['_extra_state'];epochs=extra['completed_epochs'];steps=extra['optimizer_steps']
 assert 60<=epochs<=200 and len(last['history'])==epochs
 assert 0<steps<=epochs*math.ceil(len(bundle.triagemil.order['train'])/4)
 assert max(int(s['step']) for s in last['optimizers'][0]['state'].values())==steps
 assert last['model']['network.theta_layers.0.weight'].shape==(512,dimension)
 assert last['model']['network.theta_layers.3.weight'].shape==(512,dimension)
 assert last['model']['network.fuse.weight'].shape==(dimension,6*dimension)
 assert last['model']['network.attention_temp'].shape==(6,)
 values=[row['validation']['triagemil_selection'] for row in last['history']]
 selected=values.index(max(values));assert max(values)>=0 and best['epoch']==selected
 bad=epochs-1-selected;assert extra['bad_epochs']==bad
 assert extra['training_done']==(epochs>=60 and bad>=40)
 protocol=read_json(run/'method_protocol.json')
 assert protocol['settings']['batch_size']==4 and protocol['settings']['accumulation_steps']==1
 assert protocol['settings']['hypergraph_layers']==4 and protocol['settings']['hidden']==512
 assert protocol['preprocessing']['selection_budget']==4096
 assert protocol['preprocessing']['hypergraph']=={'k_intra':10,'k_inter':5,'tiles_per_super':12,'random_state':35}
 metrics=evaluate(run);sha=digest_file(run/'test_predictions.csv')
 assert metrics==evaluate(run) and sha==digest_file(run/'test_predictions.csv')
 runs.append({'profile':'triagemil_source','feature_dimension':dimension,'hidden':512,'heads':6,
  'hypergraph_layers':4,'requested_MASS_budget':4096,'maximum_epochs':200,'minimum_epochs':60,
  'completed_epochs':epochs,'optimizer_updates':steps,'selected_epoch':selected,
  'prediction_sha256':sha,'metrics':metrics,'state':'passed'})
 write_json(output/'progress.json',runs)
 del last,best
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).is_relative_to(target),name
assert identity==software_identity()
write_json(output/'validation.json',{'state':'passed','version':milplus.__version__,'software':identity,
 'profiles':runs,'device':'cpu','synthetic':True,'import_root':str(target),
 'scope':'Independent extracted wheel. Original512hidden/sixhead/fourlayer complete TRIAGE-MIL with actual MASS quality/fouraxis/FPS selection and source semantic hypergraphs across1024/1536FM widths.200max/min60epochs,AdamW/batch4/Cox/noevent skips,strict-first-tieVALselection/40bad stopping preserved. Selected weights actually reloaded and predictions repeated. CPU synthetic only; no CUDA/naturalWSI/paper-score claim.'})
'''
    environment = {**os.environ, 'PYTHONPATH': str(target), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (output/'program.log').open('w') as log:
        result = subprocess.run([sys.executable, '-c', child, str(target), str(output), args.expected_version or ''],
            cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(json.dumps({'returncode': result.returncode, 'output': str(output), 'runtime_members': len(members)}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
