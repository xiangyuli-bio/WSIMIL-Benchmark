"""Actual independent-wheel full-D/C-way source-budget CoD training."""
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
    target = output/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        members = [n for n in archive.namelist() if n.startswith('milplus/') and not n.endswith('/')]
        if args.source_tree:
            for name in members:
                assert archive.read(name) == (Path(args.source_tree)/'src'/name).read_bytes(), name
        archive.extractall(target)
    child = r'''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch, milplus
from milplus.codmil_demo import create_codmil_demo
from milplus.data import Bundle
from milplus.engine import train, evaluate
from milplus.io import read_json, write_json, digest_file
from milplus.provenance import software_identity
target, output = map(Path, sys.argv[1:3]); expected = sys.argv[3]
assert Path(milplus.__file__).is_relative_to(target)
if expected: assert milplus.__version__ == expected
identity = software_identity(); records = []
for dimension, text_dimension, classes in [(1024, 1024, 2), (1536, 512, 3)]:
 data = create_codmil_demo(output/f'data_{dimension}', patients=10*classes,
     dimension=dimension, text_dimension=text_dimension, classes=classes)
 bundle = Bundle(data); run = output/f'codmil_source_repaired_{dimension}'
 train(data, 'CoD-MIL', run, seed=2024, config={'threads': 1})
 assert not (run/'test_predictions.csv').exists()
 request = read_json(run/'request.json'); cfg = request['config']
 assert cfg['model'] == {} and cfg['epochs'] == 200 and cfg['lr'] == .0001 and cfg['weight_decay'] == 1e-5
 assert cfg['selection_metric'] == 'codmil_val_loss' and cfg['patience'] == 20
 assert cfg['max_instances'] is None and cfg['grad_clip'] is None
 last = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
 best = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
 state, history = last['model'], last['history']; completed = len(history)
 extra = state['_extra_state']
 assert 52 <= completed <= 200 and last['epoch'] == completed-1 and extra['completed_epochs'] == completed
 assert extra['optimizer_updates'] == len(bundle.records('train'))*completed
 train_ids = {r['slide_id'] for r in bundle.records('train')}
 assert all(len(order) == len(train_ids) and set(order) == train_ids for order in extra['train_orders'])
 assert state['core.network.attention_block.attention_V.0.weight'].shape == (192, dimension)
 assert state['core.network.fc_text.0.weight'].shape == (dimension, text_dimension)
 assert state['core.network.fc_text.2.weight'].shape == (dimension, dimension)
 assert state['core.network.classifier.0.weight'].shape == (classes, dimension)
 assert state['core.network.self_attention.in_proj_weight'].shape == (3*dimension, dimension)
 assert state['text'].shape == (2*classes+3, text_dimension)
 selected = min(history, key=lambda row: row['validation']['codmil_val_loss'])['epoch']
 assert best['epoch'] == selected
 if completed < 200:
  assert last['bad'] >= 20 and last['epoch'] > 50
 for row in history:
  assert row['validation']['codmil_val_loss'] >= row['validation']['log_loss']-1e-6
 steps = {int(item['step']) for item in last['optimizers'][0]['state'].values()}
 assert steps == {extra['optimizer_updates']}
 assert last['optimizers'][0]['param_groups'][0]['lr'] == .0001
 protocol = read_json(run/'method_protocol.json')
 assert protocol['settings']['hidden'] == 192 and protocol['settings']['stop_epoch'] == 50
 assert protocol['settings']['optimizer'] == 'adam' and protocol['settings']['weighted_sampling'] is False
 assert protocol['source'] == '0fd748dc91ffd889bf13dbffd4db633b158809d6'
 assert protocol['feature_dimension'] == dimension and protocol['text_dimension'] == text_dimension
 assert protocol['classes'] == classes and protocol['completed_epochs'] == completed
 assert protocol['text_condition'] == ('matched_vlm' if dimension == text_dimension else 'learned_projection_extension')
 metrics = evaluate(run); sha = digest_file(run/'test_predictions.csv')
 assert evaluate(run) == metrics and digest_file(run/'test_predictions.csv') == sha
 records.append({'profile': 'codmil_source_repaired', 'feature_dimension': dimension,
    'text_dimension': text_dimension, 'classes': classes, 'hidden': 192, 'batch_size': 1,
    'max_epochs': 200, 'completed_epochs': completed, 'optimizer_updates': extra['optimizer_updates'],
    'train_slides': len(train_ids), 'original_patience': 20, 'original_stop_epoch': 50,
    'selected_epoch': selected, 'prediction_sha256': sha, 'metrics': metrics, 'state': 'passed'})
 write_json(output/'progress.json', records)
 del last, best, state, bundle
for name, module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module, '__file__', None):
  assert Path(module.__file__).is_relative_to(target), name
assert identity == software_identity()
write_json(output/'validation.json', {'state': 'passed', 'version': milplus.__version__,
 'software': identity, 'profiles': records, 'synthetic': True, 'device': 'cpu',
 'scope': 'Two actual independent-wheel native192gate full1024/1024/C2 and1536/512/C3 CoD runs, complete physical low/high feature+map+ordered class/normal text assets, full200maxepoch/Adam/patience20/epoch>50 source recipe. Actual epochs, all bag updates and source first-complete-VAL-loss selection checked, selected checkpoint reloaded and TEST predictions repeated. Synthetic small WSIs and explicitly named input/C-way repairs; no natural WSI/CUDA/paper-score reproduction claim.'})
'''
    env = {**os.environ, 'PYTHONPATH': str(target), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (output/'program.log').open('w') as log:
        result = subprocess.run([sys.executable, '-c', child, str(target), str(output), args.expected_version or ''],
            cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(json.dumps({'returncode': result.returncode, 'output': str(output), 'runtime_members': len(members)}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
