"""Independent installed-wheel native SCL paper and source-control budgets."""
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
sys.path.insert(0, sys.argv[1])
import torch, milplus
from milplus.data import Bundle
from milplus.engine import train, evaluate
from milplus.io import read_json, write_json, digest_file
from milplus.provenance import software_identity
from milplus.sclwc_demo import create_sclwc_demo
target, output = map(Path, sys.argv[1:3]); expected = sys.argv[3]
assert Path(milplus.__file__).is_relative_to(target)
if expected: assert milplus.__version__ == expected
identity = software_identity(); records = []
for dimension in [1024, 1536]:
 for profile, classes in [('sclwc_paper_reference', 3), ('sclwc_literal_source_control', 2)]:
  data = create_sclwc_demo(output/f'data_{profile}_{dimension}', patients=10*classes, dimension=dimension, classes=classes)
  bundle = Bundle(data); run = output/f'{profile}_{dimension}'
  train(data, profile, run, seed=2022, config={'threads': 1})
  assert not (run/'test_predictions.csv').exists()
  request = read_json(run/'request.json'); cfg = request['config']
  assert cfg['epochs'] == 100 and cfg['model'] == {} and cfg['grad_clip'] is None and cfg['max_instances'] is None
  assert cfg['lr'] == (.001 if profile == 'sclwc_paper_reference' else .0003)
  assert cfg['weight_decay'] == 1e-5 and cfg['patience'] == 20 and cfg['selection_metric'] == 'macro_auc'
  last = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
  best = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
  extra = last['model']['_extra_state']; history = last['history']
  completed = len(history)
  assert 32 <= completed <= 100 and extra['completed_epochs'] == completed
  assert extra['optimizer_updates'] == completed*len(bundle.records('train'))
  assert not extra['training_done'] or extra['bad'] >= 20 and last['epoch'] > 30
  selected = max(history, key=lambda r: r['validation']['macro_auc'])['epoch']
  assert best['epoch'] == selected
  step_counts = {int(s['step']) for s in last['optimizers'][0]['state'].values()}
  # Class-specific instance heads update only when their class is sampled.
  # Shared layers use every step; do not invent a global Adam clock.
  assert max(step_counts) == extra['optimizer_updates'] and min(step_counts) > 0
  if profile == 'sclwc_paper_reference':
   assert last['model']['network.projector.0.weight'].shape == (512, dimension)
   assert last['model']['network.attention.attention_a.0.weight'].shape == (256, 512)
   assert last['model']['network.pnm_gate.fc1.weight'].shape == (256, 512)
   assert all(r['train']['pnm_active'] == 1 for r in history)
   assert any(r['train']['wscl_active'] > 0 and r['train']['wscl'] > 0 for r in history[5:])
   for entries in last['model']['network._extra_state']['banks'].values():
    assert entries
    for entry in entries:
     assert entry['slide_id'] in {r['slide_id'] for r in bundle.records('train')}
     assert entry['features'].shape == (8, 512) and not entry['features'].requires_grad
  else:
   assert last['model']['network.attention_net.0.weight'].shape == (512, dimension)
   assert all(r['train']['wscl'] == 0 and r['train']['pnm_active'] == 0 and r['train']['forward_epoch'] == 0 for r in history)
  protocol = read_json(run/'method_protocol.json')
  assert protocol['hidden'] == 512 and protocol['attention_dim'] == 256
  assert protocol['settings']['cosine_t_max'] == 50 and protocol['settings']['stop_epoch'] == 30
  assert protocol['complete_paper_objectives'] == (profile == 'sclwc_paper_reference')
  metrics = evaluate(run); sha = digest_file(run/'test_predictions.csv')
  assert evaluate(run) == metrics and digest_file(run/'test_predictions.csv') == sha
  records.append({'profile': profile, 'feature_dimension': dimension, 'classes': classes,
   'hidden': 512, 'attention_dim': 256, 'batch_size': 1, 'max_epochs': 100,
   'completed_epochs': completed, 'optimizer_updates': extra['optimizer_updates'],
   'original_patience': 20, 'original_stop_epoch': 30, 'selected_epoch': selected,
   'train_slides': len(bundle.records('train')), 'validation_slides': len(bundle.records('val')),
   'test_slides': len(bundle.records('test')), 'prediction_sha256': sha,
   'adam_parameter_step_counts': sorted(step_counts), 'metrics': metrics, 'state': 'passed'})
  write_json(output/'progress.json', records)
  del last, best, bundle
for name, module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module, '__file__', None):
  assert Path(module.__file__).is_relative_to(target), name
assert identity == software_identity()
write_json(output/'validation.json', {'state': 'passed', 'version': milplus.__version__,
 'software': identity, 'profiles': records, 'synthetic': True, 'device': 'cpu',
 'scope': 'Four actual independent-wheel native512/256 full1024/1536FM runs. Complete paper CDA/PNM/live-anchorWSCL, and separately named incomplete literal source controls. Original100epoch maximum, one-bag balanced sampling, Adam/cosine/first-bestVAL/minimum-epoch early stop retained; actual epochs/updates recorded. Small synthetic bags, no natural WSI/CUDA/SSL pretraining/paper-score claim.'})
'''
    env = {**os.environ, 'PYTHONPATH': str(target), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (output/'program.log').open('w') as log:
        result = subprocess.run([sys.executable, '-c', child, str(target), str(output), args.expected_version or ''],
            cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(json.dumps({'returncode': result.returncode, 'output': str(output), 'runtime_members': len(members)}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
