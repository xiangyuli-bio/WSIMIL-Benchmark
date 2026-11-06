"""Actual independent-wheel full pretrained FOCUS and source-budget training."""
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
    parser.add_argument('--expected-version', default='0.61.0')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    target = output/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith('milplus/') and not name.endswith('/'):
                assert archive.read(name) == (Path(args.source_tree)/'src'/name).read_bytes(), name
        archive.extractall(target)
    child = r'''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch, milplus
from milplus.focus_demo import create_focus_demo
from milplus.focus_network import WEIGHTS_SHA
from milplus.data import Bundle
from milplus.engine import train, evaluate
from milplus.io import read_json, write_json, digest_file
from milplus.provenance import software_identity
target, output = map(Path, sys.argv[1:3]); expected = sys.argv[3]
assert Path(milplus.__file__).is_relative_to(target) and milplus.__version__ == expected
identity = software_identity(); results = []
torch.set_num_threads(1)
for dimension, classes in [(512, 2), (1536, 3)]:
 data = create_focus_demo(output/f'data_{dimension}', patients=10*classes,
     dimension=dimension, classes=classes)
 bundle = Bundle(data); run = output/f'focus_source_{dimension}'
 train(data, 'FOCUS', run, seed=2024, config={'threads': 1})
 assert not (run/'test_predictions.csv').exists()
 request = read_json(run/'request.json'); cfg = request['config']
 assert cfg['model'] == {} and cfg['epochs'] == 200 and cfg['lr'] == .0001 and cfg['weight_decay'] == 1e-5
 assert cfg['selection_metric'] == 'focus_val_error' and cfg['patience'] == 20
 assert cfg['max_instances'] is None and cfg['grad_clip'] is None
 last = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
 best = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
 state, history = last['model'], last['history']; completed = len(history)
 extra = state['_extra_state']; protocol = read_json(run/'method_protocol.json'); settings = protocol['settings']
 assert settings['window_size'] == 8 and settings['sim_threshold'] == .8 and settings['max_context_length'] == 8192
 assert settings['stop_epoch'] == 40 and settings['optimizer'] == 'adam'
 assert protocol['text_width'] == 768 and protocol['text_layers'] == 12 and protocol['text_heads'] == 12
 assert protocol['trainable_prompt_tokens'] == 16 and protocol['trainable_text_parameters'] == 85547520
 assert protocol['pretrained_weights_sha256'] == WEIGHTS_SHA
 assert 42 <= completed <= 200 and last['epoch'] == completed-1 and extra['completed_epochs'] == completed
 assert extra['optimizer_updates'] == len(bundle.records('train'))*completed
 train_ids = {r['slide_id'] for r in bundle.records('train')}
 assert all(len(order) == len(train_ids) and set(order) == train_ids for order in extra['train_orders'])
 assert state['core.network.feature_encoder.0.weight'].shape == (512, dimension)
 assert state['core.network.prompt_learner.ctx'].shape == (16, 768)
 assert state['core.network.text_encoder.text_projection'].shape == (768, 512)
 assert state['core.network.text_encoder.transformer.resblocks.11.attn.in_proj_weight'].shape == (2304, 768)
 assert state['core.network.q_proj.weight'].shape == (512, 512)
 selected = min(history, key=lambda row: row['validation']['focus_val_error'])['epoch']
 assert best['epoch'] == selected
 if completed < 200:
  assert last['bad'] >= 20 and last['epoch'] > 40
 assert all(0 <= row['validation']['focus_val_error'] <= 1 for row in history)
 assert all(group['lr'] == .0001 for group in last['optimizers'][0]['param_groups'])
 names = [n for n in state if n.startswith('core.network.text_encoder.')]
 # The actual text branch was updated, not just retained as a frozen source tensor.
 source = torch.load(bundle.focus.weights, map_location='cpu', weights_only=True, mmap=True)
 assert all(not torch.equal(state[n], source['text.'+n.removeprefix('core.network.text_encoder.')]) for n in names)
 del source, last, best, state
 metrics = evaluate(run); sha = digest_file(run/'test_predictions.csv')
 assert evaluate(run) == metrics and digest_file(run/'test_predictions.csv') == sha
 results.append({'profile': 'focus_source', 'feature_dimension': dimension, 'classes': classes,
  'text_width': 768, 'text_layers': 12, 'text_heads': 12, 'trainable_text_parameters': 85547520,
  'prompt_tokens': 16, 'selection_budget': 8192, 'window_size': 8, 'compression_threshold': .8,
  'max_epochs': 200, 'completed_epochs': completed, 'optimizer_updates': extra['optimizer_updates'],
  'train_slides': len(train_ids), 'original_patience': 20, 'original_stop_epoch': 40,
  'selected_epoch': selected, 'prediction_sha256': sha, 'metrics': metrics, 'state': 'passed'})
 write_json(output/'progress.json', {'state': 'running', 'completed': results})
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):
  assert Path(module.__file__).is_relative_to(target), name
write_json(output/'validation.json', {'state': 'passed', 'version': milplus.__version__, 'software': identity,
 'profiles': results, 'synthetic': True, 'device': 'cpu',
 'scope': 'Two actual independent-wheel full pretrained768/12/12 CONCH trainable-text FOCUS runs on full512/C2 and1536/C3 synthetic visual inputs. Source200max/patience20/epoch>40 and first strict best complete VAL-error selection preserved. Actual selected-checkpoint reload and repeated TEST prediction. Synthetic visual projection extension, no natural VLM alignment/WSI/CUDA/paper-score reproduction claim.'})
'''
    env = dict(os.environ)
    # Keep optional private tokenizer dependencies explicitly in the environment;
    # never permit source-tree milplus imports in the independent child.
    env['PYTHONPATH'] = os.pathsep.join(p for p in env.get('PYTHONPATH', '').split(os.pathsep)
        if p and Path(p).resolve() != (Path(args.source_tree)/'src').resolve())
    with (output/'program.log').open('w') as log:
        subprocess.run([sys.executable, '-c', child, str(target), str(output), args.expected_version],
            cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
