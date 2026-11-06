"""Run the complete native INS frozen-feature budget from an independent wheel."""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-tree')
    parser.add_argument('--expected-version')
    parser.add_argument('--continue-output', action='store_true', help='Verify completed runs in place; train only genuinely missing dimensions')
    parser.add_argument('--wait-for-pid', type=int, help='Wait for the identified original native verifier to terminate before continuing its output')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.wait_for_pid:
        if not args.continue_output:
            raise ValueError('--wait-for-pid requires explicit continuation')
        path = Path('/proc')/str(args.wait_for_pid)
        if path.exists():
            assert b'validate_ins_wheel.py' in (path/'cmdline').read_bytes()
        while True:
            try:
                state = (path/'stat').read_text().rsplit(')', 1)[1].split()[0]
            except FileNotFoundError:
                break
            if state == 'Z':
                break
            time.sleep(5)
    output.mkdir(parents=True, exist_ok=args.continue_output)
    target = output/'wheel_import'
    target.mkdir(exist_ok=args.continue_output)
    with zipfile.ZipFile(args.wheel) as archive:
        members = [n for n in archive.namelist() if n.startswith('milplus/') and not n.endswith('/')]
        if args.source_tree:
            for name in members:
                assert archive.read(name) == (Path(args.source_tree)/'src'/name).read_bytes(), name
        if args.continue_output:
            for name in members:
                assert archive.read(name) == (target/name).read_bytes(), name
        else:
            archive.extractall(target)
    child = r'''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch, milplus
from milplus.data import Bundle
from milplus.engine import train, evaluate
from milplus.ins_data import TrainPatchStore
from milplus.ins_demo import create_ins_demo
from milplus.io import read_json, write_json, digest_file
from milplus.provenance import software_identity
target, output = map(Path, sys.argv[1:3]); expected = sys.argv[3]
assert Path(milplus.__file__).is_relative_to(target)
if expected: assert milplus.__version__ == expected
identity = software_identity(); records = []
for dimension in [1024, 1536]:
 data = output/f'data_{dimension}'
 if not data.exists():
  data = create_ins_demo(data, patients=20, dimension=dimension)
 bundle = Bundle(data); store = TrainPatchStore(bundle)
 positive_count = int((store.labels == 1).sum()); patch_count = len(store)
 train_ids = {r['slide_id'] for r in bundle.records('train')}
 train_slides = len(train_ids); store.close()
 run = output/f'ins_frozen_source_repaired_{dimension}'
 if run.exists():
  assert read_json(run/'status.json')['state'] == 'completed', 'Do not reinterpret partial training as complete or silently retrain'
 else:
  train(data, 'INS', run, seed=2026, config={'threads': 1})
 assert not (run/'test_predictions.csv').exists()
 request = read_json(run/'request.json'); cfg = request['config']
 protocol = read_json(run/'method_protocol.json'); settings = protocol['settings']
 assert cfg['model'] == {}, 'Exercise actual public native defaults, not hidden architecture overrides'
 assert cfg['epochs'] == 150 and cfg['lr'] == .01 and cfg['weight_decay'] == .001
 assert cfg['grad_clip'] is None and cfg['max_instances'] is None and cfg['selection_metric'] == 'macro_auc'
 assert settings['batch_size'] == 64 and settings['warmup_epochs'] == 20 and settings['queue_size'] == 8192
 assert settings['projection'] == 128 and settings['contrastive_weight'] == .5
 assert settings['bag_weight'] == settings['instance_weight'] == 1
 assert settings['momentum'] == .999 and settings['prototype_momentum'] == .9
 assert settings['sgd_momentum'] == .9 and settings['cosine'] is False
 last = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
 best = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
 history = last['history']; state = last['model']; extra = state['core._extra_state']
 expected_instance = (positive_count//64)*20+(patch_count//64)*130
 expected_bag = train_slides*150
 assert len(history) == 150 and last['epoch'] == 149
 assert state['_extra_state']['completed_epochs'] == 150
 assert extra['steps'] == expected_instance and extra['bag_steps'] == expected_bag
 assert extra['confidence_momentum'] == .95+149/150*(.8-.95)
 assert state['core.network.encoder_q.encoder.conv1.weight'].shape == (64, 3, 3, 3)
 assert state['core.network.encoder_q.encoder.layer4.1.conv2.weight'].shape == (512, 512, 3, 3)
 assert state['core.network.encoder_q.head.2.weight'].shape == (128, 512)
 assert state['core.network.bag_classifier.weight'].shape == (2, 512)
 assert state['core.network.queue'].shape == (8192, 128)
 assert state['core.confidence'].shape == (patch_count, 2)
 valid, indices = state['core.queue_valid'], state['core.queue_indices']
 assert int(valid.sum()) == min(8192, expected_instance*64)
 assert (indices[valid] >= 0).all() and (indices[valid] < patch_count).all()
 labels = state['core.training_bag_labels']
 torch.testing.assert_close(state['core.confidence'][labels == 0],
     torch.tensor([[1., 0.]]).repeat(int((labels == 0).sum()), 1), rtol=0, atol=0)
 assert (state['core.confidence'][labels == 1] != .5).any()
 assert all(torch.isfinite(x).all() for x in [state['core.network.queue'], state['core.network.prototypes'], state['core.confidence']])
 assert all(r['train']['instance_update'] == r['train']['bag_update'] == 1 for r in history)
 assert all(r['train']['prototype_refinement'] == float(i >= 20) for i, r in enumerate(history))
 assert all(r['train']['contrastive_loss'] > 0 for r in history)
 assert last['optimizers'][0]['param_groups'][0]['lr'] == .01
 assert last['optimizers'][0]['param_groups'][0]['momentum'] == .9
 assert all('momentum_buffer' in s for s in last['optimizers'][0]['state'].values())
 selected = max(history[20:], key=lambda r: r['validation']['macro_auc'])['epoch']
 assert best['epoch'] == selected and selected >= 20
 protocol = read_json(run/'method_protocol.json')
 assert protocol['completed_epochs'] == 150 and protocol['feature_dimension'] == dimension
 assert protocol['instance_updates'] == expected_instance and protocol['bag_updates'] == expected_bag
 assert protocol['source_commit'] == 'cd97ba2fa7df2a951f218e6de405f664129931c8'
 metrics = evaluate(run); sha = digest_file(run/'test_predictions.csv')
 assert evaluate(run) == metrics and digest_file(run/'test_predictions.csv') == sha
 records.append({'profile': 'ins_frozen_source_repaired', 'feature_dimension': dimension,
    'native_encoder': 'actual_source_ResNet18_2_2_2_2', 'projection': 128,
    'completed_epochs': 150, 'warmup_epochs': 20, 'batch_size': 64, 'queue_size': 8192,
    'positive_train_patches': positive_count, 'train_patches': patch_count,
    'train_slides': train_slides, 'instance_updates': expected_instance, 'bag_updates': expected_bag,
    'optimizer_updates': expected_instance+expected_bag, 'selected_epoch': selected,
    'prediction_sha256': sha, 'metrics': metrics, 'state': 'passed'})
 write_json(output/'progress.json', records)
 del last, best, bundle, state
for name, module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module, '__file__', None):
  assert Path(module.__file__).is_relative_to(target), name
assert identity == software_identity()
write_json(output/'validation.json', {'state': 'passed', 'version': milplus.__version__,
 'software': identity, 'profiles': records, 'synthetic': True, 'device': 'cpu',
 'scope': 'Two actual independent-wheel full1024/1536FM runs, full native author ResNet18,8192queue,128projection,B64,150epochs/20warmup. Complete bag-only source-repaired INS frozen-feature recipe. Original1024/1536features retained, complete patch and bag phase counts verified; actual best-checkpoint reload and repeated TEST predictions. Synthetic small slide bags; no natural WSI/CUDA/RGB end-to-end/paper-score claim.'})
'''
    env = {**os.environ, 'PYTHONPATH': str(target), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    if args.continue_output and (output/'program.log').exists():
        archived = output/'program_original_metadata_verifier_r1.log'
        if archived.exists():
            raise ValueError('Preserve existing continuation evidence; choose an explicit new recovery')
        (output/'program.log').rename(archived)
    with (output/'program.log').open('w') as log:
        result = subprocess.run([sys.executable, '-c', child, str(target), str(output), args.expected_version or ''],
            cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(json.dumps({'returncode': result.returncode, 'output': str(output), 'runtime_members': len(members)}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
