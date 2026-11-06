"""Fresh public processes for both complete LNPL conditions and real recovery."""
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import pandas as pd
import pytest
import torch

from milplus.data import Bundle
from milplus.io import read_json
from milplus.hybrid_supervision_checkpoint import load
from milplus.registry import require_runnable, check_compatibility
from milplus.lnplmil_driver import bound_pointer
from milplus.lnplmil_tod import SelectedInputs
from test_lnplmil_weak import official_assets
from test_pmil_prototypical_components import equal


def cli(*args, success=True):
    result = subprocess.run([sys.executable, '-m', 'milplus', *map(str, args)],
                            capture_output=True, text=True, env=os.environ.copy())
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
    return result


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp('lnpl_public_cli')
    data = root / 'data'
    cli('demo-lnplmil', '--output', data, '--assets', official_assets())
    config = root / 'software.yaml'
    config.write_text('weak_epochs: 2\nweak_batch_size: 256\nepochs: 2\nthreads: 2\n')
    return root, data, config


@pytest.mark.parametrize('condition', ['lnplmil_released_with_reference_weak', 'lnplmil_paper_with_reference_weak'])
def test_complete_public_source_and_paper_cli_all_actual_stages_and_selected_reload(dataset, condition):
    root, data, config = dataset
    out = root / condition
    cli('run', '--data', data, '--method', condition, '--output', out, '--config', config)
    status = read_json(out / 'status.json')
    assert status['state'] == 'completed' and status['weak_updates'] == 2 and status['TOD_updates'] == 2
    assert read_json(out / 'request.json')['synthetic'] is True
    selection = SelectedInputs(bound_pointer(out, 'selection'))
    assert selection.meta['HNSW_execution']['canonical_single_build_thread'] is True
    assert selection.meta['HNSW_execution']['candidate_order'] == 'level0_XY_before_original_TopK'
    assert all(r['original_patch_count'] == 401 for r in selection.rows)
    cli('evaluate', '--run', out)
    first = (out / 'test_predictions.csv').read_bytes()
    cli('evaluate', '--run', out)
    assert (out / 'test_predictions.csv').read_bytes() == first
    frame = pd.read_csv(out / 'test_predictions.csv')
    assert set(frame.global_patient_id) == set(Bundle(data).rows.query("subset=='test'").global_patient_id)
    assert read_json(out / 'status.json')['test_evaluated'] is True
    assert read_json(out / 'test_metrics.json')['unit'] == 'slide'


@pytest.mark.parametrize('cut', [1, 3])
def test_public_real_weak_and_partial_TOD_gradient_interruptions_exact_learned_state(dataset, cut):
    root, data, config = dataset
    method = 'lnplmil_released_with_reference_weak'
    baseline = root / method
    if not baseline.exists():
        cli('run', '--data', data, '--method', method, '--output', baseline, '--config', config)
    resumed = root / ('resume_' + str(cut))
    cli('run', '--data', data, '--method', method, '--output', resumed, '--config', config, '--stop-after', cut)
    assert read_json(resumed / 'status.json')['state'] == 'interrupted'
    if cut == 3:
        assert any(v is not None for v in load(resumed / 'tod/last.pt')['gradients'].values())
    cli('run', '--data', data, '--method', method, '--output', resumed, '--config', config, '--resume')
    equal(load(baseline / 'weak/last.pt'), load(resumed / 'weak/last.pt'))
    a, b = load(baseline / 'tod/last.pt'), load(resumed / 'tod/last.pt')
    # Separate run-local manifests legitimately have different absolute paths
    # and encoded-cache hashes. Validate each complete contract and compare
    # every numerical/optimizer/gradient/RNG/order/history state unchanged.
    from milplus.io import digest_object
    for state, path in [(a, baseline), (b, resumed)]:
        assert state['contract'] == digest_object(read_json(path / 'tod/request.json'))
    equal({k: v for k, v in a.items() if k != 'contract'}, {k: v for k, v in b.items() if k != 'contract'})
    cli('evaluate', '--run', baseline); cli('evaluate', '--run', resumed)
    assert (baseline / 'test_predictions.csv').read_bytes() == (resumed / 'test_predictions.csv').read_bytes()


def test_unified_bundle_rejects_feature_only_method_and_heldout_observed_annotations(dataset, tmp_path):
    _, data, _ = dataset
    bundle = Bundle(data)
    assert bundle.lnplmil is not None
    with pytest.raises(ValueError, match='explicit complete LNPL'):
        check_compatibility(require_runnable('ABMIL'), bundle)
    with pytest.raises(ValueError, match='full weak/scoring/SP/TOD'):
        bundle.load(bundle.records('train')[0])
    output = cli('inspect', '--method', 'LNPL-MIL')
    record = json.loads(output.stdout)
    assert record['family'] == 'lnplmil' and record['formal_admission'] is False
    assert record['id'] == 'lnplmil_paper_with_reference_weak'
    import shutil
    import numpy as np
    import yaml
    bad = tmp_path / 'heldout'
    shutil.copytree(data, bad)
    task = yaml.safe_load((bad / 'task.yaml').read_text())
    task['encoders']['raw_rgb'].pop('lnplmil')
    (bad / 'task.yaml').write_text(yaml.safe_dump(task))
    index = pd.read_csv(bad / 'pixels.csv')
    index['path'] = [str(bad / 'pixels' / Path(p).name) for p in index.path]
    index.to_csv(bad / 'pixels.csv', index=False)
    with h5py.File(index.path.iloc[-1], 'r+') as file:
        del file['labelled']; del file['patch_labels']
        file.create_dataset('labelled', data=np.asarray([0], dtype=np.int64))
        file.create_dataset('patch_labels', data=np.asarray([1], dtype=np.int64))
    result = cli('prepare-lnplmil', '--data', bad, '--schema', bad / 'schema.yaml',
                 '--output', 'rejected_manifest.json', success=False)
    assert 'Only TRAIN patients' in result.stderr
    assert not (bad / 'rejected_manifest.json').exists()
