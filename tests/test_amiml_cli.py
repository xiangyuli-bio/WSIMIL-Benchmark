"""AMIML entry contracts and two-level validation-only model selection."""
import copy

import pytest
import torch

from milplus.amiml import AMIML
from milplus.amiml_demo import create_amiml_demo
from milplus.amiml_network import PAPER
from milplus.cli import dispatch, parser
from milplus.data import Bundle
from milplus.engine import train
from milplus.registry import inspect_method, profiles


def test_cli_synthetic_patient_bags_and_complete_alias_and_mixed_input_guard(tmp_path):
    result = dispatch(parser().parse_args(['demo', '--amiml', '--patients', '20', '--output', str(tmp_path/'data')]))
    b = Bundle(result['created'])
    assert b.config['synthetic'] and b.config['training_unit'] == b.config['evaluation_unit'] == 'patient'
    assert b.dim == 256 and inspect_method('AMIML')['id'] == PAPER
    assert {p['id'] for p in profiles() if p['family'] == 'amiml'} == {PAPER, 'amiml_source_repaired'}
    for flag in ['--dtmil', '--focus', '--pamil-policy', '--raw-images', '--survival', '--omics']:
        with pytest.raises(ValueError, match='mixed input'):
            dispatch(parser().parse_args(['demo', '--amiml', flag, '--output', str(tmp_path/'bad')]))
    assert not (tmp_path/'bad').exists()


def test_test_prediction_cannot_select_cluster_before_validation(tmp_path):
    b = Bundle(create_amiml_demo(tmp_path/'data', dimension=16, per_cluster=12))
    model = AMIML(b, R=11)
    model.prepare(b, 2026, 'cpu')
    with pytest.raises(ValueError, match='TEST cannot choose'): model.predict_subset(b, 'test', 'cpu', {})


def test_Cway_paper_extension_trains_all_four_full_networks_and_patient_predictions(tmp_path):
    from milplus.engine import evaluate
    from milplus.io import read_json
    data = create_amiml_demo(tmp_path/'data', patients=30, classes=3, dimension=16, per_cluster=12)
    run = tmp_path/'run'
    train(data, PAPER, run, config={'epochs':6, 'threads':1, 'model':{'R':11,'batch_size':4}})
    result = evaluate(run)
    assert result['n'] == 6 and result['unit'] == 'patient' and result['macro_auc'] is not None
    state = torch.load(run/'last.pt', weights_only=False)['model']
    assert all(state[f'networks.{c}.value.weight'].shape == (3,8) for c in range(4))
    assert len(state['_extra_state']['best_meta']) == 4
    assert not any('test' in str(m['predictions']) for m in state['_extra_state']['best_meta'])


@pytest.mark.parametrize('corruption', ['optimizer_count', 'train_identity', 'val_identity', 'scheduler', 'selected_cluster'])
def test_checkpoint_refuses_inconsistent_true_patient_cluster_and_optimizer_state(tmp_path, corruption):
    data = create_amiml_demo(tmp_path/'data', dimension=16, per_cluster=12)
    train(data, PAPER, tmp_path/'run', config={'epochs':6, 'threads':1, 'model':{'R':11, 'batch_size':4}})
    checkpoint = torch.load(tmp_path/'run/last.pt', weights_only=False)
    state = copy.deepcopy(checkpoint['model']['_extra_state'])
    if corruption == 'optimizer_count': state['optimizer_steps'][0] += 1
    elif corruption == 'train_identity': state['train_orders'][0][0][0] = 'test_patient_not_train'
    elif corruption == 'val_identity': state['best_meta'][0]['predictions'][0]['subset'] = 'test'
    elif corruption == 'scheduler': state['schedulers'][0]['last_epoch'] += 1
    else: state['selected_cluster'] = (state['selected_cluster']+1)%4
    model = AMIML(Bundle(data), R=11, batch_size=4)
    with pytest.raises(ValueError, match='AMIML checkpoint'): model.set_extra_state(state)
