"""Full actual optimizer, support cache and persistent order survive interruption."""
import copy

import numpy as np
import pytest
import torch

from milplus.data import Bundle
from milplus.engine import evaluate, train
from milplus.io import read_json
from milplus.miladapter import MILAdapter
from milplus.miladapter_demo import create_miladapter_demo


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('profile', ['miladapter_zsmil_rrtmil_zs', 'miladapter_zsmil_wikgmil_random',
    'miladapter_zsmil_ilramil_zs', 'miladapter_taskres_abmil', 'miladapter_clipadapter_abmil',
    'miladapter_tipadapter_transmil'])
def test_genuine_three_epoch_resume_preserves_all_parameters_and_source_order(tmp_path, profile):
    root = create_miladapter_demo(tmp_path/'data', encoder='KEEP' if 'tipadapter' in profile or 'clipadapter' in profile else 'CONCH')
    cfg = {'epochs': 3, 'threads': 1, 'model': {'k_shots': 2}}
    a = train(root, profile, tmp_path/'continuous', seed=87, config=cfg)
    b = train(root, profile, tmp_path/'resume', seed=87, config=cfg, stop_after=1)
    assert read_json(b/'status.json')['state'] == 'interrupted'
    assert not (b/'best.pt').exists()  # No premature validation or checkpoint selection.
    train(root, profile, b, seed=87, config=cfg, resume=True)
    sa = torch.load(a/'last.pt', map_location='cpu', weights_only=False)
    sb = torch.load(b/'last.pt', map_location='cpu', weights_only=False)
    for key in ['model', 'optimizers', 'rng', 'epoch', 'history', 'best', 'bad']:
        equal(sa[key], sb[key])
    assert sa['epoch'] == 2 and sa['optimizers'][0]['param_groups'][0]['lr'] == 0
    assert sa['model']['_extra_state']['optimizer_updates'] == 12
    selected = torch.load(a/'best.pt', map_location='cpu', weights_only=False)
    assert selected['epoch'] == 2
    assert sa['history'][0]['validation'] == {'validation_performed': False}
    assert evaluate(a) == evaluate(a) == evaluate(b)
    model = MILAdapter(Bundle(root), profile, k_shots=2, sampling_seed=87)
    model.load_state_dict(sa['model'])
    for corrupt in ['updates', 'order', 'support', 'scheduler', 'contract', 'extra']:
        state = copy.deepcopy(sa['model']['_extra_state'])
        if corrupt == 'updates':
            state['optimizer_updates'] += 1
        elif corrupt == 'order':
            state['current_order'].reverse()
        elif corrupt == 'support':
            state['train_orders'][0][0] = 'heldout_patient'
        elif corrupt == 'scheduler':
            state['scheduler']['last_epoch'] -= 1
        elif corrupt == 'contract':
            state['contract'] = 'changed_source_or_text'
        else:
            state['unrecognized'] = True
        with pytest.raises(ValueError):
            model.set_extra_state(state)
    changed = copy.deepcopy(sa['model']); changed['network.text_embeddings'][0, 0] += .5
    with pytest.raises(ValueError, match='frozen actual text'):
        model.load_state_dict(changed)


def test_generic_recipe_shortcuts_and_conflicting_support_seed_are_refused(tmp_path):
    root = create_miladapter_demo(tmp_path/'data')
    for i, options in enumerate([{'max_instances': 8}, {'grad_clip': 1.}, {'selection_metric': 'macro_auc'},
        {'model': {'sampling_seed': 1}}]):
        with pytest.raises(ValueError):
            train(root, 'MIL-Adapter', tmp_path/f'bad_{i}', config={'epochs': 1, **options})
