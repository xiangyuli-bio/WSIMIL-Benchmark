"""Actual native MGGM-MIL graph/sequence equations; no ordinary-Mamba proxy."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from milplus.data import Bundle
from milplus.engine import seed_everything, rng_state, restore_rng
from milplus.mggmmil_source import original_model, author_classes, COMMIT
from milplus.mggmmil_data import GraphInputs
from milplus.mggmmil_demo import create_demo
from milplus.mggmmil import MGGMMIL, SourceSelection, Bags, original_training
from milplus.upstream import verified_source
from test_pmil_prototypical_components import equal


def test_full_original4layer384D_GENConv_SRMamba_every_gradient_Adam_and_rng(tmp_path):
    torch.set_num_threads(2)
    bundle = Bundle(create_demo(tmp_path/'data'))
    seed_everything(101)
    actual = MGGMMIL(bundle)
    seed_everything(101)
    source, _, _ = author_classes()
    original = source.MGGMMIL(args=SimpleNamespace(KNN='4+8+12'), I=512, L=384, D=256,
                             n_classes=2, layer=4, mil_head='GABMIL')
    equal(actual.network.state_dict(), original.state_dict())
    assert len(actual.network.gcn_blocks) == 4
    for block in actual.network.gcn_blocks:
        assert block.num_scale == 3
        assert block.projector.d_model == 384 and block.projector.d_state == 16
        assert block.projector.d_conv == 4 and block.projector.expand == 1
    assert verified_source('mggmmil')[1] == COMMIT
    original.train(); actual.network.train()
    ao = torch.optim.Adam(original.parameters(), lr=3e-5, weight_decay=1e-5)
    bo = torch.optim.Adam(actual.network.parameters(), lr=3e-5, weight_decay=1e-5)
    x, graphs = actual.assets.load(bundle.records('train')[0])
    edges = [g.unsqueeze(0) for g in graphs]
    target = torch.tensor([1])
    for _ in range(2):
        state = rng_state()
        left = original(x.unsqueeze(0), edges)
        restore_rng(state)
        right = actual.network(x.unsqueeze(0), edges)
        equal(left, right)
        ao.zero_grad(); bo.zero_grad()
        F.cross_entropy(left[0], target).backward()
        F.cross_entropy(right[0], target).backward()
        for p, q in zip(original.parameters(), actual.network.parameters()):
            equal(p.grad, q.grad)
        ao.step(); bo.step()
        equal(original.state_dict(), actual.network.state_dict())
        equal(ao.state_dict(), bo.state_dict())


@pytest.mark.parametrize('length', [51, 52, 53])
def test_original_sequence_reordering_padding_removal_is_not_a_reverse_scan(length):
    _, srm, _ = author_classes()
    x = torch.arange(length*384, dtype=torch.float32).reshape(1, 384, length)
    actual = srm.TransposeTokenReEmbedding.transpose_normal_padding(x, rate=10)
    padded = F.pad(x, (0, (-length) % 10))
    expected = padded.reshape(1, 384, -1, 10).permute(0, 1, 3, 2).reshape(1, 384, -1)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual[:, :, :length], x.flip(-1))
    assert torch.equal(srm.TransposeTokenReEmbedding.transpose_remove_padding(actual, 10, length), x)


def test_original50epoch_selection_boundary_ties20patience_and_exact_state():
    selection = SourceSelection()
    for epoch in range(1, 50):
        selection.observe(epoch, .01)
        assert not selection.improved and selection.selected_epoch is None and selection.counter == 0
    selection.observe(50, .7)
    assert selection.improved and selection.selected_epoch == 50
    restored = SourceSelection(); restored.load(vars(selection))
    assert vars(restored) == vars(selection)
    for epoch in range(51, 71):
        selection.observe(epoch, .7)
        restored.observe(epoch, .7)
        assert vars(restored) == vars(selection)
        assert selection.done == (epoch == 70)
    assert selection.counter == 20 and selection.selected_epoch == 50


def test_literal_source_train_and_validation_loops_preserve_full_graph_states(tmp_path):
    torch.set_num_threads(2)
    bundle = Bundle(create_demo(tmp_path/'data'))
    cfg = {'epochs': 200, 'lr': 3e-5, 'weight_decay': 1e-5, 'patience': 20,
           'grad_clip': None, 'max_instances': None, 'selection_metric': 'mggmmil_val_negative_ce', 'threads': 2}
    seed_everything(72)
    actual = MGGMMIL(bundle); actual.configure_training(cfg)
    opt = actual.make_optimizer(3e-5, 1e-5)
    # Capture exact original initialization/RNG before either whole-bag loop.
    initial = copy.deepcopy(actual.network.state_dict()); state = rng_state()
    actual.start_epoch(0); actual.epoch(bundle, opt, torch.device('cpu'))
    frame = actual.predict_subset(bundle, 'val', torch.device('cpu'), cfg)
    actual.selection_objective(actual.validation_metrics(frame), 0, 0)
    final_rng = rng_state()
    original, _ = original_model()
    original.load_state_dict(initial)
    source_opt = torch.optim.Adam(original.parameters(), lr=3e-5, weight_decay=1e-5)
    trainer = original_training()
    owner = SimpleNamespace(args=SimpleNamespace(n_epochs=1, device=torch.device('cpu'), baseline_model='MGGMMIL'),
                            model=original, optimizer=source_opt, loss=torch.nn.CrossEntropyLoss(), flag=1)
    observed = {}
    def validate(epoch):
        original.eval(); loss = []
        with torch.no_grad():
            for x, y, path, adj, edges in DataLoader(Bags(actual, 'val'), batch_size=1, shuffle=False):
                loss.append(float(trainer.test_inference(owner, x, y, 'MGGMMIL', adj, edges)[0]))
        observed['loss'] = float(np.mean(loss))
    owner.valid = validate
    owner.train_inference = lambda *a, **kw: trainer.train_inference(owner, *a, **kw)
    owner.train_loader = DataLoader(Bags(actual, 'train'), batch_size=1, shuffle=True)
    owner.best_valid_metrics = None
    restore_rng(state)
    trainer.train(owner)
    equal(original.state_dict(), actual.network.state_dict())
    equal(source_opt.state_dict(), opt.state_dict())
    equal(final_rng, rng_state())
    assert observed['loss'] == actual._val_loss
    actual.validate_optimizer_state()
    saved = copy.deepcopy(actual.state_dict())
    restored = MGGMMIL(bundle); restored.configure_training(cfg)
    restored_opt = restored.make_optimizer(3e-5, 1e-5)
    restored.load_state_dict(saved); restored_opt.load_state_dict(opt.state_dict())
    restored.validate_optimizer_state(); equal(actual.state_dict(), restored.state_dict())


def test_invalid_or_tampered_native_graphs_cannot_be_rebuilt_or_fallback(tmp_path):
    import json
    bundle = Bundle(create_demo(tmp_path/'data'))
    assets = GraphInputs(bundle)
    path = bundle.root/'graphs.json'
    record = json.loads(path.read_text())
    record['slides']['synthetic_0']['features_sha256'] = '0'*64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='bytes changed'):
        assets.verify(deep=True)
    with pytest.raises(ValueError, match='binding changed'):
        GraphInputs(bundle)
    with pytest.raises(ValueError, match='actual512D'):
        original_model(1024)


def test_full328_profile153_family_scope_preserves_all327_prior_records_and_pending_admission():
    import json
    from pathlib import Path
    from milplus.registry import profiles, methods
    old = json.loads((Path(__file__).parent/'fixtures/mggmmil_prior327_profiles.json').read_text())
    historical_ids={p['id'] for p in old}|{'mggmmil_uaf_original_graph_patient_reference'}
    current=[p for p in profiles() if p['id'] in historical_ids]
    assert len(current)==328 and len({p["family"] for p in current})==153
    assert [p for p in current if p["family"]!="mggmmil"] == old
    assert len(methods()) == 164
    method = next(m for m in methods() if m['id'] == 'mggmmil')
    assert method['profiles'] == ['mggmmil_uaf_original_graph_patient_reference']
    assert method['status']=='adapted_under_verification'
    from milplus.registry import inspect_method
    assert inspect_method('MGGMMIL')['id']==method['profiles'][0]
