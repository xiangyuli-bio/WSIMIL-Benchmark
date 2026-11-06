"""HEALNet source fidelity, active multimodal paths and complete recipe recovery."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train
from milplus.healnet import healnet_epoch, load_healnet
from milplus.models import build_bundle_model
from milplus.upstream import source_root


@pytest.fixture(autouse=True)
def check_source():
    if not source_root('healnet').is_dir():
        pytest.skip('Fetch pinned HEALNet source')
    torch.set_num_threads(1)


def fixture(tmp_path, **options):
    bundle = Bundle(create_demo(tmp_path/'data', patients=20, survival=True, omics=True))
    model = build_bundle_model('healnet', bundle, {'bins': 3, 'attn_dropout': 0., 'ff_dropout': 0., **options})
    model.prepare(bundle, 2026, 'cpu')
    return model, bundle


@pytest.mark.parametrize('layout', ['vector', 'gene_tokens'])
@pytest.mark.parametrize('self_attention,tied', [(0, False), (1, True)])
def test_author_full_output_and_gradient(tmp_path, layout, self_attention, tied):
    model, bundle = fixture(tmp_path, omics_layout=layout, self_per_cross_attn=self_attention,
                            weight_tie_layers=tied, depth=3)
    reference = load_healnet(strict=False).HealNet(n_modalities=2,
        channel_dims=[18 if layout == 'vector' else 1, bundle.dim], num_spatial_axes=[1, 1],
        out_dims=3, depth=3, l_c=4, l_d=16, num_freq_bands=2, max_freq=2.,
        x_heads=1, l_heads=8, cross_dim_head=16, latent_dim_head=16,
        self_per_cross_attn=self_attention, weight_tie_layers=tied)
    reference.load_state_dict(model.network.state_dict())
    x, omic = torch.randn(7, bundle.dim, requires_grad=True), torch.randn(18, requires_grad=True)
    xx, oo = [v.detach().clone().requires_grad_() for v in (x, omic)]
    a = model(x, omic)['hazard_logits']
    normalized = (oo-model.omics_mean)/model.omics_scale
    shaped = normalized[None, None] if layout == 'vector' else normalized[None, :, None]
    b = reference([shaped, xx[None]])
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().mean().backward(); b.square().mean().backward()
    torch.testing.assert_close(x.grad, xx.grad, rtol=0, atol=0)
    torch.testing.assert_close(omic.grad, oo.grad, rtol=0, atol=0)
    assert x.grad.abs().sum() > 0 and omic.grad.abs().sum() > 0
    for p, q in zip(model.network.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize('weighting', ['inverse', 'inverse_root', 'none'])
def test_weighted_nll_exact_author_definition_not_plain_weighted_ce(tmp_path, weighting):
    model, _ = fixture(tmp_path, bin_weighting=weighting)
    logits = torch.randn(3, 3, requires_grad=True)
    labels, events = torch.tensor([0, 1, 2]), torch.tensor([1., 0., 1.])
    hazard = logits.sigmoid()
    actual = model.weighted_loss(hazard, labels, events)
    s = torch.cat([torch.ones(3, 1), (1-hazard).cumprod(1)], 1)
    e = -events*(s[range(3), labels].log() + hazard[range(3), labels].log())
    c = -(1-events)*s[range(3), labels+1].log()
    w = 1 if weighting == 'none' else (model.bin_weights/model.bin_weights.sum())[labels]
    expected = ((1-model.alpha)*(e+c)*w + model.alpha*e).mean()
    torch.testing.assert_close(actual, expected)
    ga, gb = [torch.autograd.grad(loss, logits, retain_graph=True)[0] for loss in [actual, expected]]
    torch.testing.assert_close(ga, gb)


def test_missing_modalities_keep_slots_and_invalid_channels_raise(tmp_path):
    model, bundle = fixture(tmp_path)
    x, omics = torch.randn(7, bundle.dim), torch.randn(18)
    for a, b in [(x, None), (None, omics)]:
        output = model(a, b)
        assert output['hazard_logits'].shape == (1, 3) and torch.isfinite(output['logits']).all()
    with pytest.raises(ValueError, match='at least one'):
        model(None, None)
    with pytest.raises(ValueError, match='WSI'):
        model(torch.randn(4, 5), omics)
    # The source used bare except/pass at precisely this failing cross-attention call.
    with pytest.raises(RuntimeError):
        model.network([torch.randn(1, 1, 17), x[None]])


def test_training_only_normalization_bins_and_weights(tmp_path):
    model, bundle = fixture(tmp_path)
    held = bundle.rows.query("subset != 'train'").global_patient_id
    bundle.omics.values.loc[held.unique()] = 1e6
    bundle.rows.loc[bundle.rows.subset != 'train', 'time'] = 1e8
    other = build_bundle_model('healnet', bundle, {'bins': 3})
    other.prepare(bundle, 5, 'cpu')
    for key in ['bin_edges', 'bin_weights', 'omics_mean', 'omics_scale']:
        torch.testing.assert_close(getattr(model, key), getattr(other, key))


def test_batch_updates_l1_divisor_and_onecycle_match_direct_author_batch(tmp_path):
    model, bundle = fixture(tmp_path, batch_size=3)
    reference = copy.deepcopy(model)
    cfg = {'epochs': 2, 'grad_clip': 1e5, 'max_instances': None}
    original_load = bundle.load
    bundle.load = lambda row: tuple(a[:8] if a is not None else None for a in original_load(row))
    groups = [[row] for row in bundle.records('train')[:5]]
    for m in [model, reference]:
        m.configure_training(cfg)
        m.make_optimizer(.008, 0)
    actual = healnet_epoch(model, bundle, groups, model.optimizer, 'cpu', cfg)
    reference.ensure_scheduler()
    expected = []
    for start in [0, 3]:
        rows = [g[0] for g in groups[start:start+3]]
        x = torch.stack([bundle.load(row)[0] for row in rows])
        o = torch.stack([reference.normalize_omics(bundle.omics.tensor(row['global_patient_id'])) for row in rows])
        logits = reference.network([o[:, None], x])
        loss = reference.weighted_loss(logits.sigmoid(), reference.time_label([r['time'] for r in rows]),
                                      torch.tensor([r['event'] for r in rows]))
        penalty = reference.l1_weight*sum(p.abs().sum() for p in reference.network.parameters())
        total = loss/reference.loss_divisor + penalty
        reference.optimizer.zero_grad(); total.backward(); reference.optimizer.step(); reference.scheduler.step()
        expected.append(float(total.detach()))
    np.testing.assert_allclose([r['loss'] for r in actual], expected, rtol=1e-6)
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    assert model.scheduler.state_dict() == reference.scheduler.state_dict()


@pytest.mark.parametrize('missing', [None, 'omics', 'wsi'])
def test_train_reload_exact_resume_and_held_out_missing_ablation(tmp_path, missing):
    root = create_demo(tmp_path/'data', patients=20, survival=True, omics=True)
    cfg = {'epochs': 2, 'threads': 1, 'model': {'bins': 3, 'attn_dropout': .1,
                                               'ff_dropout': .1, 'eval_missing': missing}}
    full = train(root, 'healnet', tmp_path/'full', config=cfg)
    part = train(root, 'healnet', tmp_path/'part', config=cfg, stop_after=1)
    train(root, 'healnet', part, config=cfg, resume=True)
    a, b = [torch.load(path/'last.pt', weights_only=False) for path in [full, part]]
    assert a['history'] == b['history']
    assert a['model']['_extra_state'] == b['model']['_extra_state']
    for key in a['model']:
        if key != '_extra_state':
            torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    torch.testing.assert_close(a['optimizers'], b['optimizers'], rtol=0, atol=0)
    assert not (full/'test_predictions.csv').exists()
    metrics = evaluate(full)
    assert np.isfinite(metrics['survival_loss'])
    frame = pd.read_csv(full/'test_predictions.csv')
    assert len(frame) == 4
    evaluate(full)
    pd.testing.assert_frame_equal(frame, pd.read_csv(full/'test_predictions.csv'))
