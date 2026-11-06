"""Full author models, dropout masks, packed supervision and exact continuation."""
import copy

import h5py
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train, units
from milplus.mil_dropout import MILDropout, load_dropout_sources
from milplus.packing import PackedMIL, load_pack_sources, packed_epoch
from milplus.recipes import classification_loss, classification_step
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import source_root


def source(name):
    if not source_root(name).is_dir():
        pytest.skip('Fetch pinned '+name+' source')
    torch.set_num_threads(1)


def author_dropout_network(dim, classes, kind):
    _, attn, dsmil, trans = load_dropout_sources(safe=False)
    if kind in {'abmil', 'gated'}:
        return attn.AttentionMIL(dim, classes, attn_mode='linear' if kind == 'abmil' else 'gated')
    if kind == 'dsmil':
        return dsmil.MILNet(dsmil.FCLayer(32, classes), dsmil.BClassifier(32, 128, classes),
                           L=32, in_features=dim)
    return trans.TransMIL(dim, classes, mDim=32)


@pytest.mark.parametrize('dim', [16, 32])
@pytest.mark.parametrize('kind', ['abmil', 'gated', 'dsmil', 'transmil'])
def test_dropout_author_models_ce_and_gradients(dim, kind):
    source('mil_dropout')
    model = MILDropout(dim, 3, backbone=kind, **({'hidden': 32} if kind in {'dsmil', 'transmil'} else {}))
    reference = author_dropout_network(dim, 3, kind)
    reference.load_state_dict(model.network.state_dict())
    x, y = torch.randn(64, dim), torch.tensor([2])
    torch.manual_seed(72); actual = model(x)
    torch.manual_seed(72); expected = reference(x)[0]
    torch.testing.assert_close(actual['logits'], expected, atol=0, rtol=0)
    left = classification_loss('mil_dropout_'+kind, actual, y)
    right = F.cross_entropy(expected, y)
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    left.backward(); right.backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), reference.named_parameters()):
        if kind == 'dsmil' and name.startswith('i_classifier.'):
            # Source returns only bag CE; critical-index sort is discrete.
            assert p.grad is None and q.grad is None
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    old = copy.deepcopy(model.network.state_dict())
    model.make_optimizer(.001, .001).step()
    assert any(not torch.equal(old[k], v) for k, v in model.network.state_dict().items())
    model.eval(); reference.eval()
    reference.load_state_dict(model.network.state_dict())
    torch.testing.assert_close(model(x)['logits'], reference(x)[0], atol=0, rtol=0)


def test_dropout_mask_neighbors_scale_gradient_and_failure_policy():
    source('mil_dropout')
    drop, *_ = load_dropout_sources()
    torch.manual_seed(36)
    x = torch.randn(31, 8, requires_grad=True)
    layer = drop.mildropout(topk=2, kernel=3)
    order = x.detach().mean(1).sigmoid().argsort(descending=True)
    removed = set(order[:2].tolist())
    for i in order[:2]:
        sim = F.cosine_similarity(x[i].detach()[None], x[order[2:]].detach(), dim=1)
        removed.update(order[2:][sim.argsort(descending=True)[:3]].tolist())
    mask = x.new_ones(len(x), 1); mask[list(removed)] = 0
    expected = x*mask*len(x)/mask.sum()
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(x.grad, mask.expand_as(x)*len(x)/mask.sum())
    layer.eval(); assert torch.equal(layer(x), x)
    layer.train(); assert torch.equal(layer(x[:1]), x[:1])
    with pytest.raises(ValueError, match='masked every instance'):
        layer(x[:2])
    raw, *_ = load_dropout_sources(safe=False)
    assert not torch.isfinite(raw.mildropout(topk=2, kernel=3)(x[:2])).all()
    with pytest.raises(ValueError, match='topk'):
        MILDropout(16, 2, topk=0)


def test_dropout_lookahead_interpolation_and_fast_moments():
    source('mil_dropout')
    model = MILDropout(16, 2, lookahead_steps=3, lookahead_alpha=.4).double()
    ref = copy.deepcopy(model.network)
    opt = model.make_optimizer(.001, .01)
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=.001, weight_decay=.01)
    slow = [p.detach().clone() for p in ref.parameters()]
    for step in range(7):
        x, y = torch.randn(48, 16, dtype=torch.float64), torch.tensor([step % 2])
        classification_step(model, 'mil_dropout_abmil', x, None, y, [opt], grad_clip=100.)
        ref_opt.zero_grad(); F.cross_entropy(ref(x)[0], y).backward()
        nn.utils.clip_grad_norm_(ref.parameters(), 100.); ref_opt.step()
        if (step+1) % 3 == 0:
            with torch.no_grad():
                for p, s in zip(ref.parameters(), slow):
                    s.copy_(.6*s+.4*p); p.copy_(s)
        for p, q in zip(model.network.parameters(), ref.parameters()):
            torch.testing.assert_close(p, q, atol=2e-8, rtol=1e-6)
    assert model.steps == 7
    assert all(s['step'] == 7 for s in opt.state.values())


@pytest.mark.parametrize('dim', [16, 32])
@pytest.mark.parametrize('survival', [False, True])
def test_pack_author_full_network_loss_gradients_and_ads(dim, survival):
    source('packmil')
    model = PackedMIL(dim, 3, hidden=32, survival=survival, pack_length=32,
                      min_sequence=2, downsample_ratio=2, batch_size=4)
    raw, *_ = load_pack_sources()
    reference = raw.PackMIL(mil='abmil', task_type='surv' if survival else 'sub', token_dropout=.5,
        group_max_seq_len=32, min_seq_len=2, pack_residual=True, downsample_mode='ads',
        residual_loss='nll' if survival else 'focal', residual_downsample_r=2,
        input_dim=dim, inner_dim=32, n_classes=4 if survival else 3, dropout=.25, act='gelu', mil_bias=True)
    reference.load_state_dict(model.network.state_dict())
    bags = [torch.randn(n, dim) for n in [41, 33, 19, 2, 55]]
    y, events = torch.tensor([0, 1, 2, 1, 0]), torch.tensor([1, 0, 1, 0, 1])
    torch.manual_seed(48); actual = model.batch_objective(bags, y, events if survival else None)
    torch.manual_seed(48); expected, aux, *_ = reference(bags, label=(y, 1-events) if survival else y)
    torch.testing.assert_close(actual['logits'], expected, atol=0, rtol=0)
    if survival:
        hazards = expected.sigmoid(); curve = (1-hazards).cumprod(1)
        prev = F.pad(curve, (1, 0), value=1.)
        ind = torch.arange(len(y))
        main = (-events*(prev[ind, y]*hazards[ind, y]).log()-(1-events)*curve[ind, y].log()).mean()
    else:
        main = F.cross_entropy(expected, y)
    loss = main+model.auxiliary_weight*aux
    torch.testing.assert_close(actual['loss'], loss)
    actual['loss'].backward(); loss.backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), reference.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad, atol=3e-7, rtol=1e-5)
    assert model.network.downsampler.attention[0].weight.grad.abs().sum() > 0
    assert model.network.downsampler.downsample.weight.grad.abs().sum() > 0
    assert actual['auxiliary_loss'] > 0
    model.eval(); reference.eval()
    torch.testing.assert_close(model.network(bags[0][None]), reference(bags[0][None]), atol=0, rtol=0)
    # ADS eval keeps every token; it does not secretly use random training compression.
    assert model.network.apply_inference_downsample(bags[0][None]).shape == bags[0][None].shape


def test_pack_masked_main_attention_matches_individual_selected_bags():
    source('packmil')
    model = PackedMIL(16, 3, hidden=32, pack_length=16, min_sequence=2,
                      downsample_ratio=2, dropout=0.)
    _, packing, _, _, _, _ = load_pack_sources()
    bags = [torch.randn(n, 16) for n in [25, 11, 49, 3, 22]]
    labels = torch.tensor([0, 1, 2, 0, 1])
    kept, dropped = packing.get_packs(bags, .5, 16, 2, None, 'cpu', need_attn_mask=False,
        labels=labels, residual=True, enable_drop=True, seq_downsampler=model.network.downsampler, all_pu=True)
    args = {'attn_mask': None, 'num_images': kept['num_feats'], 'batched_image_ids': kept['batched_feat_ids'],
            'batched_image_ids_1': kept['batched_feat_ids_1'], 'key_pad_mask': kept['key_pad_mask'],
            'key_pad_mask_no_cls': kept['key_pad_mask_no_cls'], 'cls_token_mask': None,
            'no_norm_pad': False, 'residual': False}
    actual = model.network.predictor(model.network.mil(kept['patches'], pack_args=args))
    expected = []
    for i in range(len(kept['patches'])):
        for k in range(int(kept['num_feats'][i])):
            mask = (kept['batched_feat_ids'][i] == k) & ~kept['key_pad_mask'][i]
            expected.append(model.network.predictor(model.network.mil(kept['patches'][i, mask][None])))
    torch.testing.assert_close(actual, torch.cat(expected), atol=2e-7, rtol=1e-5)
    assert torch.equal(torch.cat(kept['batched_labels']), labels)
    assert torch.equal(torch.cat(dropped['batched_labels']), labels)
    assert (kept['key_pad_mask']).any()


def test_pack_hyperslide_targets_focal_and_mixed_censor_likelihood():
    source('packmil')
    _, _, util, _, losses, _ = load_pack_sources()
    target = util.mixup_target_batched([torch.tensor([0, 0, 1]), torch.tensor([2, 2])],
                                      num_classes=3, target_task='sub', multi_label=True)
    torch.testing.assert_close(target, torch.tensor([[1., .5, 0.], [0., 0., 1.]]))
    logits = torch.tensor([[.2, -.3, .4], [.5, .1, -.8]], requires_grad=True)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    expected = (.25*(1-bce.neg().exp()).square()*bce).mean()
    torch.testing.assert_close(losses.FocalLoss()(logits, target), expected)
    targets = [torch.tensor([[0, 1], [2, 0], [3, 1]]),
               torch.tensor([[0, 0], [1, 1]]), torch.tensor([[1, 1], [3, 1]])]
    mixed = util.mixup_target_batched(targets, num_classes=4, target_task='surv', multi_label=False)
    torch.testing.assert_close(mixed['Y'].flatten(), torch.tensor([2., 0., 1.]))
    torch.testing.assert_close(mixed['Y_censored'].flatten(), torch.tensor([0., 0., 1.]))
    torch.testing.assert_close(mixed['c'].flatten(), torch.tensor([2/3, 0., 1.]))
    risk = torch.randn(3, 4, requires_grad=True)
    h = risk.sigmoid(); s = (1-h).cumprod(1)
    expect = (-(1-2/3)*(s[0, 1]*h[0, 2]).log()-(2/3)*s[0, 0].log()-h[1, 0].log()-s[2, 1].log())/3
    actual = losses.NLLSurvMulLoss()(logits=risk, **mixed)
    torch.testing.assert_close(actual, expect)
    torch.testing.assert_close(torch.autograd.grad(actual, risk)[0], torch.autograd.grad(expect, risk)[0])


def test_pack_partition_complement_short_duplication_and_invalid_empty_branch():
    source('packmil')
    _, pack, *_ = load_pack_sources()
    bags = [torch.arange(n).float()[:, None].repeat(1, 4) for n in [21, 3]]
    kept, drop = pack.get_packs(bags, .25, 48, 4, None, 'cpu', need_attn_mask=False,
                               labels=torch.tensor([0, 1]), residual=True, enable_drop=True)
    a = kept['patches'][0, (kept['batched_feat_ids'][0] == 0) & ~kept['key_pad_mask'][0], 0]
    b = drop['patches'][0, (drop['batched_feat_ids'][0] == 0) & ~drop['key_pad_mask'][0], 0]
    assert len(a) == 5 and len(b) == 16
    assert not set(a.tolist()) & set(b.tolist())
    assert set(a.tolist()) | set(b.tolist()) == set(range(21))
    # Below the minimum threshold the source explicitly sends the full bag to both branches.
    for data in [kept, drop]:
        short = data['patches'][0, (data['batched_feat_ids'][0] == 1) & ~data['key_pad_mask'][0], 0]
        torch.testing.assert_close(short, torch.arange(3).float())
    model = PackedMIL(16, token_dropout=.001, min_sequence=1, pack_length=16, downsample_ratio=2)
    with pytest.raises(ValueError, match='empty branch'):
        model.batch_objective([torch.randn(2, 16)], torch.tensor([1]))


def test_pack_patient_bins_tail_update_and_no_validation_fit(tmp_path):
    source('packmil')
    root = create_demo(tmp_path/'data', patients=20, survival=True)
    bundle = Bundle(root)
    model = PackedMIL(16, survival=True, hidden=32, batch_size=4,
                      pack_length=32, min_sequence=2, downsample_ratio=2)
    model.prepare(bundle, 2, 'cpu'); original = model.bin_edges.clone()
    bundle.rows.loc[bundle.rows.subset != 'train', 'time'] *= 1000
    model.prepare(bundle, 2, 'cpu'); torch.testing.assert_close(original, model.bin_edges, atol=0, rtol=0)
    optimizer = model.make_optimizer(.001, .0001)
    groups = units(bundle, 'train')[:5]
    assert all(len({r['global_patient_id'] for r in rows}) == 1 for rows in groups)
    results = packed_epoch(model, bundle, groups, optimizer, 'cpu', {'max_instances': None, 'grad_clip': 5.})
    assert len(results) == 2
    assert all(v['step'] == 2 for v in optimizer.state.values())
    with pytest.raises(ValueError, match='batch_objective'):
        model(torch.randn(12, 16))
    # Ordinal/PANDA mixing is a separate author protocol, not silently used for generic classes.
    bundle.task_type = 'ordinal'
    with pytest.raises(ValueError, match='does not support'):
        check_compatibility(require_runnable('packmil_abmil'), bundle)


@pytest.mark.parametrize('method', ['mil_dropout_abmil', 'mil_dropout_gated', 'mil_dropout_dsmil',
                                  'mil_dropout_transmil', 'packmil_abmil', 'packmil_survival'])
def test_dropout_pack_exact_resume_prediction_and_weights(tmp_path, method):
    source('packmil' if method.startswith('packmil_') else 'mil_dropout')
    root = create_demo(tmp_path/'data', patients=20, survival=method == 'packmil_survival')
    options = ({'hidden': 32, 'batch_size': 4, 'pack_length': 32, 'min_sequence': 2, 'downsample_ratio': 2}
               if method.startswith('packmil_') else
               {'hidden': 32} if method in {'mil_dropout_dsmil', 'mil_dropout_transmil'} else {})
    config = {'epochs': 3, 'threads': 1, 'model': options}
    full = train(root, method, tmp_path/'full', config=config)
    partial = train(root, method, tmp_path/'partial', config=config, stop_after=1)
    train(root, method, partial, config=config, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in [full, partial]]
    assert a['history'] == b['history']
    torch.testing.assert_close(a['model'], b['model'], atol=0, rtol=0)
    assert evaluate(full) == evaluate(partial)
    pd.testing.assert_frame_equal(pd.read_csv(full/'test_predictions.csv'), pd.read_csv(partial/'test_predictions.csv'))
    feat = root/pd.read_csv(root/'features.csv').iloc[0].path
    with h5py.File(feat, 'r+') as f:
        f['features'][:] = f['features'][:]+.1
    with pytest.raises(ValueError, match='inputs differ'):
        evaluate(full)
