"""RankMix source-step equivalence, indexing counterexamples and two-stage runs."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train
from milplus.io import read_json
from milplus.models import build_model
from milplus.rankmix import RankMix, author_definitions, selected_ids
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import load_definitions

PROFILES = ['rankmix_dsmil', 'rankmix_dsmil_source', 'rankmix_frmil', 'rankmix_frmil_source']
torch.set_num_threads(1)


def reference(repaired=False):
    replacements = [('.cuda()', ".to('cpu')")]
    if repaired:
        replacements.append(('torch.sort(indices, 0)[1]', 'torch.sort(indices, 0)[0]'))
        replacements.append(('_, ins_prediction = milnet(feats) # (2,N)',
            ("_, ins_prediction = milnet(feats) # (2,N)\n"
            "            ins_prediction = ins_prediction.clone()\n"
            "            for row in range(2):\n"
            "                invalid = float('inf') if strategy.startswith('rank2') and label[row].item()==0 else -float('inf')\n"
            "                ins_prediction[row, zero_idx[row]:] = invalid")))
    return load_definitions('rankmix', 'train_mil_rankmix.py',
        ['mixup_features', 'mixup_label', 'mixup', 'mixup_by_rank', 'get_mix_strategy', 'train', 'train_frmil'],
        {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'Tuple': tuple,
         'FeatMag': author_definitions()[3].FeatMag, 'sys': SimpleNamespace(stdout=SimpleNamespace(write=lambda _: None))},
        replacements=replacements)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('direction', ['rank', 'rank2'])
def test_author_full_rankmix_step_value_gradient_and_adam(profile, direction):
    model = build_model(profile, 16, 2, {'warmup_epochs': 1, 'rank_direction': direction, 'keep_percent': 90})
    ref = reference(repaired=not model.literal)
    author = copy.deepcopy(model.network)
    optim = torch.optim.Adam(model.network.parameters(), lr=.0002, betas=(.5,.9), weight_decay=.005)
    original_optim = torch.optim.Adam(author.parameters(), lr=.0002, betas=(.5,.9), weight_decay=.005)
    labels = torch.tensor([0., 1.])
    bags = [torch.randn(11,16), torch.randn(7,16)]
    args = SimpleNamespace(mix_algorithm='mixup', mix_domain='feature', do_mixup=.1,
        feats_size=16, mix_strategy=direction+'_90', top_k='min', loss='FrmilLoss', mag=10.17, num_classes=1)
    criterion = nn.BCEWithLogitsLoss()
    torch.manual_seed(232); np.random.seed(123)
    if model.backbone == 'dsmil':
        loader = [((labels[:1].reshape(1,1), bags[0][None]), (labels[1:].reshape(1,1), bags[1][None]))]
        expected = ref.train(loader, author, None, criterion, original_optim, args)
    else:
        features = torch.stack((bags[0], F.pad(bags[1], (0,0,0,4))))
        class Loader(list):
            dataset = SimpleNamespace(count_dict={0:1,1:1}, pos_weight=torch.tensor(1.))
        class Data:
            def __init__(self):
                self.count_dict = {0:1,1:1}
                self.pos_weight = torch.tensor(1.)
            def __len__(self):
                return 1
        loader = Loader([(features, labels.reshape(2,1), None, torch.tensor([11,7]))])
        loader.dataset = Data()
        expected = ref.train_frmil(loader, author, None, criterion, original_optim, args)
    torch.manual_seed(232); np.random.seed(123)
    lam = float(np.random.beta(.1, 1 if model.backbone == 'dsmil' else .1))
    if model.backbone == 'dsmil':
        mixed, target = model.mix_pair(bags, labels, lam)
        result = model.dsmil_objective(mixed, target)
    else:
        mixed, target = model.mix_pair(features, labels, lam, [11,7])
        result = model.frmil_objective(torch.cat((features, mixed[None])),
                                       torch.cat((labels, target.reshape(1))), mixed=True)
    assert float(result['loss'].detach()) == expected
    result['loss'].backward()
    for (name, p), (_, q) in zip(model.network.named_parameters(), author.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    optim.step()
    for p, q in zip(model.network.parameters(), author.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_literal_index_bug_and_repair_counterexample():
    scores = torch.tensor([.1,.8,.2,.9,.3])
    # Correct selected identities are 1 and 3; source argsort gives positions 1,0.
    assert selected_ids(scores, 2).tolist() == [1,3]
    assert selected_ids(scores, 2, literal=True).tolist() == [1,0]
    assert selected_ids(scores, 2, descending=False).tolist() == [0,2]


def test_frmil_repair_excludes_high_scoring_padding_and_restores_mode():
    model = RankMix(8, 2, backbone='frmil', warmup_epochs=1)
    class Scores(nn.Module):
        def forward(self, x):
            # Padded positions deliberately score higher than genuine instances.
            return torch.zeros(2,1), torch.tensor([[.1,.3,.4,.2,99,99],[.3,.2,.1,.4,99,99]])
    model.network = Scores().eval()
    bag = torch.arange(48.).reshape(6,8)
    out, label = model.mix_pair([bag, bag+100], torch.tensor([0.,1.]), .25, [4,4])
    torch.testing.assert_close(out[:4], bag[:4]+25)
    assert torch.count_nonzero(out[4:]) == 0
    assert label == .25 and not model.network.training


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('stop_after', [1, 3])
def test_two_stage_exact_resume_and_best_reload(tmp_path, profile, stop_after):
    data = create_demo(tmp_path/'data', patients=20)
    cfg = {'epochs':4, 'patience':10, 'threads':1, 'model':{'warmup_epochs':2}}
    full = train(data, profile, tmp_path/'full', config=cfg)
    part = train(data, profile, tmp_path/'part', config=cfg, stop_after=stop_after)
    train(data, profile, part, config=cfg, resume=True)
    a, b = [torch.load(p/'last.pt', weights_only=False) for p in (full, part)]
    assert a['history'] == b['history']
    for key in a['model']:
        if key != '_extra_state':
            torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
    for state in [a,b]:
        assert state['model']['_extra_state']['stage_transitions'] == 1
        assert [h['train']['rankmix_stage'] for h in state['history']] == [0,0,1,1]
    assert torch.load(full/'best.pt', weights_only=False)['epoch'] >= 2
    assert not (full/'test_predictions.csv').exists()
    assert evaluate(full) == evaluate(part)
    first = pd.read_csv(full/'test_predictions.csv')
    evaluate(full)
    pd.testing.assert_frame_equal(first, pd.read_csv(full/'test_predictions.csv'))
    assert read_json(full/'method_protocol.json')['warmup_epochs'] == 2


@pytest.mark.parametrize('backbone', ['dsmil','frmil'])
def test_author_bag_only_prediction_and_warmup_objective(backbone):
    model = RankMix(16, 2, backbone=backbone, warmup_epochs=1)
    x, y = torch.randn(2,11,16), torch.tensor([0.,1.])
    model.eval()
    p = model(x[0])['logits'].softmax(-1)[0,1]
    author_positive = model.network(x[0])[1] if backbone == 'dsmil' else model.network(x[:1])[0]
    torch.testing.assert_close(p, author_positive.sigmoid().squeeze())
    model.train(); torch.manual_seed(55)
    if backbone == 'dsmil':
        actual = model.dsmil_objective(x[0], y[0])
        torch.manual_seed(55)
        ins, bag, _, _ = model.network(F.dropout(x[0], p=.2))
        expected = (F.binary_cross_entropy_with_logits(ins.max(0).values, y[:1]) +
                    F.binary_cross_entropy_with_logits(bag.flatten(), y[:1]))/2
    else:
        actual = model.frmil_objective(x, y)
        torch.manual_seed(55)
        bag, query, ins = model.network(F.dropout(x, p=.2))
        mag = author_definitions()[3].FeatMag(10.17)(query[1:], query[:1], w_scale=11)
        expected = (F.binary_cross_entropy_with_logits(bag.flatten(), y) +
                    F.binary_cross_entropy(ins.max(1).values, y) + mag)*(1./3)
    torch.testing.assert_close(actual['loss'], expected, rtol=0, atol=0)


def test_stage_boundary_restores_best_initialization_and_resets_adam():
    model = RankMix(16, 2, warmup_epochs=2)
    model.configure_training({'epochs':5,'selection_metric':'macro_auc','grad_clip':None})
    optimizer = model.make_optimizer(.001,.005)
    model.start_epoch(0)
    model.after_validation({'macro_auc':.9})
    best = copy.deepcopy(model.network.state_dict())
    loss = model.dsmil_objective(torch.randn(11,16), torch.tensor(1.))['loss']
    loss.backward(); optimizer.step(); model.finish_epoch(None,None,None)
    assert optimizer.state
    model.start_epoch(1); model.after_validation({'macro_auc':.2})
    model.start_epoch(2)
    assert not optimizer.state and optimizer.param_groups[0]['lr'] == .001
    assert model.scheduler.T_max == 3
    for key, value in model.network.state_dict().items():
        torch.testing.assert_close(value,best[key],rtol=0,atol=0)


@pytest.mark.parametrize('bad', [{'classes':3},{'warmup_epochs':0},{'alpha':0}, {'keep_percent':0},
                                {'keep_percent':.5},{'rank_direction':'random'},{'backbone':'frmil','heads':3}])
def test_invalid_recipe_rejected(bad):
    options = {'dim':16, 'classes':2, **bad}
    with pytest.raises(ValueError):
        RankMix(**options)


def test_warmup_only_run_and_patient_concatenation_rejected(tmp_path):
    data = create_demo(tmp_path/'data', patients=20)
    with pytest.raises(ValueError,match='warmup AND'):
        train(data, 'rankmix_dsmil', tmp_path/'bad', config={'epochs':1,'model':{'warmup_epochs':1}})
    bundle = Bundle(data)
    bundle.config['training_unit'] = 'patient'
    with pytest.raises(ValueError,match='individual slide'):
        check_compatibility(require_runnable('rankmix'),bundle)


def test_cosine_scheduler_resets_for_finetuning():
    model = RankMix(8,2,warmup_epochs=2)
    model.configure_training({'epochs':6,'selection_metric':'macro_auc','grad_clip':None})
    optimizer = model.make_optimizer(.001,.005)
    for epoch in range(6):
        model.start_epoch(epoch)
        t, horizon = (epoch,2) if epoch < 2 else (epoch-2,4)
        expected = .000005+(.001-.000005)*(1+math.cos(math.pi*t/horizon))/2
        assert optimizer.param_groups[0]['lr'] == pytest.approx(expected)
        optimizer.step(); model.finish_epoch(None,None,None)
        model.after_validation({'macro_auc':.6})


def test_padding_preparation_reads_only_training_and_empty_rank_rejected(tmp_path, monkeypatch):
    bundle = Bundle(create_demo(tmp_path/'data', patients=20))
    model = RankMix(bundle.dim, 2, backbone='frmil', warmup_epochs=1, keep_percent=1)
    records = bundle.records
    def training_only(subset):
        assert subset == 'train'
        return records(subset)
    monkeypatch.setattr(bundle, 'records', training_only)
    model.prepare(bundle, 1, 'cpu')
    expected = max(len(bundle.load(row)[0]) for row in records('train'))
    assert int(model.padding_length) == expected
    with pytest.raises(ValueError,match='no instances'):
        model.mix_pair([torch.randn(2,bundle.dim)]*2, torch.tensor([0.,1.]), .5)
