"""Full original train-loop/state equivalence and independent spline semantics."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

from milplus.conmil_demo import create_conmil_demo
from milplus.conmil_source import build_model, definitions
from milplus.data import Bundle
from milplus.engine import train, seed_everything, rng_state
from milplus.io import read_json
from milplus.upstream import load_definitions


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def source_trainer():
    source = definitions()
    return load_definitions('conmil', 'train_text_guided_mil_kan.py',
        ['BagDataset', 'collate_fn', 'make_loader', 'compute_metrics', 'train_epoch', 'eval_epoch', 'train'],
        {'os': __import__('os'), 'json': __import__('json'), 'np': np, 'pd': pd,
         'torch': torch, 'nn': nn, 'optim': torch.optim, 'Dataset': Dataset, 'DataLoader': DataLoader,
         'roc_auc_score': metrics.roc_auc_score, 'accuracy_score': metrics.accuracy_score,
         'f1_score': metrics.f1_score, 'average_precision_score': metrics.average_precision_score,
         'confusion_matrix': metrics.confusion_matrix,
         'build_text_guided_mil_kan': source.build_text_guided_mil_kan},
        replacements=[('return best_ckpt, run_name',
                       'return best_ckpt, run_name, model, optimizer, scheduler, history')])


@pytest.mark.parametrize('negative,grid', [(False, 5), (True, 5), (True, 6)])
def test_full_actual_original_train_weights_adam_cosine_selection_rng(tmp_path, negative, grid):
    torch.set_num_threads(2)
    data = create_conmil_demo(tmp_path/'data', patients=10, patches=4097)
    profile = 'conmil_posneg_source' if negative else 'conmil_pos_source'
    run = train(data, profile, tmp_path/'adapted', seed=53,
        config={'epochs': 2, 'threads': 2, 'model': {'kan_grid': grid}})
    checkpoint = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
    best = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    final_rng = rng_state()
    bundle = Bundle(data)
    args = SimpleNamespace(feat_dir=str(data/'features'), num_workers=0, in_dim=512,
        hidden_dim=256, n_classes=2, dropout=.25, alpha_init=.5, use_neg=negative,
        neg_weight=.5, kan_grid=grid, lr=2e-4, weight_decay=1e-5, epochs=2, output=str(tmp_path/'author'))
    (tmp_path/'author').mkdir()
    frames = {subset: pd.DataFrame([{'image': row['slide_id']+'.svs',
        'type': 'tumor' if row['target'] else 'normal'} for row in bundle.records(subset)])
        for subset in ['train', 'val']}
    author = source_trainer()
    seed_everything(53)
    path, _, network, optimizer, scheduler, history = author.train(args,
        frames['train'], frames['val'], 'cpu', bundle.conmil.text.positive, bundle.conmil.text.negative)
    expected = {k.removeprefix('network.'): v for k, v in checkpoint['model'].items() if k.startswith('network.')}
    equal(expected, network.state_dict()); equal(checkpoint['optimizers'][0], optimizer.state_dict())
    equal(checkpoint['model']['_extra_state']['scheduler'], scheduler.state_dict())
    equal(final_rng, rng_state())
    expected_best = {k.removeprefix('network.'): v for k, v in best['model'].items() if k.startswith('network.')}
    equal(expected_best, torch.load(path, map_location='cpu', weights_only=False))
    selected = [r['epoch']-1 for r in history if r['is_best']]
    assert best['epoch'] == selected[-1]
    assert [round(r['train']['loss'], 6) for r in checkpoint['history']] == [r['train_loss'] for r in history]
    assert all(len(sample['indices']) == 4096 for group in checkpoint['model']['_extra_state']['samples_history'] for sample in group)
    assert all(r['validation']['conmil_val_accuracy'] == h['ACC'] for r, h in zip(checkpoint['history'], history))
    assert read_json(run/'method_protocol.json')['native']['hidden'] == 256


def test_immutable_grid_storage_port_keeps_forward_and_gradients_bitwise():
    torch.set_num_threads(2); source = definitions()
    torch.manual_seed(17); original = source.build_text_guided_mil_kan(use_neg=True)
    torch.manual_seed(17); portable = build_model(2, 'conmil_posneg_source')
    with pytest.raises(RuntimeError, match='single memory location'):
        # Actual checkpoint tensors have separate storage; a self-copy aliases
        # the exact same buffer and PyTorch legitimately treats it as a no-op.
        original.load_state_dict({key: value.clone() for key, value in original.state_dict().items()})
    portable.load_state_dict(original.state_dict())
    x, pos, neg = torch.randn(17, 512), torch.randn(4, 512), torch.randn(2, 512)
    original.eval(); portable.eval()
    a, b = original(x, pos, neg), portable(x, pos, neg)
    equal(a, b)
    nn.functional.cross_entropy(a[0], torch.tensor([1])).backward()
    nn.functional.cross_entropy(b[0], torch.tensor([1])).backward()
    for (name, p), (other, q) in zip(original.named_parameters(), portable.named_parameters()):
        assert name == other; equal(p.grad, q.grad)
    for name, value in original.state_dict().items():
        equal(value, portable.state_dict()[name])


def test_exact_positive_minus_negative_guidance_and_cubic_partition():
    source = definitions(); model = build_model(2, 'conmil_posneg_source')
    model.eval(); x, pos, neg = torch.randn(17, 512), torch.randn(4, 512), torch.randn(2, 512)
    logits, raw, similarity = model(x, pos, neg)
    expected = (x@pos.T).mean(1, keepdim=True)-.5*(x@neg.T).mean(1, keepdim=True)
    h = model.feat_proj(x)
    guided = model.attention_W(torch.tanh(model.attention_V(h))*torch.sigmoid(model.attention_U(h)))
    guided = guided+model.alpha*model.sim_proj(expected)
    equal(similarity, expected.squeeze()); equal(raw, guided.squeeze())
    equal(logits, model.classifier((guided.softmax(0)*h).sum(0, keepdim=True)))
    layer = source.KANLinear(3, 2)
    basis = layer.b_splines(torch.tensor([[-.73, .11, .63], [.22, -.52, -.13]]))
    assert basis.shape == (2, 3, 8)
    torch.testing.assert_close(basis.sum(-1), torch.ones(2, 3), rtol=1e-6, atol=1e-6)
    assert bool((basis >= 0).all())


def test_original_binary_and_invalid_grid_constraints():
    with pytest.raises(ValueError, match='binary'):
        build_model(3, 'conmil_posneg_source')
    with pytest.raises(ValueError, match='positive integer'):
        build_model(2, 'conmil_posneg_source', kan_grid=0)
