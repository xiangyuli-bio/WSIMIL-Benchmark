import ast
import copy
import hashlib
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint

from milplus.kmgmamba_source import LegacyDropPath, build_model, scan_backend, source_svm
from milplus.upstream import load_definitions, source_root


@pytest.fixture(autouse=True)
def runtime():
    if not source_root('kmgmamba').is_dir():
        pytest.skip('Fetch the pinned KMG-Mamba source')
    torch.set_num_threads(1)


def author_drop():
    path = Path(__file__).parent/'fixtures/kmgmamba/timm_v0_4_12_drop.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '99cd8bf3d9e503ca4633264141fda078662e4bb7e94367a17c7a5fda646eab4d'
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {'drop_path', 'DropPath'}]
    scope = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), scope)
    return scope['DropPath']


def author_scan():
    return load_definitions('kmgmamba', 'models/mamba/mamba_ssm/ops/selective_scan_interface.py',
        ['selective_scan_ref'], {'torch': torch, 'F': F, 'rearrange': rearrange, 'repeat': repeat}).selective_scan_ref


def author_model(dimension, classes, monkeypatch):
    # The exact author graph is evaluated on CPU only by redirecting .cuda();
    # no architecture, forward, loss, scale or parameter is rewritten here.
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    monkeypatch.setattr(nn.Module, 'cuda', lambda self, *a, **k: self)
    scope = dict(torch=torch, nn=nn, F=F, np=np, math=math, partial=partial,
        Callable=Callable, Any=Any, checkpoint=checkpoint, rearrange=rearrange, repeat=repeat,
        DropPath=author_drop(), selective_scan_fn_v1=author_scan())
    cross = load_definitions('kmgmamba', 'models/vmamba.py',
        ['Cross_Mamba_Attention_SSM', 'CrossMambaFusion_SS2D_SSM', 'Mlp', 'CrossMambaFusionBlock'], scope)
    attention = load_definitions('kmgmamba', 'models/datten.py',
        ['Attention', 'AttentionGated', 'DAttention'], scope)
    scope.update(CrossMambaFusionBlock=cross.CrossMambaFusionBlock, DAttention=attention.DAttention,
                 SmoothTop1SVM=source_svm())
    model = load_definitions('kmgmamba', 'models/model_MG_Mamba.py',
        ['initialize_weights', 'Embeddings', 'WiKG', 'GraphRepresentation', 'CrossMamba', 'MG_Mamba_main'], scope)
    return model.MG_Mamba_main(SimpleNamespace(input_size=dimension, hidden_size=96),
                              img_size=3000, num_classes=classes, sequence_length_min=0)


@pytest.mark.parametrize('dimension', [1024, 1536])
def test_unmodified_author_full_graph_loss_input_and_all_parameter_gradients(dimension, monkeypatch):
    torch.manual_seed(743)
    port, _ = build_model(dimension, 2)
    original = author_model(dimension, 2, monkeypatch)
    original.load_state_dict(copy.deepcopy(port.state_dict()))
    x, y = torch.randn(31, dimension, requires_grad=True), torch.randn(73, dimension, requires_grad=True)
    a, b = x.detach().clone().requires_grad_(), y.detach().clone().requires_grad_()
    state = torch.get_rng_state()
    actual = port(x, y, torch.tensor([1])); after = torch.get_rng_state()
    torch.set_rng_state(state)
    expected = original(a, b, torch.tensor([1]))
    assert torch.equal(after, torch.get_rng_state())
    for left, right in zip(actual[:3], expected):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(actual[2], .5*actual[4]+.5*(actual[5]+actual[6]), atol=0, rtol=0)
    actual[2].backward(); expected[2].backward()
    torch.testing.assert_close(x.grad, a.grad, atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(y.grad, b.grad, atol=1e-6, rtol=2e-5)
    unused = []
    for (name, p), (other, q) in zip(port.named_parameters(), original.named_parameters()):
        assert name == other and (p.grad is None) == (q.grad is None)
        if p.grad is None:
            unused.append(name)
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=2e-5)
    assert unused == ['transformer.cross_mamba.op.conv2d.weight', 'transformer.cross_mamba.op.conv2d.bias',
        'transformer.cross_mamba.norm2.weight', 'transformer.cross_mamba.norm2.bias',
        'transformer.cross_mamba.mlp.fc1.weight', 'transformer.cross_mamba.mlp.fc1.bias',
        'transformer.cross_mamba.mlp.fc2.weight', 'transformer.cross_mamba.mlp.fc2.bias',
        'transformer.norm.weight', 'transformer.norm.bias']
    assert port.transformer.encoder.layer1.topk == 9
    assert port.transformer.cross_mamba.op.CMA_ssm.d_state == 4
    assert port.transformer.cross_mamba.op.CMA_ssm.d_inner == 96


@pytest.mark.parametrize('seed', [0, 1, 6, 17, 123])
def test_legacy_drop_path_matches_exact_original_timm_rng(seed):
    x = torch.linspace(-2, 3, 120).reshape(3, 10, 4)
    original = author_drop()(.2).train(); port = LegacyDropPath(.2).train()
    torch.manual_seed(seed); actual = port(x); after = torch.get_rng_state()
    torch.manual_seed(seed); expected = original(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(after, torch.get_rng_state())


def test_source_binary_heads_fail_multiclass_and_explicit_repair_covers_all_classes(monkeypatch):
    original = author_model(1024, 3, monkeypatch).eval()
    x, y = torch.randn(9, 1024), torch.randn(36, 1024)
    with pytest.raises(IndexError):
        original(x, y, torch.tensor([2]))
    with pytest.raises(ValueError, match='binary'):
        build_model(1024, 3, 'kmgmamba_source')
    repaired, _ = build_model(1024, 3, 'kmgmamba_multiclass_repaired')
    for target in range(3):
        repaired.zero_grad()
        result = repaired(x, y, torch.tensor([target])); result[2].backward()
        assert result[0].shape == (1, 3) and result[2].isfinite()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in repaired.classifiers.parameters())
        assert repaired.instance_classifier.weight.grad.abs().sum() > 0


def test_unlabeled_inference_and_both_scales_are_used():
    network, _ = build_model(1536, 2); network.eval()
    x, y = torch.randn(9, 1536), torch.randn(37, 1536)
    with torch.no_grad():
        a, b = [network(x, y, torch.tensor([c])) for c in [0, 1]]
        changed_low = network(x+torch.randn_like(x), y, torch.tensor([0]))
        changed_high = network(x, y+torch.randn_like(y), torch.tensor([0]))
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    assert not torch.equal(a[0], changed_low[0]) and not torch.equal(a[0], changed_high[0])


def test_explicit_scan_backend_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError):
        scan_backend('auto', 64)
    with pytest.raises(RuntimeError, match='without CUDA'):
        scan_backend('cuda', 64)
