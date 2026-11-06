"""All26 native branches versus the actual original full training function."""
import contextlib
import copy
import io
import random
import sys
import types
from functools import lru_cache

import numpy as np
import pytest
import torch

from milplus.data import Bundle
from milplus.engine import rng_state, seed_everything
from milplus.miladapter import MILAdapter
from milplus.miladapter_demo import create_miladapter_demo
from milplus.miladapter_source import COMMIT, PROFILES, TEMPERATURE
from milplus.upstream import load_definitions, package, pinned_text, verified_source


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


@lru_cache(maxsize=2)
def original_modules(keep_width_port=False):
    root, sha = verified_source('miladapter'); assert sha == COMMIT
    prefix = '_miladapter_independent_original_reference'+('_keep_width_port' if keep_width_port else '')
    package(prefix, root/'utils'); parent = package(prefix+'.MIL', root/'utils/MIL')
    for name in ['mil_template', 'nystrom_attention', 'ABMIL', 'TransMIL', 'WIKGMIL', 'ILRAMIL', 'RRTMIL', 'adapters']:
        rel = 'utils/'+('' if name == 'adapters' else 'MIL/')+name+'.py'
        code = pinned_text(root, sha, rel)
        code = code.replace('from utils.MIL.', 'from '+prefix+'.MIL.')
        code = code.replace('from utils.MIL import', 'from '+prefix+'.MIL import')
        if name == 'adapters' and keep_width_port:
            code = code.replace('self.c_in, self.reduction, self.ratio = 512, 4, 0.2',
                                'self.c_in, self.reduction, self.ratio = text_embeddings.shape[0], 4, 0.2')
        module = types.ModuleType(prefix+'.'+name)
        module.__file__ = str(root/rel); sys.modules[module.__name__] = module
        if name != 'adapters':
            sys.modules[prefix+'.MIL.'+name] = module
        exec(compile(code, module.__file__, 'exec'), module.__dict__)
        if name in {'ABMIL', 'TransMIL', 'WIKGMIL'}:
            setattr(parent, name, getattr(module, name))
    # An explicit independent reference repair supplies the author's existing
    # temperature in the missing global. The executable source is unchanged.
    module.logit_scale = TEMPERATURE
    return module


def original_functions():
    functions = load_definitions('miladapter', 'utils/utils.py', ['shuffle_data', 'fewshot_sampling'],
        {'random': random, 'np': np})
    trainer = load_definitions('miladapter', 'utils/trainer.py', ['train_model'],
        {'torch': torch, 'shuffle_data': functions.shuffle_data})
    return functions, trainer


@pytest.fixture(scope='module', params=['CONCH', 'KEEP'])
def data(tmp_path_factory, request):
    return create_miladapter_demo(tmp_path_factory.mktemp('miladapter_source')/'data', encoder=request.param)


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_original_entire_trainer_and_all_native_parameter_updates(data, profile, monkeypatch):
    torch.set_num_threads(1)
    # The original hardcodes CUDA; CPU compatibility changes device only.
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *a, **k: self)
    bundle = Bundle(data)
    seed_everything(809)
    port = MILAdapter(bundle, profile, k_shots=2, sampling_seed=809)
    config = {'epochs': 2, 'lr': .001, 'weight_decay': 1e-5, 'threads': 1,
        'max_instances': None, 'grad_clip': None, 'selection_metric': 'miladapter_completed_epoch'}
    port.configure_training(config); optimizer = port.make_optimizer(.001, 1e-5)
    for epoch in range(2):
        port.start_epoch(epoch); port.epoch(bundle, optimizer, 'cpu')
    port_rng = rng_state()
    port_state = copy.deepcopy(port.network.state_dict())
    port_optimizer = copy.deepcopy(optimizer.state_dict())
    port_scheduler = copy.deepcopy(port.scheduler.state_dict())

    seed_everything(809)
    source, trainer = original_functions()
    pool = bundle.records('train'); xs = [bundle.miladapter.load(r).numpy() for r in pool]
    ys = np.array([r['target'] for r in pool])
    selected, _, labels, _ = source.fewshot_sampling(xs, ys, 2, seed=809)
    # Recover exact indices independently with identity-valued source inputs.
    chosen, _, _, _ = source.fewshot_sampling(list(range(len(xs))), ys, 2, seed=809)
    assert port.support_ids == [pool[i]['slide_id'] for i in chosen]
    route, aggregator, init = PROFILES[profile]
    kwargs = {'text_embeddings': bundle.miladapter.prototypes, 'aggregator': aggregator}
    if route == 'ZSMIL':
        kwargs['init'] = init
    if route == 'TIPAdapter':
        kwargs.update(train_data=selected, train_labels=labels)
    reference = getattr(original_modules(keep_width_port=bundle.dim == 768), route)(**kwargs)
    if route in {'CLIPAdapter', 'TIPAdapter'} and aggregator == 'ABMIL':
        original_forward = reference.ABMIL.forward
        reference.ABMIL.forward = lambda x: original_forward(x)[0]
    optim = torch.optim.AdamW(reference.parameters(), lr=.001, betas=(.9, .999), weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=2)
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.train_model(reference, optim, torch.nn.CrossEntropyLoss(reduction='sum'), sched, selected, labels, 2)
    # No tolerance: all original network parameters, moments, scheduler and
    # complete global RNG transition must be bitwise identical.
    for key, value in reference.state_dict().items():
        equal(value, port_state[key])
    equal(reference.text_embeddings, port_state['text_embeddings'])
    equal(optim.state_dict(), port_optimizer)
    equal(sched.state_dict(), port_scheduler)
    equal(rng_state(), port_rng)
    assert port.optimizer_updates == 8


def test_public_source_missing_temperature_and_tuple_defects_remain_documented(data, monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *a, **k: self)
    bundle = Bundle(data); source = original_modules(); x = bundle.miladapter.load(bundle.records('train')[0])
    temperature = source.logit_scale
    del source.logit_scale
    try:
        broken = source.ZSMIL(bundle.miladapter.prototypes, init='ZS', aggregator='BGAP')
        with pytest.raises(NameError, match='logit_scale'):
            broken(x)
    finally:
        source.logit_scale = temperature
    model = source.CLIPAdapter(bundle.miladapter.prototypes, aggregator='ABMIL')
    with pytest.raises(TypeError):
        model(x)


def test_task_residual_and_tip_cache_equations_use_actual_support_values(data):
    bundle = Bundle(data)
    residual = MILAdapter(bundle, 'miladapter_taskres_bgap', k_shots=2, sampling_seed=6)
    assert torch.equal(residual.network.classifier, torch.from_numpy(bundle.miladapter.prototypes))
    assert torch.equal(residual.network.text_embeddings, residual.network.classifier)
    tip = MILAdapter(bundle, 'miladapter_tipadapter_transmil', k_shots=2, sampling_seed=6)
    expected = np.stack([np.mean(bundle.miladapter.load(r).numpy(), axis=0) for r in tip.support])
    assert torch.equal(tip.network.cache_keys, torch.from_numpy(expected))
    assert tip.network.cache_keys.requires_grad and not tip.network.cache_values.requires_grad
    assert not tip.network.classifier.requires_grad and tip.network.beta == 5 and tip.network.alpha == 1
    assert tip.network.cache_values.argmax(1).tolist() == [r['target'] for r in tip.support]
