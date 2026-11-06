"""Full DAE/source training-step fidelity, clean TEST and explicit state contracts."""
import ast
import copy
import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from milplus.augdiff import PROFILES, AugDiffMIL
from milplus.augdiff_assets import export_trained_dae
from milplus.augdiff_data import AugDiffInputs
from milplus.augdiff_demo import create_augdiff_demo
from milplus.augdiff_pretrain import DiffusionTrainer, restore_rng, rng_state, training_contract
from milplus.augdiff_source import definitions, sampler_class
from milplus.data import Bundle
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(scope='module')
def augdiff_bundle(tmp_path_factory):
    supplied = os.environ.get('MILPLUS_AUGDIFF_TEST_BUNDLE')
    if supplied:
        bundle = Bundle(supplied)
        assert bundle.config.get('synthetic') is True, 'Point tests at an explicitly synthetic software fixture'
        assert bundle.augdiff is not None
        assert bundle.augdiff.metadata['pretrain_protocol']['optimizer_updates'] > 0
        return bundle
    base = tmp_path_factory.mktemp('actual_augdiff_preparation')
    if shutil.disk_usage(base).free < 6*1024**3:
        raise RuntimeError('Full AugDiff preparation needs6GiB scratch; choose a larger --basetemp or set MILPLUS_AUGDIFF_TEST_BUNDLE to a verified actual prepared fixture')
    root = create_augdiff_demo(base/'data', patients=18, dimension=512, instances=2)
    torch.set_num_threads(1)
    inputs = AugDiffInputs(Bundle(root), 'augdiff_views.json')
    trainer = DiffusionTrainer(inputs, epochs=1, batch_size=14)
    trainer.fit(base/'pretrain')
    del trainer
    manifest = export_trained_dae(Bundle(root), 'augdiff_views.json', base/'pretrain', root/'dae.json')
    path = root/'task.yaml'; task = yaml.safe_load(path.read_text())
    task['encoders']['toy']['augdiff'] = {'manifest': manifest}
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return Bundle(root)


@pytest.fixture(scope='module')
def frozen_dae(augdiff_bundle):
    return augdiff_bundle.augdiff.load_dae('cpu')


def source_training_step(dae, reference, repaired):
    """Actual unmodified author method body, with the actual full trained DAE."""
    root, sha = verified_source('augdiff')
    tree = ast.parse(pinned_text(root, sha, 'MIL/models/model_interface_uncondition.py'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelInterfaceUncondition')
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'training_step')
    scope = {'torch': torch, 'sampler': sampler_class(repaired)(dae), 'aug_model': dae}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'actual_original_AugDiff_training_step.py', 'exec'), scope)  # noqa: S102 -- actual fixed Git definition
    holder = SimpleNamespace(total_steps=20, scale=.4, model=reference, loss=torch.nn.CrossEntropyLoss())
    return lambda batch: scope['training_step'](holder, batch, 0)


@pytest.mark.parametrize('profile', list(PROFILES))
def test_augdiff_original_training_step_all_classifier_gradients_and_RNG(augdiff_bundle, frozen_dae, profile):
    torch.set_num_threads(1)
    model = AugDiffMIL(augdiff_bundle, profile)
    object.__setattr__(model, '_dae', frozen_dae)
    name = {'amil': 'AMIL', 'lossattn': 'LossAttn', 'dsmil': 'DSMIL'}[model.head]
    module = definitions('augdiff', 'MIL/models/'+name+'.py', replacements=[('.cuda()', '')] if model.head == 'dsmil' else [])
    original = getattr(module, name)(feats_size=augdiff_bundle.dim, n_classes=len(augdiff_bundle.classes))
    original.load_state_dict(model.network.state_dict(), strict=True)
    row = augdiff_bundle.records('train')[0]
    x, coords = augdiff_bundle.load(row)
    assert len(x) >= 2, 'Original source-step diagnostic requires at least2 software rows; N1 rank port has its own check'
    # Bounded original-step diagnostic; the end-to-end gate separately consumes
    # every row and all seven complete TRAIN passes without slicing.
    raw = torch.cat((coords if coords is not None else torch.zeros(len(x), 2), x), dim=-1)[None]
    batch = (raw, torch.tensor([row['target']]))
    step = source_training_step(frozen_dae, original, model.repaired)
    before = rng_state()
    actual = model.result(row, 'cpu', augmented=True)
    after = rng_state()
    restore_rng(before)
    expected = step(batch)
    assert torch.equal(actual['loss'], expected['loss'])
    assert torch.equal(torch.get_rng_state(), after['torch'])
    assert np.array_equal(np.random.get_state()[1], after['numpy'][1])
    actual['loss'].backward(); expected['loss'].backward()
    for (name_a, a), (name_b, b) in zip(model.network.named_parameters(), original.named_parameters()):
        assert name_a == name_b
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad), name_a
    assert all(not p.requires_grad and p.grad is None for p in frozen_dae.parameters())
    assert not any(name.startswith('_dae') for name in model.state_dict())


def test_augdiff_clean_test_never_allocates_or_calls_DAE(augdiff_bundle):
    model = AugDiffMIL(augdiff_bundle, next(iter(PROFILES)))
    frame = model.predict_subset(augdiff_bundle, 'test', 'cpu', {})
    assert model._dae is None
    assert len(frame) == len(augdiff_bundle.records('test'))
    assert frame.slide_id.is_unique and (frame.augdiff_draws == 1).all()
    assert np.isfinite(model.validation_metrics(frame)['augdiff_source_ce'])


def test_augdiff_validation_has7draws_but_one_true_case_and_exact_replay(augdiff_bundle, frozen_dae):
    model = AugDiffMIL(augdiff_bundle, 'augdiff_amil_schedule_repaired_reconstruction')
    object.__setattr__(model, '_dae', frozen_dae)
    model.start_epoch(0)
    first = model.predict_subset(augdiff_bundle, 'val', 'cpu', {})
    stream = rng_state()
    second = model.predict_subset(augdiff_bundle, 'val', 'cpu', {})
    pd_values = first.to_csv(index=False)
    assert second.to_csv(index=False) == pd_values
    assert torch.equal(torch.get_rng_state(), stream['torch'])
    assert first.slide_id.is_unique and (first.augdiff_draws == 7).all()
    assert len(first) == len(augdiff_bundle.records('val'))


def test_augdiff_cached_sampler_retains_source_draws_and_all_RNG(augdiff_bundle, frozen_dae):
    x, _ = augdiff_bundle.load(augdiff_bundle.records('train')[0])
    cache = sampler_class(True)
    literal = sampler_class.__wrapped__(True)
    before = rng_state()
    a, _ = cache(frozen_dae).sample(S=20, batch_size=len(x), shape=[x.shape[1], 1],
                                  nu=.4, x_T=x, eta=0., verbose=False)
    after = torch.get_rng_state()
    restore_rng(before)
    b, _ = literal(frozen_dae).sample(S=20, batch_size=len(x), shape=[x.shape[1], 1],
                                    nu=.4, x_T=x, eta=0., verbose=False)
    assert torch.equal(a, b) and torch.equal(after, torch.get_rng_state())
    assert sampler_class(True) is cache


def test_augdiff_pretraining_identity_uses_recorded_Torch_build(augdiff_bundle, monkeypatch):
    assets = augdiff_bundle.augdiff
    protocol = assets.metadata['pretrain_protocol']
    args = (assets.views, protocol['max_epochs'], protocol['batch_size'], protocol['precision'],
            protocol['seed'], protocol['device_type'], protocol['torch_version'], protocol['backend_flags'])
    before = training_contract(*args)
    monkeypatch.setattr(torch, '__version__', '2.5.1+different_consumer_build')
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', not protocol['backend_flags']['cudnn_deterministic'])
    assert training_contract(*args) == before == assets.metadata['pretrain_contract']


def test_augdiff_fresh_validation_records_stream_before_replay(augdiff_bundle, frozen_dae):
    model = AugDiffMIL(augdiff_bundle, 'augdiff_amil_schedule_repaired_reconstruction')
    object.__setattr__(model, '_dae', frozen_dae)
    frame = model.predict_subset(augdiff_bundle, 'val', 'cpu', {})
    assert frame.slide_id.is_unique and (frame.augdiff_draws == 7).all()
    assert model.validation_rng is not None


@pytest.mark.parametrize('settings', [{'steps': 1}, {'nu': 0}, {'precision': 'unknown'}, {'max_dense_bytes': 0}])
def test_augdiff_does_not_silently_disable_or_shrink_augmentation(augdiff_bundle, settings):
    with pytest.raises(ValueError):
        AugDiffMIL(augdiff_bundle, 'augdiff_amil_schedule_repaired_reconstruction', **settings)


def test_augdiff_full_bag_and_original_selection_are_required(augdiff_bundle):
    model = AugDiffMIL(augdiff_bundle, next(iter(PROFILES)))
    good = {'max_instances': None, 'grad_clip': None, 'selection_metric': 'augdiff_source_ce'}
    model.configure_training(good)
    for changed in [{'max_instances': 1}, {'grad_clip': 1.}, {'selection_metric': 'macro_auc'}]:
        with pytest.raises(ValueError):
            model.configure_training({**good, **changed})


def test_augdiff_checkpoint_rejects_changed_asset_contract(augdiff_bundle):
    model = AugDiffMIL(augdiff_bundle, next(iter(PROFILES)))
    state = copy.deepcopy(model.get_extra_state()); state['contract'] = '0'*64
    with pytest.raises(ValueError, match='contract'):
        model.set_extra_state(state)


def test_catalog_regeneration_preserves_all_existing_execution_contracts(tmp_path):
    # VLSA was previously registered manually and could disappear on catalog
    # regeneration. Audit all existing routes, not only the new family count.
    root = Path(__file__).resolve().parents[1]
    target = tmp_path/'isolated_catalog'
    resources = target/'src/milplus/resources'; resources.mkdir(parents=True)
    shutil.copy2(root/'src/milplus/resources/sources.lock.json', resources/'sources.lock.json')
    spec = importlib.util.spec_from_file_location('isolated_milplus_catalog', root/'scripts/build_catalog.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.__file__ = str(target/'scripts/build_catalog.py'); module.main()
    before = {r['id']: r for r in json.loads((root/'src/milplus/resources/profiles.json').read_text())}
    after = {r['id']: r for r in json.loads((resources/'profiles.json').read_text())}
    assert before.keys() == after.keys()
    for key in before:
        for field in ['family', 'recipe', 'tasks', 'required_inputs', 'class_count', 'external_source']:
            assert before[key][field] == after[key][field], (key, field)
