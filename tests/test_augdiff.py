"""Source fidelity, complete native DAE and seven-view leakage boundaries.

These checks do not admit an unfinished MIL trainer or claim paper reproduction.
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from milplus.augdiff_data import INDEPENDENT, KIND, PAIRED, VIEWS, AugDiffInputs
from milplus.augdiff_source import (
    LAYOUT,
    augment,
    build_classifier,
    build_diffusion,
    classifier_optimizer,
    classifier_result,
    definitions,
    epsilon_loss,
)
from milplus.io import digest_file, digest_object, write_json


@pytest.mark.parametrize('head', ['amil', 'lossattn', 'dsmil'])
@pytest.mark.parametrize('dimension', [384, 512])
def test_augdiff_exact_original_head_logits_loss_and_all_gradients(head, dimension):
    filename = {'amil': 'AMIL', 'lossattn': 'LossAttn', 'dsmil': 'DSMIL'}[head]
    # Original source directly reads [1,N,D+2]. Its two leading coordinates
    # are provably unused, so arbitrary test values must not affect predictions.
    original = definitions('augdiff', 'MIL/models/'+filename+'.py',
        replacements=[('.cuda()', '')] if head == 'dsmil' else [])
    reference = getattr(original, filename)(feats_size=dimension, n_classes=3)
    adapted = build_classifier(head, dimension, 3)
    adapted.load_state_dict(reference.state_dict(), strict=True)
    torch.manual_seed(67)
    a = torch.randn(5, dimension, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    unconsumed = torch.randn(5, 2)
    direct = reference(data=torch.cat((unconsumed, a), dim=1)[None])
    actual = classifier_result(adapted, b, torch.tensor([2]))
    assert torch.equal(actual['logits'], direct['logits'])
    expected_loss = torch.nn.functional.cross_entropy(direct['logits'], torch.tensor([2]))
    assert torch.equal(actual['loss'], expected_loss)
    actual['loss'].backward(); expected_loss.backward()
    assert torch.equal(a.grad, b.grad)
    for (_, expected), (_, result) in zip(reference.named_parameters(), adapted.named_parameters()):
        if expected.grad is None:
            assert result.grad is None
        else:
            assert torch.equal(expected.grad, result.grad)


@pytest.mark.parametrize('head', ['amil', 'lossattn', 'dsmil'])
def test_augdiff_single_patch_rank_port_retains_complete_head(head):
    model = build_classifier(head, 512, 3)
    result = classifier_result(model, torch.randn(1, 512), torch.tensor([1]))
    assert result['logits'].shape == (1, 3)
    result['loss'].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_augdiff_original_radam_lookahead_numerics_and_portable_resume():
    torch.manual_seed(59)
    a = torch.nn.Linear(4, 3); b = copy.deepcopy(a)
    radam = definitions('augdiff', 'MIL/MyOptimizer/radam.py', ['RAdam'],
        {'Optimizer': torch.optim.Optimizer}).RAdam
    from collections import defaultdict
    original = definitions('augdiff', 'MIL/MyOptimizer/lookahead.py', ['Lookahead'],
        {'Optimizer': torch.optim.Optimizer, 'defaultdict': defaultdict}).Lookahead
    group = definitions('augdiff', 'MIL/MyOptimizer/optim_factory.py', ['add_weight_decay']).add_weight_decay
    expected = original(radam(group(a, 1e-5), lr=2e-4, weight_decay=0.))
    # Original wrapper predates modern Optimizer hooks. Supply only framework
    # initialization; the numerical step/update_slow definitions remain exact.
    torch.optim.Optimizer.__init__(expected, expected.base_optimizer.param_groups, expected.defaults)
    expected.param_groups = expected.base_optimizer.param_groups
    actual = classifier_optimizer(b)
    # Cross first slow-buffer creation at6, second update12 and RAdam's ring.
    for index in range(17):
        x = torch.randn(7, 4); y = torch.randn(7, 3)
        for model, optimizer in [(a, expected), (b, actual)]:
            optimizer.zero_grad()
            (model(x)-y).square().mean().backward(); optimizer.step()
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
        if index == 7:
            state = copy.deepcopy(actual.state_dict())
            b = copy.deepcopy(b); actual = classifier_optimizer(b); actual.load_state_dict(state)
    assert [g['lookahead_step'] for g in actual.param_groups] == [17, 17]
    assert sum('slow_buffer' in value for value in actual.state.values()) == 2


@pytest.fixture(scope='module')
def native_diffusion():
    model = build_diffusion(512)
    model.train()
    optimizer = model.configure_optimizers()
    loss, _ = epsilon_loss(model, torch.randn(2, 512))
    loss.backward(); optimizer.step(); model.on_train_batch_end(); optimizer.zero_grad(set_to_none=True)
    del optimizer
    yield model
    del model


def test_augdiff_full_published_width_epsilon_objective_and_real_update(native_diffusion):
    model = native_diffusion; model.train(); model.requires_grad_(True)
    unet = model.model.diffusion_model
    assert unet.model_channels == 256
    assert tuple(unet.channel_mult) == (1, 2, 4)
    assert unet.num_res_blocks == 2
    assert len(unet.input_blocks) == 9 and len(unet.output_blocks) == 9
    assert sum(p.numel() for p in model.parameters()) == 122055680
    assert model.num_timesteps == 1000 and model.feature_layout == LAYOUT
    assert model.cond_stage_model is None
    assert model.logvar.shape == (1000,) and 'logvar' in dict(model.named_buffers())
    torch.manual_seed(73)
    x = torch.randn(2, 512); noise = torch.randn_like(x); t = torch.tensor([0, 999])
    alpha = model.alphas_cumprod[t, None, None]
    noisy = alpha.sqrt()*x[..., None] + (1-alpha).sqrt()*noise[..., None]
    direct = model.model(noisy, t)
    expected = (direct-noise[..., None]).square().mean(dim=(1, 2)).mean()
    loss, report = epsilon_loss(model, x, t, noise)
    assert torch.equal(loss, expected)
    assert torch.equal(report['train/loss_simple'], expected)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert all(g['weight_decay'] == .01 for g in optimizer.param_groups)
    weight = unet.out[-1].weight
    before = weight.detach().clone()
    name = next(name for name, value in model.model.named_parameters() if value is weight)
    shadow = getattr(model.model_ema, model.model_ema.m_name2s_name[name])
    shadow_before = shadow.clone()
    previous_updates = int(model.model_ema.num_updates)
    loss.backward(); optimizer.step(); model.on_train_batch_end(); optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(before, weight)
    # Actual EMA warmup decay is(1+n)/(10+n), not constant.9999.
    n = previous_updates+1
    expected_ema = shadow_before-(1-(1+n)/(10+n))*(shadow_before-weight.detach())
    torch.testing.assert_close(shadow, expected_ema, atol=1e-10, rtol=1e-6)
    assert int(model.model_ema.num_updates) == previous_updates+1


def test_augdiff_trained_native_ddim_uses_original_semantic_features(native_diffusion):
    model = native_diffusion; model.eval(); model.requires_grad_(False)
    assert int(model.model_ema.num_updates) >= 1
    x = torch.randn(2, 512)
    visited = []
    hook = model.model.diffusion_model.register_forward_pre_hook(
        lambda module, values: visited.append(int(values[1][0])))
    try:
        torch.manual_seed(89)
        actual = augment(model, x)
    finally:
        hook.remove()
    assert visited == [351, 301, 251, 201, 151, 101, 51, 1]
    assert actual.shape == x.shape and torch.isfinite(actual).all() and not torch.equal(actual, x)
    torch.manual_seed(89)
    assert torch.equal(augment(model, x), actual)


@pytest.mark.parametrize('kwargs', [{'layout': 'literal_image_yaml'}, {'dimension': 0}, {'dimension': True}])
def test_augdiff_does_not_silently_invent_a_smaller_native_network(kwargs):
    with pytest.raises(ValueError):
        build_diffusion(**{'dimension': 512, **kwargs})


def fixture_assets(root, dimension=512):
    """Only synthetic software fixture, never claimed image/FM extraction."""
    root = Path(root); root.mkdir()
    rows, records = [], []
    for i, subset in enumerate(['train', 'train', 'val', 'test']):
        path = root/f'original{i}.npy'
        value = np.random.default_rng(i).normal(size=(2, dimension)).astype(np.float32)
        np.save(path, value)
        row = {'slide_id': f'slide{i}', 'global_patient_id': f'patient{i}',
               'path': str(path), 'subset': subset}
        rows.append(row)
        if subset != 'train':
            continue
        views = {}; patch_hash = digest_object([f'patch{i}_0', f'patch{i}_1'])
        for j, name in enumerate(VIEWS):
            view_path = path if j == 0 else root/f'view{i}_{j}.npy'
            if j:
                np.save(view_path, value + j*.01)  # explicitly synthetic tensor fixture
            views[name] = {'path': str(view_path), 'sha256': digest_file(view_path),
                           'ordered_patch_ids_sha256': patch_hash}
        records.append({'slide_id': row['slide_id'], 'diffusion_subset': 'train' if i == 0 else 'val',
                        'ordered_patch_ids_sha256': patch_hash, 'views': views})
    bundle = SimpleNamespace(root=root, task_type='classification', config={'synthetic': True},
        encoder={'model': 'synthetic_software_only'}, fm='toy', dim=dimension)
    bundle.records = lambda subset: [r for r in rows if r['subset'] == subset]
    bundle.load = lambda row: (torch.from_numpy(np.load(row['path'], allow_pickle=False)), None)
    metadata = {'schema_version': 1, 'kind': KIND, 'synthetic': True, 'dimension': dimension,
        'view_pairing_policy': PAIRED,
        'fm': bundle.fm, 'encoder_identity': digest_object(bundle.encoder), 'records': records,
        **{key: 'synthetic software fixture; no real WSI/image augmentation/FM claim' for key in
           ['augmentation_provenance', 'weights_provenance', 'feature_preprocessing',
            'row_correspondence_provenance', 'diffusion_split_provenance']}}
    path = root/'views.json'; write_json(path, metadata)
    return bundle, path, metadata


def test_augdiff_seven_actual_views_are_all_consumed(tmp_path):
    bundle, path, _ = fixture_assets(tmp_path/'views')
    assets = AugDiffInputs(bundle, path)
    for subset in ['train', 'val']:
        dataset = assets.patches(subset)
        assert len(dataset) == 14
        values = torch.stack([dataset[i] for i in range(len(dataset))])
        assert values.shape == (14, 512)
        assert not torch.equal(values[0], values[2])
        original = next(v for v in assets.records.values() if v['diffusion_subset'] == subset)
        assert torch.equal(values[0], torch.from_numpy(np.load(original['views']['images']['path']))[0])


@pytest.mark.parametrize('bad', ['missing_view', 'heldout', 'row_hash', 'view_bytes', 'same_patient', 'wrong_fm'])
def test_augdiff_rejects_missing_views_leakage_and_unaligned_assets(tmp_path, bad):
    bundle, path, metadata = fixture_assets(tmp_path/'views')
    if bad == 'missing_view':
        metadata['records'][0]['views'].pop(VIEWS[-1])
    elif bad == 'heldout':
        metadata['records'][0]['slide_id'] = 'slide2'
    elif bad == 'row_hash':
        metadata['records'][0]['views'][VIEWS[-1]]['ordered_patch_ids_sha256'] = 'a'*64
    elif bad == 'view_bytes':
        metadata['records'][0]['views'][VIEWS[-1]]['sha256'] = 'a'*64
    elif bad == 'same_patient':
        rows = bundle.records('train')
        for row in rows:
            row['global_patient_id'] = 'patient0'
        original_records = bundle.records
        bundle.records = lambda subset: rows if subset == 'train' else original_records(subset)
    elif bad == 'wrong_fm':
        metadata['encoder_identity'] = 'a'*64
    write_json(path, metadata)
    with pytest.raises(ValueError):
        AugDiffInputs(bundle, path)


def test_augdiff_mutated_view_rejected_during_patch_loading(tmp_path):
    bundle, path, metadata = fixture_assets(tmp_path/'views')
    assets = AugDiffInputs(bundle, path); dataset = assets.patches('train')
    selected = Path(metadata['records'][0]['views']['images']['path'])
    values = np.load(selected); values[0, 0] += 10; np.save(selected, values)
    with pytest.raises(ValueError, match='changed'):
        dataset[0]


@pytest.mark.parametrize('patch_ids', [True, False])
def test_augdiff_independent_view_pools_keep_all_rows_without_pairing(tmp_path, patch_ids):
    bundle, path, metadata = fixture_assets(tmp_path/'views')
    # The released unconditional DAE concatenates independent style pools;
    # its epsilon objective never compares corresponding patch rows.
    metadata.pop('view_pairing_policy')  # faithful independent source default
    metadata.pop('row_correspondence_provenance')
    metadata['view_identity_provenance'] = 'Synthetic independently extracted style pools, not real image/FM data'
    record = metadata['records'][0]
    entry = record['views'][VIEWS[1]]
    value = np.arange(5*bundle.dim, dtype=np.float32).reshape(5, bundle.dim)
    np.save(entry['path'], value)
    entry['sha256'] = digest_file(entry['path'])
    entry['ordered_patch_ids_sha256'] = digest_object(['independent', 5])
    if not patch_ids:
        for item in metadata['records']:
            item.pop('ordered_patch_ids_sha256')
            for view in item['views'].values():
                view.pop('ordered_patch_ids_sha256')
    write_json(path, metadata)
    assets = AugDiffInputs(bundle, path)
    assert assets.pairing == INDEPENDENT
    dataset = assets.patches('train')
    assert len(dataset) == 17  # 5 independent rows plus six original 2-row pools
    actual = torch.stack([dataset[i] for i in range(len(dataset))])
    assert torch.equal(actual[2:7], torch.from_numpy(value))
    assert len(assets.patches('val')) == 14
    # Choosing paired semantics is a real additional constraint, not a silent
    # truncation or invented correspondence to make independent pools fit.
    metadata['view_pairing_policy'] = PAIRED
    metadata['row_correspondence_provenance'] = 'Synthetic pairing audit'
    write_json(path, metadata)
    with pytest.raises(ValueError):
        AugDiffInputs(bundle, path)


def test_augdiff_native_attention_checkpoint_preserves_autocast_and_all_gradients():
    from milplus.augdiff_source import classes
    torch.set_num_threads(1)
    # Full1024-channel native level, nontrivial token relation and an explicit
    # synthetic warm projection so zero initialization cannot hide gradients.
    actual = classes().network.AttentionBlock(1024, num_head_channels=32)
    with torch.no_grad():
        actual.proj_out.weight.normal_(std=.02)
    reference = copy.deepcopy(actual)
    x = torch.randn(2, 1024, 4, dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = actual(x)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        # Same original full _forward, direct-autograd memory reference only;
        # the production model retains actual activation checkpointing.
        expected = reference._forward(y)
    assert torch.equal(output, expected)
    direction = torch.randn_like(output).float()
    (output.float()*direction).sum().backward()
    (expected.float()*direction).sum().backward()
    assert torch.equal(x.grad, y.grad)
    for (_, a), (_, b) in zip(actual.named_parameters(), reference.named_parameters()):
        assert a.grad is not None and b.grad is not None and torch.equal(a.grad, b.grad)
