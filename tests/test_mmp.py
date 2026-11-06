"""Independent MMP author trainer, prototype, UOT, leakage and recovery checks."""
import ast
import copy
import math
from functools import partial
from itertools import combinations
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from einops import rearrange, reduce
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, load_unit, train, units
from milplus.mmp import PROFILES, MultimodalPrototyping, author_mmp, mmp_epoch, prototype_sample
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import pinned_text, verified_source


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def fixture(tmp_path, **options):
    data = create_demo(tmp_path/'data', patients=20, survival=True, omics=True)
    config = yaml.safe_load((data/'task.yaml').read_text())
    config['omics']['normalization'] = 'none'
    (data/'task.yaml').write_text(yaml.safe_dump(config))
    bundle = Bundle(data)
    model = MultimodalPrototyping(bundle.dim, bundle.omics, **{
        'hidden': 16, 'prototypes': 3, 'prototype_patches': 10, 'batch_size': 5, 'cluster_iterations': 5, **options})
    model.configure_training({'epochs': 3, 'lr': 1e-4, 'grad_clip': None, 'max_instances': None})
    model.prepare(bundle, 27, 'cpu')
    return model, bundle


def independent_scope(hidden=256):
    import ot
    root, sha = verified_source('mmp')
    legacy_ot = SimpleNamespace(unif=ot.unif, dist=ot.dist, unbalanced=SimpleNamespace(
        sinkhorn_knopp_unbalanced=partial(ot.unbalanced.sinkhorn_knopp_unbalanced, reg_type='entropy')))
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math, 'ot': legacy_ot,
             'einsum': torch.einsum, 'rearrange': rearrange, 'reduce': reduce, 'combinations': combinations,
             'optim': torch.optim, 'device': torch.device('cpu'), 'concordance_index_censored': lambda *a, **kw: (0.5,)}
    for path in ['src/utils/losses.py', 'src/mil_models/components.py', 'src/mil_models/model_multimodal.py',
                 'src/mil_models/PANTHER/networks.py', 'src/mil_models/PANTHER/layers.py',
                 'src/mil_models/OT/otk/sinkhorn.py', 'src/mil_models/OT/otk/layers.py',
                 'src/utils/utils.py', 'src/training/trainer.py']:
        code = pinned_text(root, sha, path).replace('.cuda()', '')
        if hidden != 256:
            code = code.replace('init_per_path_model(omic_sizes)', f'init_per_path_model(omic_sizes, hidden_dim={hidden})')
        nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)  # noqa: S102 -- independent locked Git reference
    return scope


@pytest.mark.parametrize('profile', sorted(PROFILES))
@pytest.mark.parametrize('loss_kind', ['cox_source', 'nll'])
def test_author_full_trainer_six_adamw_updates(tmp_path, profile, loss_kind):
    model, _ = fixture(tmp_path, profile=profile, loss=loss_kind)
    ref = independent_scope(model.hidden)
    cls = ref['coattn_mot'] if '_uot_' in profile else ref['coattn']
    reference = cls(**model.network_args)
    reference.load_state_dict(model.network.state_dict())
    criterion = ref['NLLSurvLoss'](alpha=model.alpha) if loss_kind == 'nll' else ref['CoxLoss']()
    opt_ref = ref['get_optim'](SimpleNamespace(opt='adamW', lr=1e-4, wd=1e-5), model=reference)
    opt = model.make_optimizer(1e-4, 1e-5)
    # Three steps per epoch, one full warmup epoch, then cosine to step nine.
    sched_ref = torch.optim.lr_scheduler.LambdaLR(opt_ref,
        lambda step: step/3 if step < 3 else .5*(1+math.cos(math.pi*(step-3)/6)))
    records_ref, records = [], []
    def record(optim, record_list):
        pending = []
        def before(optim, args, kwargs):
            params = [p for group in optim.param_groups for p in group['params']]
            pending.append([None if p.grad is None else p.grad.detach().clone() for p in params])
        def after(optim, args, kwargs):
            params = [p for group in optim.param_groups for p in group['params']]
            record_list.append((pending.pop(), [p.detach().clone() for p in params]))
        optim.register_step_pre_hook(before)
        optim.register_step_post_hook(after)
    record(opt_ref, records_ref)
    record(opt, records)
    torch.manual_seed(730)
    batches = []
    for i in range(3):
        # One size-one all-censored tail exercises differentiable zero Cox and AdamW.
        count = 3 if i < 2 else 1
        x = torch.randn(count, model.prototypes, 2*model.dim+1 if model.histo_kind == 'PANTHER' else model.dim)
        omics = torch.randn(count, len(model.columns))
        time = torch.arange(count).double()+10*(i+1)
        event = (torch.arange(count)+i) % 2
        y = model.time_label(time) if loss_kind == 'nll' else time
        batches.append({'img': x, 'omics': [omics[:, g] for g in model.group_indices],
                        'label': y[:, None], 'survival_time': time[:, None], 'censorship': (1-event)[:, None],
                        'raw_omics': omics})
    torch.manual_seed(502)
    for _ in range(2):
        ref['train_loop_survival'](reference, batches, opt_ref, sched_ref, criterion, accum_steps=1)
    torch.manual_seed(502)
    for batch in batches*2:
        opt.zero_grad(set_to_none=True)
        logits = model.batch_logits(batch['img'], batch['raw_omics'])
        loss = model.batch_loss(logits, batch['survival_time'], 1-batch['censorship'])
        loss.backward()
        opt.step()
        model.scheduler.step()
    assert len(records) == len(records_ref) == 6
    for actual, expected in zip(records, records_ref):
        for a, b in zip(actual[0]+actual[1], expected[0]+expected[1]):
            if a is None or b is None:
                assert a is b
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert opt.param_groups[0]['lr'] == opt_ref.param_groups[0]['lr']
    assert not any(p.requires_grad for p in model.histology.parameters())


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_source_histology_tokens_match_independent_code(tmp_path, profile):
    model, _ = fixture(tmp_path, profile=profile)
    ref = independent_scope()
    torch.manual_seed(350)
    x = torch.randn(17, model.dim)*.1
    if model.histo_kind == 'PANTHER':
        original = ref['PANTHERBase'](model.dim, p=3, L=1, tau=1, ot_eps=1, out='allcat', load_proto=False)
        original.load_state_dict(model.histology.state_dict())
        flat, _ = original(x[None])
        expected = torch.cat([flat[:, :3, None], flat[:, 3:3+3*model.dim].reshape(1,3,-1),
                              flat[:, 3+3*model.dim:].reshape(1,3,-1)], -1)
    else:
        original = ref['OTKernel'](model.dim, 3, heads=1, eps=1, max_iter=100, log_domain=False)
        original.load_state_dict(model.histology.state_dict())
        expected = original(x[None])
    torch.testing.assert_close(model.tokenize(x), expected, rtol=2e-5, atol=2e-6)


def test_uot_gradient_entropy_and_zero_distance():
    import ot
    transport = author_mmp().OT_Attn()
    torch.manual_seed(765)
    x, y = torch.rand(4, 8, requires_grad=True), torch.rand(3, 8, requires_grad=True)
    flow = transport.OT(x, y)
    assert flow.requires_grad
    cost = torch.cdist(x, y).square()
    cost = (cost/cost.max()).double()
    a, b = cost.new_full((4,), .25), cost.new_full((3,), 1/3)
    kernel = (-cost/.1).exp()
    u, v = torch.ones_like(a), torch.ones_like(b)
    for _ in range(200):
        u = (a/(kernel@v))**(.5/.6)
        v = (b/(kernel.T@u))**(.5/.6)
    expected = u[:, None]*kernel*v[None]
    weights = torch.arange(flow.numel()).reshape_as(flow)
    actual_grad = torch.autograd.grad((flow*weights).sum(), (x,y), retain_graph=True)
    expected_grad = torch.autograd.grad((expected*weights).sum(), (x,y))
    torch.testing.assert_close(flow, expected, rtol=2e-5, atol=2e-6)
    for a, b in zip(actual_grad, expected_grad):
        assert a.abs().sum() > 0
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-5)
    new_default = ot.unbalanced.sinkhorn_knopp_unbalanced(cost.new_full((4,), .25), cost.new_full((3,), 1/3),
                                                        cost, reg=.1, reg_m=.5)
    assert not torch.allclose(flow, new_default, rtol=1e-3, atol=1e-4)
    zeros = torch.zeros(4, 8, requires_grad=True)
    zero_flow = transport.OT(zeros, zeros[:3])
    zero_flow.sum().backward()
    assert torch.isfinite(zero_flow).all() and torch.isfinite(zeros.grad).all()


def test_cox_source_ties_breslow_and_all_censored(tmp_path):
    model, _ = fixture(tmp_path)
    z = torch.tensor([[1.], [2.], [3.]], requires_grad=True)
    source = model.batch_loss(z, [1.,1.,3.], [1,1,1])
    model.loss_kind = 'cox_breslow'
    expected = ((z.flatten().logsumexp(0)-z[:2].flatten()).sum())/3
    repaired = model.batch_loss(z, [1.,1.,3.], [1,1,1])
    torch.testing.assert_close(repaired, expected)
    assert not torch.allclose(source, repaired)
    perm = torch.tensor([1,0,2])
    torch.testing.assert_close(repaired, model.batch_loss(z[perm], [1.,1.,3.], [1,1,1]))
    for kind in ['cox_source', 'cox_breslow']:
        model.loss_kind = kind
        zero = model.batch_loss(z, [1.,2.,3.], [0,0,0])
        assert zero.item() == 0
        assert torch.autograd.grad(zero, z)[0].eq(0).all()


def test_train_only_sampling_normalization_and_bins(tmp_path):
    model, bundle = fixture(tmp_path, loss='nll')
    bundle.omics.normalization = model.normalization = 'train_standard'
    model.prepare(bundle, 27, 'cpu')
    state = copy.deepcopy(model.state_dict())
    sample, ledger = prototype_sample(bundle, 300000, 17)
    assert len(sample) == sum(len(bundle.load(row)[0]) for row in bundle.records('train'))
    assert len(sample) == sum(r['sampled'] for r in ledger)
    assert all(r['patient_id'] in set(bundle.rows[bundle.rows.subset == 'train'].global_patient_id) for r in ledger)
    # Held-out values, outcomes and WSI features cannot alter any fitted preprocessing.
    heldout = bundle.rows[bundle.rows.subset != 'train']
    bundle.omics.values.loc[heldout.global_patient_id.unique()] = 1e8
    bundle.rows.loc[bundle.rows.subset != 'train', 'time'] = 1e10
    for path in bundle.root.joinpath('features').glob('*.h5'):
        if path.stem in set(heldout.slide_id):
            with h5py.File(path, 'r+') as f:
                f['features'][:] = 99999
    model.prepare(bundle, 27, 'cpu')
    for key in ['omics_mean','omics_scale','bin_edges', 'histology.priors.m']:
        torch.testing.assert_close(state[key], model.state_dict()[key], rtol=0, atol=0)


def test_token_cache_content_invalidation_and_outcome_free_inference(tmp_path):
    model, bundle = fixture(tmp_path, cache_patients=2)
    model.eval()
    x, _ = load_unit(bundle, units(bundle,'val')[0], 'cpu')
    first = model.tokenize(x).clone()
    torch.testing.assert_close(first, model.tokenize(x.clone()), rtol=0, atol=0)
    assert not torch.allclose(first, model.tokenize(x+3))
    with torch.no_grad():
        model.centers.add_(2)
    assert not torch.allclose(first, model.tokenize(x))
    assert len(model.token_cache) == 2
    rows = units(bundle,'test')[0]
    pred = model.forward_unit(bundle,rows,'cpu',{})['logits']
    changed = copy.deepcopy(rows)
    for row in changed:
        row['time'], row['event'] = 1e10, 1-row['event']
    torch.testing.assert_close(pred, model.forward_unit(bundle,changed,'cpu',{})['logits'], rtol=0, atol=0)
    model.load_state_dict(model.state_dict())
    assert not model.token_cache


@pytest.mark.parametrize('profile', sorted(PROFILES))
def test_all_branches_gradients_and_patient_tail(tmp_path, profile):
    model, bundle = fixture(tmp_path, profile=profile, batch_size=11, warmup_epochs=0)
    optimizer = model.make_optimizer(1e-4,1e-5)
    rows = units(bundle,'train')
    logs = mmp_epoch(model,bundle,rows,optimizer,'cpu',{'grad_clip':None})
    assert [r['batch_patients'] for r in logs] == [11,1]
    model.train()
    x = torch.cat([model.tokenize(load_unit(bundle,r,'cpu')[0]) for r in rows[:3]])
    molecules = torch.stack([bundle.omics.tensor(r[0]['global_patient_id']) for r in rows[:3]])
    optimizer.zero_grad(set_to_none=True)
    model.batch_loss(model.batch_logits(x,molecules),[1,2,3],[1,1,1]).backward()
    for prefix in ['path_proj_net','sig_networks','coattn','histo_embedding','gene_embedding','classifier']:
        grads = [p.grad for n,p in model.network.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and sum(float(g.abs().sum()) for g in grads) > 0, prefix
        assert all(torch.isfinite(g).all() for g in grads)
    assert model.centers.grad is None


def test_validation_full_cohort_riskset_and_native_network(tmp_path):
    model, bundle = fixture(tmp_path, hidden=256, prototypes=16)
    ref = independent_scope()
    reference = ref['coattn'](**model.network_args).eval()
    reference.load_state_dict(model.network.state_dict())
    model.eval()
    x = model.tokenize(load_unit(bundle,units(bundle,'val')[0],'cpu')[0])
    molecules = torch.randn(1,len(model.columns))
    torch.testing.assert_close(model.batch_logits(x,molecules),
        reference.forward_no_loss(x,[molecules[:,g] for g in model.group_indices])['logits'],rtol=0,atol=0)
    frame = pd.DataFrame({'risk':[.2,.5,.7], 'time':[1,2,3], 'event':[1,0,1]})
    assert model.validation_metrics(frame)['survival_loss'] == pytest.approx(float(model.batch_loss(
        torch.tensor([[.2],[.5],[.7]]),frame.time,frame.event)))
    assert model.validation_metrics(frame)['survival_loss'] > 0  # per-patient Cox would all be zero


@pytest.mark.parametrize('profile', sorted(PROFILES))
@pytest.mark.parametrize('loss_kind', ['cox_source','nll'])
def test_resume_reload_and_scheduler_state(tmp_path, profile, loss_kind):
    _,bundle = fixture(tmp_path)
    config = {'epochs':3,'threads':1,'model':{'hidden':16,'prototypes':3,'prototype_patches':10,
              'cluster_iterations':5,'batch_size':5,'loss':loss_kind}}
    full = train(bundle.root,profile,tmp_path/'full',seed=42,config=config)
    part = train(bundle.root,profile,tmp_path/'resume',seed=42,config=config,stop_after=1)
    assert not (part/'test_predictions.csv').exists()
    train(bundle.root,profile,part,seed=42,config=config,resume=True)
    a,b = [torch.load(r/'last.pt',weights_only=False) for r in [full,part]]
    assert a['model']['_extra_state'] == b['model']['_extra_state']
    for key in a['model']:
        if key != '_extra_state':
            torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    torch.testing.assert_close(a['optimizers'],b['optimizers'],rtol=0,atol=0)
    assert a['history'] == b['history']
    assert evaluate(full) == evaluate(part)
    first = (full/'test_predictions.csv').read_bytes()
    evaluate(full)
    assert first == (full/'test_predictions.csv').read_bytes()


@pytest.mark.parametrize('options', [{'prototypes':0},{'hidden':1},{'loss':'ranking'}, {'append_embed':'bad'},
    {'histo_agg':'bad'},{'eps':float('nan')},{'cache_patients':-1},{'batch_size':True}])
def test_contract_rejects_invalid_options(tmp_path, options):
    with pytest.raises(ValueError):
        fixture(tmp_path, **options)


def test_missing_omics_and_generic_sampling_rejected(tmp_path):
    model,bundle = fixture(tmp_path)
    with pytest.raises(ValueError,match='max_instances'):
        model.configure_training({'max_instances':3})
    bundle.omics = None
    with pytest.raises(ValueError,match='omics'):
        check_compatibility(require_runnable('MMP'),bundle)


@pytest.mark.parametrize('embedding', ['none','modality','proto','random'])
def test_embedding_shared_ffn_and_concatenation_options(tmp_path, embedding):
    model, bundle = fixture(tmp_path, append_embed=embedding, histo_agg='cat', net_indiv=False, loss='nll')
    model.eval()
    output = model.forward_unit(bundle, units(bundle,'val')[0], 'cpu', {})
    assert output['hazards'].shape == (1,4)
    assert torch.isfinite(output['hazards']).all()
    assert (output['survival'][:,1:] <= output['survival'][:,:-1]).all()


def test_covariance_floor_prevents_float32_cancellation(tmp_path):
    model, _ = fixture(tmp_path)
    with torch.no_grad():
        model.centers.fill_(10000)
    x = torch.full((1,10,model.dim),10000.)
    weights = torch.full((1,10,model.prototypes),1/model.prototypes)
    _,_,cov = model.histology.priors.map_m_step(x, weights, tau=1)
    assert cov.ge(model.numerics['variance_floor']).all()
    ref = independent_scope()['DirNIWNet'](model.prototypes,model.dim,eps=1,load_proto=False)
    ref.load_state_dict(model.histology.priors.state_dict())
    assert ref.map_m_step(x,weights,tau=1)[2].eq(0).any()


def test_low_lr_constant_schedule_and_invalid_warmup(tmp_path):
    model,_ = fixture(tmp_path)
    model.configure_training({'epochs':3,'lr':1e-5,'max_instances':None,'grad_clip':None})
    optimizer = model.make_optimizer(1e-5,1e-5)
    for _ in range(9):
        optimizer.step()
        model.scheduler.step()
        assert optimizer.param_groups[0]['lr'] == 1e-5
    with pytest.raises(ValueError,match='after warmup'):
        model.configure_training({'epochs':1,'lr':1e-4,'max_instances':None,'grad_clip':None})


def test_equal_slide_sampling_is_reproducible_and_does_not_mutate_features():
    matrices = [torch.arange(36.).reshape(12,3), torch.arange(36.,72.).reshape(12,3)]
    original = [x.clone() for x in matrices]
    bundle = SimpleNamespace(records=lambda split: [{'slide_id':str(i),'global_patient_id':str(i)} for i in range(2)],
                             load=lambda row: (matrices[int(row['slide_id'])],None))
    actual,ledger = prototype_sample(bundle,10,29)
    rng = np.random.RandomState(29)
    expected = np.concatenate([x[rng.permutation(12)[:5]].numpy() for x in matrices])
    np.testing.assert_array_equal(actual,expected)
    assert [r['sampled'] for r in ledger] == [5,5]
    for a,b in zip(matrices,original):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
