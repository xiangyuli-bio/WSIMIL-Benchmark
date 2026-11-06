"""Patient alignment, censoring, train-only fitting, genuine modality gradients and checkpoint replay."""
import ast
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.discrete_survival import DiscreteSurvival, discrete_epoch, discrete_nll
from milplus.engine import evaluate, train, units
from milplus.graph_survival import patient_graph, spatial_edges
from milplus.io import read_json
from milplus.models import build_bundle_model
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import pinned_text, source_root, verified_source

METHODS = ['survpath', 'mcat', 'porpoise', 'patchgcn']


def needs_source(method):
    if not source_root(method).is_dir():
        pytest.skip(f'Prepare pinned author checkout with milplus setup --method {method}')
    if method == 'patchgcn':
        pytest.importorskip('torch_geometric')


def config(method, epochs=2):
    options = {'bins': 3, 'accumulation_steps': 5}
    if method == 'patchgcn':
        options['hidden'] = 16
    return {'epochs': epochs, 'threads': 1, 'model': options}


def test_discrete_likelihood_by_hand_and_extreme_logits():
    logits = torch.zeros(3, 4, dtype=torch.float64, requires_grad=True)
    events = torch.tensor([1., 1., 0.])
    labels = torch.tensor([0, 2, 1])
    loss = discrete_nll(logits, labels, events)
    torch.testing.assert_close(loss, torch.tensor(2*math.log(2), dtype=torch.float64))
    loss.backward()
    # Event hazard is encouraged, preceding and censored intervals are discouraged.
    torch.testing.assert_close(logits.grad, torch.tensor([[-1, 0, 0, 0], [1, 1, -1, 0],
                                                         [1, 1, 0, 0]], dtype=torch.float64)/6)
    extreme = torch.tensor([[-1000., 1000., -1000., 1000.]], requires_grad=True)
    loss = discrete_nll(extreme, torch.tensor([3]), torch.tensor([1]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()
    with pytest.raises(ValueError, match='integers'):
        discrete_nll(logits, torch.tensor([0., 1.5, 2.]), events)


@pytest.mark.parametrize('alpha', [0., .15, .5])
def test_nll_matches_pinned_author_value_and_gradient(alpha):
    needs_source('survpath')
    root, sha = verified_source('survpath')
    tree = ast.parse(pinned_text(root, sha, 'utils/loss_func.py'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'nll_loss')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned SurvPath nll_loss>', 'exec'), scope)  # noqa: S102 -- single definition from verified author commit
    torch.manual_seed(5)
    logits = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    labels, events = torch.arange(8) % 4, torch.arange(8) % 2
    ours = discrete_nll(logits, labels, events, alpha)
    reference = scope['nll_loss'](logits, labels[:, None], (1-events)[:, None], alpha=alpha, reduction='mean')
    torch.testing.assert_close(ours, reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(ours, logits, retain_graph=True)[0],
                               torch.autograd.grad(reference, logits)[0], rtol=1e-12, atol=1e-12)


def test_bins_and_omics_fit_unique_training_patients_only(tmp_path):
    root = create_demo(tmp_path/'data', patients=20, survival=True, omics=True)
    bundle = Bundle(root)
    model = DiscreteSurvival(bins=3)
    model.prepare(bundle, 12, 'cpu')
    mean, scale = bundle.omics.training_statistics(bundle)
    train_rows = bundle.rows[bundle.rows.subset == 'train'].drop_duplicates('global_patient_id')
    matrix = bundle.omics.values.loc[sorted(train_rows.global_patient_id)].to_numpy()
    np.testing.assert_allclose(mean.numpy(), matrix.mean(0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(scale.numpy(), matrix.std(0), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(model.time_label(model.bin_edges), torch.tensor([1, 2]))
    torch.testing.assert_close(model.time_label([0, 1e20]), torch.tensor([0, 2]))
    # Mutating held-out outcomes and omics must change the bundle hash, but no fitted training quantities.
    held_out = set(bundle.rows.loc[bundle.rows.subset != 'train', 'global_patient_id'])
    molecular = pd.read_csv(root/'omics.csv')
    molecular.loc[molecular.global_patient_id.isin(held_out), bundle.omics.columns] = 1e8
    molecular.to_csv(root/'omics.csv', index=False)
    slides = pd.read_csv(root/'slides.csv')
    slides.loc[slides.patient_id.isin(held_out), 'time'] = 1e10
    slides.to_csv(root/'slides.csv', index=False)
    changed = Bundle(root)
    other = DiscreteSurvival(bins=3)
    other.prepare(changed, 12, 'cpu')
    assert changed.fingerprint != bundle.fingerprint
    torch.testing.assert_close(model.bin_edges, other.bin_edges, rtol=0, atol=0)
    m2, s2 = changed.omics.training_statistics(changed)
    torch.testing.assert_close(mean, m2, rtol=0, atol=0)
    torch.testing.assert_close(scale, s2, rtol=0, atol=0)


@pytest.mark.parametrize('mutation,match', [('duplicate','unique'),('missing','patient sets'),
                                          ('nan','finite'),('unknown_gene','present')])
def test_omics_rejects_ambiguous_or_incomplete_inputs(tmp_path, mutation, match):
    root = create_demo(tmp_path/'data', patients=20, survival=True, omics=True)
    frame = pd.read_csv(root/'omics.csv')
    if mutation == 'duplicate':
        frame = pd.concat([frame, frame.iloc[:1]])
    elif mutation == 'missing':
        frame = frame.iloc[1:]
    elif mutation == 'nan':
        frame.loc[0, 'gene_0'] = float('inf')
    else:
        pathways = json.loads((root/'pathways.json').read_text())
        pathways[0]['features'].append('not_a_measured_gene')
        (root/'pathways.json').write_text(json.dumps(pathways))
    frame.to_csv(root/'omics.csv', index=False)
    with pytest.raises(ValueError, match=match):
        Bundle(root)


def test_omics_row_order_and_global_alias_alignment(tmp_path):
    root = create_demo(tmp_path/'data', patients=20, survival=True, omics=True)
    original = Bundle(root)
    slides, splits = pd.read_csv(root/'slides.csv'), pd.read_csv(root/'split.csv')
    slides['global_patient_id'] = slides.patient_id
    slides.patient_id = 'local_' + slides.patient_id
    splits.patient_id = 'local_' + splits.patient_id
    slides.to_csv(root/'slides.csv', index=False)
    splits.to_csv(root/'split.csv', index=False)
    molecular = pd.read_csv(root/'omics.csv').sample(frac=1, random_state=7)
    molecular.to_csv(root/'omics.csv', index=False)
    aliased = Bundle(root)
    for patient in original.rows.global_patient_id:
        torch.testing.assert_close(original.omics.tensor(patient), aliased.omics.tensor(patient))
    assert len(units(aliased, 'train')) == len(units(original, 'train'))


def test_incompatible_molecular_method_has_explicit_matrix_blocker(tmp_path):
    root = create_demo(tmp_path/'data', patients=20, survival=True)
    with pytest.raises(ValueError, match='omics'):
        check_compatibility(require_runnable('survpath'), Bundle(root))


def test_graph_has_no_cross_slide_edges_and_respects_sampling_budget(tmp_path):
    pytest.importorskip('torch_geometric')
    root = create_demo(tmp_path/'data', patients=20, survival=True)
    bundle = Bundle(root)
    group = next(rows for rows in units(bundle, 'train') if len(rows) == 2)
    graph = patient_graph(bundle, group, 'cpu', neighbors=3, max_instances=15)
    assert graph.num_nodes == 15
    assert len(graph.slide_index.unique()) == 2
    source, target = graph.edge_index
    assert (source != target).all()
    assert (graph.slide_index[source] == graph.slide_index[target]).all()
    edges = spatial_edges(torch.tensor([[0., 0.], [1., 0.], [10., 0.]]), 1)
    assert edges.tolist() == [[1, 0, 1], [0, 1, 2]]
    assert spatial_edges(torch.zeros(1, 2)).shape == (2, 0)
    with pytest.raises(ValueError, match='Duplicate'):
        spatial_edges(torch.zeros(2, 2))


@pytest.mark.parametrize('method', METHODS)
def test_survival_train_reload_and_exact_resume(tmp_path, method):
    needs_source(method)
    root = create_demo(tmp_path/'data', patients=20, survival=True, omics=True)
    cfg = config(method)
    full = train(root, method, tmp_path/'full', config=cfg)
    split = train(root, method, tmp_path/'split', config=cfg, stop_after=1)
    train(root, method, split, config=cfg, resume=True)
    left, right = [torch.load(p/'last.pt', weights_only=False) for p in (full, split)]
    assert left['history'] == right['history']
    for key in left['model']:
        torch.testing.assert_close(left['model'][key], right['model'][key], rtol=0, atol=0)
    assert not (full/'test_predictions.csv').exists()
    evaluate(full)
    first = pd.read_csv(full/'test_predictions.csv')
    evaluate(full)
    pd.testing.assert_frame_equal(first, pd.read_csv(full/'test_predictions.csv'))
    assert len(first) == Bundle(root).rows.query("subset == 'test'").global_patient_id.nunique()
    assert all(f'hazards_{i}' in first for i in range(3))
    np.testing.assert_allclose(first.risk, -first[[f'survival_{i}' for i in range(3)]].sum(1))
    assert read_json(full/'survival_protocol.json')['event_definition'] == '1=observed; 0=censored'
    # Changing any molecular file invalidates existing results, even when feature files are unchanged.
    molecular = pd.read_csv(root/'omics.csv')
    molecular.loc[0, 'gene_0'] += 1
    molecular.to_csv(root/'omics.csv', index=False)
    with pytest.raises(ValueError, match='inputs differ'):
        evaluate(full)


@pytest.mark.parametrize('method', ['survpath', 'mcat', 'porpoise'])
def test_both_modalities_have_gradient_and_change_prediction(tmp_path, method):
    needs_source(method)
    torch.set_num_threads(1)
    root = create_demo(tmp_path/'data', patients=20, survival=True, omics=True)
    bundle = Bundle(root)
    model = build_bundle_model(method, bundle, {'bins': 3})
    model.prepare(bundle, 1, 'cpu')
    model.eval()
    x, molecular = torch.randn(13, 16, requires_grad=True), torch.randn(18, requires_grad=True)
    logits = model(x, molecular)['hazard_logits']
    logits.square().sum().backward()
    assert x.grad.abs().sum() > 0 and molecular.grad.abs().sum() > 0
    assert not torch.allclose(logits, model(x+3, molecular)['hazard_logits'])
    assert not torch.allclose(logits, model(x, molecular+3)['hazard_logits'])


def test_survival_preparation_rejects_insufficient_events(tmp_path):
    root = create_demo(tmp_path/'data', patients=20, survival=True)
    bundle = Bundle(root)
    with pytest.raises(ValueError, match='distinct observed'):
        DiscreteSurvival(bins=30).prepare(bundle, 0, 'cpu')


def test_mcat_requires_six_explicit_signatures(tmp_path):
    root = create_demo(tmp_path/'data', patients=20, survival=True, omics=True)
    pathways = json.loads((root/'pathways.json').read_text())
    (root/'pathways.json').write_text(json.dumps(pathways[:2]))
    with pytest.raises(ValueError, match='six'):
        check_compatibility(require_runnable('mcat'), Bundle(root))


def test_pathway_definition_changes_bundle_hash(tmp_path):
    root = create_demo(tmp_path/'data', patients=20, survival=True, omics=True)
    original = Bundle(root)
    pathways = json.loads((root/'pathways.json').read_text())
    pathways[0]['features'].reverse()
    (root/'pathways.json').write_text(json.dumps(pathways))
    assert original.fingerprint != Bundle(root).fingerprint
    config = yaml.safe_load((root/'task.yaml').read_text())
    config['omics']['columns'] = ['gene_0', 'gene_0']
    (root/'task.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='unique'):
        Bundle(root)


def test_final_accumulation_window_uses_actual_patient_count():
    class SmallSurvival(DiscreteSurvival):
        def __init__(self):
            super().__init__(bins=2, accumulation_steps=2)
            self.head = torch.nn.Linear(1, 2, bias=False)
            self.prepared.fill_(True)
            self.bin_edges.fill_(2)

        def forward_unit(self, bundle, rows, device, config, train=False):
            return self.result(self.head(torch.tensor([[rows[0]['x']]])))

    torch.manual_seed(8)
    model, reference = SmallSurvival(), SmallSurvival()
    reference.load_state_dict(model.state_dict())
    groups = [[{'x': x, 'time': time, 'event': event}] for x, time, event in [(1., 1., 1), (2., 3., 0), (4., 2., 1)]]
    ours, expected = torch.optim.SGD(model.parameters(), lr=.1), torch.optim.SGD(reference.parameters(), lr=.1)
    discrete_epoch(model, None, groups, ours, 'cpu', {'grad_clip': 100})
    for window in (groups[:2], groups[2:]):
        expected.zero_grad()
        logits = reference.head(torch.tensor([[g[0]['x']] for g in window]))
        loss = discrete_nll(logits, reference.time_label([g[0]['time'] for g in window]),
                             torch.tensor([g[0]['event'] for g in window]))
        loss.backward()
        expected.step()
    torch.testing.assert_close(model.head.weight, reference.head.weight, rtol=0, atol=1e-7)


def test_patchgcn_message_passing_receives_gradients(tmp_path):
    needs_source('patchgcn')
    root = create_demo(tmp_path/'data', patients=20, survival=True)
    bundle = Bundle(root)
    model = build_bundle_model('patchgcn', bundle, {'hidden': 16, 'bins': 3})
    model.prepare(bundle, 2, 'cpu')
    graph = patient_graph(bundle, units(bundle, 'train')[0], 'cpu')
    model(graph)['hazard_logits'].square().sum().backward()
    for layer in model.network.layers:
        gradients = [p.grad for p in layer.conv.parameters() if p.grad is not None]
        assert gradients and sum(g.abs().sum() for g in gradients) > 0


@pytest.mark.parametrize('method,fusion', [('mcat', 'bilinear'), ('porpoise', 'concat')])
def test_alternative_fusion_with_another_fm_dimension(tmp_path, method, fusion):
    needs_source(method)
    root = create_demo(tmp_path/'data', patients=20, dimension=32, survival=True, omics=True)
    bundle = Bundle(root)
    model = build_bundle_model(method, bundle, {'bins': 3, 'fusion': fusion})
    model.prepare(bundle, 1, 'cpu')
    logits = model(torch.randn(7, 32), torch.randn(18))['hazard_logits']
    assert logits.shape == (1, 3)
    logits.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.network.mm.parameters())
