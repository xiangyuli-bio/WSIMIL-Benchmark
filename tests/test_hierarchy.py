"""HMIL author-step equivalence, explicit source defects and two-level data invariants."""
import ast
import logging
import math
from collections import defaultdict
from types import MethodType, SimpleNamespace
from typing import Any, Optional

import pandas as pd
import pytest
import torch
import yaml
from sklearn.metrics import classification_report, roc_auc_score
from torch import nn
from torch.nn import functional as F

from milplus.data import Bundle
from milplus.demo import create_demo, create_hierarchy_demo
from milplus.engine import evaluate, train
from milplus.hierarchy import HierarchyLabels
from milplus.hmil import HierarchicalMIL, hmil_epoch
from milplus.io import read_json
from milplus.metrics import classification_metrics
from milplus.models import build_bundle_model
from milplus.registry import check_compatibility, require_runnable
from milplus.upstream import load_definitions, pinned_text, verified_source


def labels(classes=3):
    return HierarchyLabels(SimpleNamespace(task_type='classification', classes=[str(i) for i in range(classes)],
        config={'hierarchy': {'coarse_classes': ['A', 'B'],
                'fine_to_coarse': {str(i): ('A' if i < classes-1 else 'B') for i in range(classes)},
                'label_provenance': 'synthetic only'}}, rows=pd.DataFrame()))


def original_model():
    return load_definitions('hmil', 'src/models.py', ['HMIL', 'GatedAttention', 'initialize_weights'],
        {'torch': torch, 'nn': nn, 'F': F, 'List': list, 'Tuple': tuple}).HMIL


def reference_network(actual, dropout):
    """Original source class, scalar layers restored independently at runtime."""
    net = original_model()([2, actual.classes], dropout)
    net.coarse_classifier = nn.Linear(1024, 1)
    net.fine_classifier = nn.Linear(256, 1)
    net._compute_logits = MethodType(lambda self, a, b:
        [self.coarse_classifier(a).T, self.fine_classifier(b).T], net)
    net.load_state_dict(actual.network.state_dict())
    return net


def reference_epoch():
    supcon = load_definitions('hmil', 'src/loss.py', ['SupConLoss'],
        {'torch': torch, 'nn': nn, 'Optional': Optional}).SupConLoss
    # Device-only changes; fix metric collector's vector/scalar shapes so the
    # unmodified source objective/backprop/Adam steps can run to completion.
    return load_definitions('hmil', 'src/trainer.py',
        ['TrainingMetrics', 'compute_hierarchical_loss', 'compute_attention_matching_loss',
         'compute_temperature', 'compute_contrastive_loss', '_run_epoch'],
        {'torch': torch, 'F': F, 'math': math, 'Dict': dict, 'List': list, 'Tuple': tuple,
         'Optional': Optional, 'Any': Any, 'SupConLoss': supcon, 'logging': logging,
         'defaultdict': defaultdict, 'tqdm': lambda x, **kw: x, 'roc_auc_score': roc_auc_score,
         'classification_report': classification_report, 'print_metrics': lambda *a: None,
         'gen_mapping_dict': lambda cfg: dict(enumerate(cfg.mapping))},
        replacements=[('.cuda()', ''),
            ('self.correct_coarse +=', 'logits_coarse = logits_coarse.reshape(1, -1)\n        logits_fine = logits_fine.reshape(1, -1)\n        labels_coarse = labels_coarse.reshape(1)\n        labels_fine = labels_fine.reshape(1)\n        self.correct_coarse +='),
            ('F.softmax(logits_coarse)', 'F.softmax(logits_coarse, dim=1)'),
            ('F.softmax(logits_fine)', 'F.softmax(logits_fine, dim=1)')])._run_epoch


def test_hmil_current_source_shape_failure_and_prior_scalar_head_evidence():
    net = original_model()([2, 3])
    _, logits, _ = net(torch.randn(5, 1024))
    assert [x.shape for x in logits] == [torch.Size([2, 2]), torch.Size([3, 3])]
    with pytest.raises(ValueError):
        F.cross_entropy(logits[1], torch.tensor([2]))
    root, prior_sha = verified_source('hmil_prior')
    prior = pinned_text(root, prior_sha, 'src/models.py')
    assert 'self.coarse_classifier = nn.Linear(size[0], 1)' in prior
    assert 'self.fine_classifier(M2).permute(1, 0)' in prior
    # Compare the prior author architecture itself with repaired current output.
    prior = prior.replace('GatedAttention(L=', 'GatedAttention(input_dim=').replace(', D=', ', hidden_dim=')
    scope = {'torch': torch, 'nn': nn, 'F': F, 'List': list, 'Tuple': tuple}
    tree = ast.parse(prior)
    nodes = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))], type_ignores=[])
    exec(compile(nodes, '<pinned prior HMIL>', 'exec'), scope)  # noqa: S102 -- verified author Git object
    model = HierarchicalMIL(1024, 3, labels()).eval()
    before = scope['HMIL']([2, 3]).eval()
    state = {k.replace('coarse_attention.', 'coarse_attention.0.').replace('fine_attention.', 'fine_attention.0.'): v
             for k, v in model.network.state_dict().items()}
    before.load_state_dict(state)
    x = torch.randn(11, 1024)
    expected = before(x)
    got = model(x)
    for actual, reference in [(got['logits'], expected[1][1]), (got['coarse_logits'], expected[1][0]),
                              (got['attention'], expected[0][1]), (got['embedding'], expected[2])]:
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize('dropout', [False, True])
@pytest.mark.parametrize('epoch', [0, 4, 9])
def test_hmil_author_full_epoch_loss_gradient_and_adam_updates(epoch, dropout):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = HierarchicalMIL(1024, 3, labels(), dropout=dropout, batch_size=4)
    model.configure_training({'epochs': 10, 'grad_clip': None}); model.start_epoch(epoch)
    reference = reference_network(model, dropout)
    bags = [torch.randn(n, 1024) for n in [5, 7, 9, 4, 13, 8, 11, 3, 6]]
    targets = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    groups = [[{'target': y, 'i': i}] for i, y in enumerate(targets)]
    bundle = SimpleNamespace(load=lambda row: (bags[row['i']], None))
    loader = [(bags[i:i+4], [(model.hierarchy.mapping[t], t) for t in targets[i:i+4]])
              for i in range(0, len(bags), 4)]
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    refopt = torch.optim.Adam(reference.parameters(), lr=1e-4)
    torch.manual_seed(42)
    expected = reference_epoch()(reference, loader, epoch, SimpleNamespace(
        mapping=model.hierarchy.mapping, n_class=[2, 3], num_epochs=10), refopt)
    torch.manual_seed(42)
    got = hmil_epoch(model, bundle, groups, opt, 'cpu', {'max_instances': None, 'grad_clip': None})
    assert sum(v['loss'] for v in got)/len(got) == pytest.approx(expected['loss'], abs=2e-6)
    assert len(got) == 3  # includes final single-bag batch
    for (name, p), (rname, q) in zip(model.network.named_parameters(), reference.named_parameters()):
        assert name == rname
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        torch.testing.assert_close(p, q, rtol=0, atol=0)


@pytest.mark.parametrize('term', ['coarse_ce', 'fine_ce', 'bag_alignment', 'attention_matching', 'contrastive'])
def test_hmil_each_required_loss_has_nonzero_gradient(term):
    torch.manual_seed(8)
    model = HierarchicalMIL(16, 3, labels(), hidden=8, attention_dim=4)
    model.configure_training({'epochs': 10, 'grad_clip': None}); model.start_epoch(5)
    outs = [model(torch.randn(n, 16)) for n in [5, 7, 9, 6]]
    losses = model.batch_loss(outs, torch.tensor([0, 1, 2, 0]))
    losses[term].backward()
    assert model.network.feature_extractor[0].weight.grad.norm() > 0
    if term in {'coarse_ce', 'attention_matching'}:
        assert model.network.coarse_attention.attention_c.weight.grad.norm() > 0
    if term != 'coarse_ce':
        assert model.network.fine_attention.attention_c.weight.grad.norm() > 0


def test_hmil_source_objective_and_curriculum_are_explicit():
    model = HierarchicalMIL(16, 3, labels())
    model.configure_training({'epochs': 10, 'grad_clip': None})
    for epoch in [0, 5, 9]:
        model.start_epoch(epoch)
        output = [model(torch.randn(5,16)), model(torch.randn(7,16))]
        terms = model.batch_loss(output, torch.tensor([1, 2]))
        alpha = 1-(epoch/10)**2
        expected = alpha*(terms['coarse_ce']+terms['attention_matching'])+terms['fine_ce']+terms['bag_alignment']+(1-alpha)*terms['contrastive']
        torch.testing.assert_close(terms['loss'], expected, rtol=0, atol=0)
        assert terms['temperature'] == pytest.approx(.45*(1+math.cos(2*math.pi*epoch/10))+.1)
        mapped = torch.cat([o['mapped_coarse_probabilities'] for o in output])
        torch.testing.assert_close(terms['bag_alignment'], F.cross_entropy(mapped, torch.tensor([0,1])))
        assert not torch.isclose(terms['bag_alignment'], F.nll_loss(mapped.log(), torch.tensor([0,1])))


@pytest.mark.parametrize('dim', [8, 17, 1024])
def test_hmil_singleton_bag_and_nonstandard_fm(dim):
    model = HierarchicalMIL(dim, 3, labels())
    output = model(torch.randn(1, dim))
    assert output['logits'].shape == (1,3)
    torch.testing.assert_close(output['attention'], torch.ones(3,1))
    model.batch_loss([output], torch.tensor([1]))['loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval(); model(torch.randn(2,dim))  # no target required for prediction


@pytest.mark.parametrize('change', [
    {'coarse_classes': ['A', 'A']}, {'coarse_classes': ['A']}, {'coarse_classes': ['A', '']},
    {'coarse_classes': [0,1]}, {'label_provenance': ''}, {'fine_to_coarse': {'0':'A','1':'B'}},
    {'fine_to_coarse': {'0':'A','1':'A','2':'A'}}, {'extra_key': 'typo'}])
def test_hierarchy_invalid_contract_rejected(change):
    bundle = SimpleNamespace(task_type='classification', classes=['0','1','2'], rows=pd.DataFrame(),
        config={'hierarchy':{'coarse_classes':['A','B'], 'fine_to_coarse':{'0':'A','1':'A','2':'B'}, 'label_provenance':'fixture'}})
    bundle.config['hierarchy'].update(change)
    with pytest.raises(ValueError):
        HierarchyLabels(bundle)


def test_hierarchy_compatibility_and_coarse_label_consistency(tmp_path):
    plain = Bundle(create_demo(tmp_path/'plain',patients=20))
    with pytest.raises(ValueError, match='hierarchy'):
        check_compatibility(require_runnable('HMIL'),plain)
    path = create_hierarchy_demo(tmp_path/'hierarchy')
    bundle = Bundle(path)
    bundle.config['training_unit'] = 'patient'
    with pytest.raises(ValueError, match='individual slide'):
        check_compatibility(require_runnable('HMIL'),bundle)
    rows = pd.read_csv(path/'slides.csv'); rows['coarse_label']='coarse_A'; rows.to_csv(path/'slides.csv',index=False)
    with pytest.raises(ValueError, match='coarse_label'):
        Bundle(path)


@pytest.mark.parametrize('unit', ['slide', 'patient'])
def test_hmil_resume_and_two_level_evaluation(tmp_path, unit):
    path = create_hierarchy_demo(tmp_path/'data')
    cfgpath=path/'task.yaml'; cfg=yaml.safe_load(cfgpath.read_text()); cfg['evaluation_unit']=unit
    cfgpath.write_text(yaml.safe_dump(cfg))
    config={'epochs':3,'threads':1,'model':{'hidden':8,'attention_dim':4,'batch_size':4,'dropout':True}}
    full=train(path,'HMIL',tmp_path/'full',config=config)
    part=train(path,'HMIL',tmp_path/'part',config=config,stop_after=1)
    train(path,'HMIL',part,config=config,resume=True)
    a,b=[torch.load(p/'last.pt',weights_only=False) for p in [full,part]]
    assert a['history']==b['history']
    for k in a['model']:
        torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
    assert not (full/'test_predictions.csv').exists()
    metrics=evaluate(full); assert metrics==evaluate(part)
    frame=pd.read_csv(full/'test_predictions.csv')
    evaluate(full); pd.testing.assert_frame_equal(frame,pd.read_csv(full/'test_predictions.csv'))
    coarse=frame[['global_patient_id','coarse_target','coarse_p_0','coarse_p_1']].rename(
        columns={'coarse_target':'target','coarse_p_0':'p_0','coarse_p_1':'p_1'})
    expected=classification_metrics(coarse,2,unit)
    for key, value in expected.items():
        if isinstance(value, float):
            assert metrics['coarse_'+key] == pytest.approx(value, abs=1e-12)
        else:
            assert metrics['coarse_'+key] == value
    assert metrics['coarse_n']==metrics['n']
    assert read_json(full/'method_protocol.json')['hierarchy']['label_provenance']
    assert read_json(full/'request.json')['method']=='hmil_source_repaired'


def test_changed_hierarchy_rejects_resume(tmp_path):
    path=create_hierarchy_demo(tmp_path/'data')
    config={'epochs':2,'model':{'hidden':8,'batch_size':4}}
    run=train(path,'HMIL',tmp_path/'run',config=config,stop_after=1)
    task=path/'task.yaml'; cfg=yaml.safe_load(task.read_text())
    cfg['hierarchy']['fine_to_coarse']['class_1']='coarse_B'
    task.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='Resume refused'):
        train(path,'HMIL',run,config=config,resume=True)


def test_invalid_model_and_training_options(tmp_path):
    bundle=Bundle(create_hierarchy_demo(tmp_path/'data'))
    for cfg in [{'hidden':0},{'attention_dim':0},{'batch_size':0},{'batch_size':1.5},{'dropout':.5}]:
        with pytest.raises((ValueError,TypeError)):
            build_bundle_model('hmil_source_repaired',bundle,cfg)
    model=build_bundle_model('hmil_source_repaired',bundle)
    for clip in [0,-1,float('nan')]:
        with pytest.raises(ValueError,match='grad_clip'):
            model.configure_training({'epochs':2,'grad_clip':clip})
