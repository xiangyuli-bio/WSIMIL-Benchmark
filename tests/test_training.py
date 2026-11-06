import numpy as np
import pandas as pd
import pytest
import torch

from milplus.demo import create_demo
from milplus.engine import evaluate, train
from milplus.io import read_json
from milplus.models import build_model
from milplus.recipes import classification_step, cox_breslow_loss, make_optimizers

torch.set_num_threads(1)


@pytest.mark.parametrize('method', ['mean_mil','max_mil','abmil','gated_abmil','clam_sb','clam_mb','dsmil','transmil','dtfd', 'ilra', 'wikg', 'acmil_ga', 'aem_abmil'])
def test_real_train_save_reload_predict(tmp_path, method):
    data = create_demo(tmp_path/'task', patients=20)
    cfg = {'epochs': 2, 'threads': 1}
    if method in ['mean_mil','max_mil','abmil','gated_abmil','transmil','dtfd', 'ilra', 'wikg', 'acmil_ga', 'aem_abmil']:
        cfg['model'] = {'hidden': 32}
    out = train(data, method, tmp_path/'run', config=cfg)
    assert read_json(out/'status.json')['state'] == 'completed'
    assert not (out/'test_predictions.csv').exists()
    first = evaluate(out)
    assert 0 <= first['macro_auc'] <= 1
    predictions = pd.read_csv(out/'test_predictions.csv')
    second = evaluate(out)
    pd.testing.assert_frame_equal(predictions, pd.read_csv(out/'test_predictions.csv'))
    assert first == second


def test_clam_updates_instance_head():
    model = build_model('clam_sb', 16, 2, {'subtyping': True})
    before = [p.detach().clone() for p in model.model.instance_classifiers.parameters()]
    optimizers = make_optimizers(model, 'clam_sb', .001, 0)
    result = classification_step(model, 'clam_sb', torch.randn(24,16), None, torch.tensor([1]), optimizers)
    assert result['instance_loss'] > 0
    assert all(not torch.equal(a,b) for a,b in zip(before, model.model.instance_classifiers.parameters()))


def test_dtfd_both_tiers_update():
    model = build_model('dtfd', 16, 2, {'hidden': 16})
    first = model.reduce[0].weight.detach().clone()
    second = model.tier2[-1].weight.detach().clone()
    losses = classification_step(model,'dtfd',torch.randn(24,16),None,torch.tensor([1]),make_optimizers(model,'dtfd',.001,0))
    assert losses['tier1_loss'] > 0 and losses['tier2_loss'] > 0
    assert not torch.equal(first,model.reduce[0].weight)
    assert not torch.equal(second,model.tier2[-1].weight)


def test_exact_resume(tmp_path):
    data = create_demo(tmp_path/'task', patients=20)
    cfg = {'epochs': 3, 'threads': 1, 'model': {'hidden': 16, 'dropout': .2}, 'patience': 8}
    full = train(data,'abmil',tmp_path/'full',config=cfg)
    part = train(data,'abmil',tmp_path/'partial',config=cfg,stop_after=1)
    assert read_json(part/'status.json')['state'] == 'interrupted'
    train(data,'abmil',part,config=cfg,resume=True)
    a,b = [torch.load(p/'last.pt',weights_only=False) for p in (full,part)]
    for key in a['model']:
        assert torch.equal(a['model'][key],b['model'][key]), key
    assert a['history'] == b['history']


def test_resume_changed_data_rejected(tmp_path):
    data = create_demo(tmp_path/'task', patients=20)
    cfg = {'epochs': 2, 'model': {'hidden': 16}}
    out = train(data,'abmil',tmp_path/'run',config=cfg,stop_after=1)
    with pytest.raises(ValueError,match='Resume refused'):
        train(data,'abmil',out,seed=123,config=cfg,resume=True)


def test_linear_train_only_scaler(tmp_path):
    data = create_demo(tmp_path/'task', patients=20)
    out = train(data, 'mean_linear', tmp_path/'run')
    from milplus.data import Bundle
    from milplus.linear import matrix
    x,_ = matrix(Bundle(data),'train')
    weights=np.load(out/'linear.npz')
    np.testing.assert_allclose(weights['mean'],x.mean(0))
    assert evaluate(out)['macro_auc'] >= 0


def test_cox_risk_sets_and_ties():
    r=torch.tensor([.3,.7,-.2],requires_grad=True)
    t=torch.tensor([1.,1.,3.]); e=torch.tensor([1.,1.,0.])
    expected=-(r[:2]-torch.logsumexp(r,0)).mean()
    actual=cox_breslow_loss(r,t,e)
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert torch.isfinite(r.grad).all()


def test_survival_train(tmp_path):
    data=create_demo(tmp_path/'task',patients=20,survival=True)
    out=train(data,'gated_abmil_cox',tmp_path/'run',config={'epochs':2,'model':{'hidden':16},'threads':1})
    assert 0 <= evaluate(out)['c_index'] <= 1
