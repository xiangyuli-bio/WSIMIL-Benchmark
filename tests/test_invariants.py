import copy

import h5py
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, load_unit, seed_everything, survival_epoch, train, units
from milplus.io import read_json
from milplus.models import build_model
from milplus.recipes import cox_breslow_loss


def test_full_risk_gradient_matches_two_pass_with_dropout(tmp_path):
    data=create_demo(tmp_path/'task',patients=20,survival=True)
    bundle=Bundle(data)
    groups=units(bundle,'train')
    cfg={'max_instances':18,'grad_clip':100}
    seed_everything(4)
    direct=build_model('abmil_cox',16,1,{'hidden':16,'dropout':.2})
    replay=copy.deepcopy(direct)
    a=torch.optim.SGD(direct.parameters(),lr=.01)
    b=torch.optim.SGD(replay.parameters(),lr=.01)
    seed_everything(7)
    risks=[]
    for group in groups:
        x,c=load_unit(bundle,group,'cpu',18,train=True)
        risks.append(direct(x,coords=c)['logits'].flatten())
    loss=cox_breslow_loss(torch.cat(risks),torch.tensor([g[0]['time'] for g in groups]),
                         torch.tensor([g[0]['event'] for g in groups]))
    loss.backward(); a.step()
    seed_everything(7)
    replay_loss=survival_epoch(replay,bundle,groups,b,'cpu',cfg)
    assert replay_loss == pytest.approx(float(loss.detach()),abs=1e-6)
    for pa,pb in zip(direct.parameters(),replay.parameters()):
        torch.testing.assert_close(pa,pb,rtol=1e-5,atol=1e-7)


def test_evaluation_rejects_changed_features(tmp_path):
    data=create_demo(tmp_path/'task',patients=20)
    out=train(data,'mean_linear',tmp_path/'run')
    feature=next((data/'features').glob('*.h5'))
    with h5py.File(feature,'a') as f:
        f['features'][0,0]+=1
    with pytest.raises(ValueError,match='differ'):
        evaluate(out)


def test_new_run_refuses_overwrite(tmp_path):
    data=create_demo(tmp_path/'task',patients=20)
    out=train(data,'mean_linear',tmp_path/'run')
    original=(out/'request.json').read_bytes()
    with pytest.raises(FileExistsError):
        train(data,'mean_linear',out)
    assert (out/'request.json').read_bytes() == original


def test_ordinal_selects_qwk(tmp_path):
    data=create_demo(tmp_path/'task',patients=30,classes=3)
    p=data/'task.yaml'; cfg=yaml.safe_load(p.read_text());cfg['task_type']='ordinal'
    p.write_text(yaml.safe_dump(cfg))
    out=train(data,'abmil',tmp_path/'run',config={'epochs':1,'model':{'hidden':16},'threads':1})
    assert read_json(out/'request.json')['config']['selection_metric'] == 'qwk'
    assert evaluate(out)['qwk'] is not None


@pytest.mark.parametrize('dim,classes',[(12,2),(768,3)])
@pytest.mark.parametrize('method',['abmil','clam_sb','dsmil','transmil','dtfd','ilra','wikg','acmil_ga'])
def test_feature_dimension_and_multiclass(method,dim,classes):
    cfg={'hidden':32} if method not in {'clam_sb','dsmil'} else {}
    model=build_model(method,dim,classes,cfg).eval()
    result=model(torch.randn(25,dim))
    assert result['logits'].shape == (1,classes)
    assert torch.isfinite(result['logits']).all()
