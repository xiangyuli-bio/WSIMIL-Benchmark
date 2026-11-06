"""Prototype/DP source-step comparisons, edge cases and checkpoint continuity."""
import copy
import math
from abc import ABCMeta, abstractmethod
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import beta
from sklearn.mixture import BayesianGaussianMixture
from torch import nn
from torch.nn import functional as F

from milplus.cdpmil import CDPMILSource
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train
from milplus.io import read_json
from milplus.models import build_model
from milplus.registry import check_compatibility, require_runnable
from milplus.tpmil import TPMIL, normalized_attention, sampled_groups
from milplus.upstream import load_definitions, load_single

torch.set_num_threads(1)


@pytest.mark.parametrize('classes', [2, 3])
@pytest.mark.parametrize('gate', [False, True])
@pytest.mark.parametrize('dropout', [False, True])
def test_tpmil_full_author_training_step(classes, gate, dropout, capsys):
    model = TPMIL(1024, classes, gate=gate, dropout=dropout)
    author = load_single('tpmil', 'model_tpmil.py').CLAM_SB(n_classes=classes, gate=gate, dropout=dropout)
    author.load_state_dict(model.network.state_dict())
    trainer = load_single('tpmil', 'train_utils.py')
    # Capture gradients before the author's post-step zero_grad.
    class CapturingAdam(torch.optim.Adam):
        def step(self, closure=None):
            self.gradients = {n: p.grad.clone() if p.grad is not None else None for n,p in author.named_parameters()}
            return super().step(closure)
    ref_opt = CapturingAdam(author.parameters(), lr=.0002, weight_decay=1e-5)
    opt = torch.optim.Adam(model.parameters(), lr=.0002, weight_decay=1e-5)
    x, y = torch.randn(11, 1024), torch.tensor([classes-1])
    torch.manual_seed(32)
    trainer.train_loop_tpmil(0, author, [(x,y)], ref_opt, classes, loss_fn=nn.CrossEntropyLoss())
    torch.manual_seed(32)
    result = model(x, label=y)
    result['total_loss'].backward()
    for name, p in model.network.named_parameters():
        expected = ref_opt.gradients[name]
        assert (p.grad is None) == (expected is None), name
        if expected is not None:
            torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
    opt.step()
    for p,q in zip(model.network.parameters(), author.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    # Independently compare the author's forward output and objective at this state.
    torch.manual_seed(51)
    expected = author(x, label=y)
    torch.manual_seed(51)
    actual = model(x, label=y)
    torch.testing.assert_close(actual['total_loss'], .5*F.cross_entropy(expected[0],y)+.5*expected[3], rtol=0,atol=0)
    assert 'train_pro_loss' in capsys.readouterr().out


def test_tpmil_prototype_gradients_reach_all_instances_but_not_pseudo_attention():
    model = TPMIL(8,3,hidden=16,attention_dim=8,dropout=False)
    x = torch.randn(9,8,requires_grad=True)
    model(x,label=torch.tensor([2]))['prototype_loss'].backward()
    assert (x.grad.norm(dim=1)>0).all()
    assert model.network.prototype_loss.Dist.centers.grad.norm()>0
    for p in model.network.attention_net[-1].parameters():
        assert p.grad is None
    for p in model.network.instance_classifiers.parameters():
        assert p.grad is None  # CLAM auxiliary objective is deliberately replaced.


@pytest.mark.parametrize('n', [1,7])
def test_tpmil_constant_attention_repair_and_label_free_prediction(n):
    model = TPMIL(8,2,hidden=16,attention_dim=8,dropout=False)
    x = torch.ones(n,8)
    out = model(x,label=torch.tensor([1]))
    assert torch.isfinite(out['total_loss'])
    out['total_loss'].backward()
    assert torch.isfinite(model.network.prototype_loss.Dist.centers.grad).all()
    model.eval()
    prediction = model(x)
    with torch.no_grad():
        original = model.network(x,label=torch.tensor([0]))
    torch.testing.assert_close(prediction['logits'],original[0],rtol=0,atol=0)
    torch.testing.assert_close(normalized_attention(torch.ones(1,n)), torch.full((1,n),.5))


def test_tpmil_balancing_matches_author_sampler_weights():
    groups = [[{'target': x,'id':i}] for i,x in enumerate([0,0,0,1])]
    torch.manual_seed(4)
    actual = sampled_groups(groups,True)
    torch.manual_seed(4)
    weights = torch.tensor([4/3,4/3,4/3,4.],dtype=torch.double)
    indices = list(torch.utils.data.WeightedRandomSampler(weights,4,replacement=True))
    assert actual == [groups[i] for i in indices]
    assert sampled_groups(groups,False) is groups


def cdpmil_reference():
    return load_definitions('cdpmil','model/dpmil.py',
        ['Distribution','ReparametrizedGaussian_VI','DirichletProcess_VI','DP_Classifier'],
        {'torch':torch,'nn':nn,'F':F,'math':math,'np':np,'beta':beta,'ABCMeta':ABCMeta,'abstractmethod':abstractmethod},
        replacements=[('.cuda()', ".to('cpu')")])


@pytest.mark.parametrize('classes', [2,3])
def test_cdpmil_full_author_preprocessing_and_training_step(tmp_path, classes):
    model = CDPMILSource(8,classes,clusters=3)
    ref = cdpmil_reference().DP_Classifier(classes,10,1,8,1)
    ref.load_state_dict(model.network.state_dict())
    raw = torch.randn(19,8)
    feature_dir = tmp_path/'slide'; feature_dir.mkdir()
    torch.save(raw,feature_dir/'features.pt')
    preprocessing = load_definitions('cdpmil','DP_feats_aggr.py',['get_feats'],
        {'torch':torch,'np':np,'os':__import__('os'),'tqdm':lambda x:x,'BayesianGaussianMixture':BayesianGaussianMixture})
    preprocessing.get_feats([str(feature_dir)],3,SimpleNamespace(concentration=.1,save_dir=str(tmp_path),dataset='data'))
    centroids = np.load(tmp_path/'data/DP_EM_feats_concentration0.1/slide.npy')
    np.testing.assert_array_equal(model.aggregate(raw).numpy(),centroids)
    train_ref = load_definitions('cdpmil','main.py',['train'],
        {'torch':torch,'np':np,'tqdm':lambda x:x,'device':'cpu'}).train
    opt = torch.optim.Adam(model.network.parameters(),lr=.01,betas=(.5,.9),weight_decay=.005)
    ref_opt = torch.optim.Adam(ref.parameters(),lr=.01,betas=(.5,.9),weight_decay=.005)
    np.random.seed(41)
    loss_ref = train_ref(['slide'],[classes-1],ref,nn.CrossEntropyLoss(),ref_opt,
                         SimpleNamespace(aggr_feat_dir=str(tmp_path/'data/DP_EM_feats_concentration0.1')))
    np.random.seed(41)
    result = model(raw)
    loss = F.cross_entropy(result['logits'],torch.tensor([classes-1]))
    assert loss.item() == loss_ref
    loss.backward()
    for p,q in zip(model.network.parameters(),ref.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    opt.step()
    for p,q in zip(model.network.parameters(),ref.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    # Public train only calls infer; adding gamma updates would change the source path.
    torch.testing.assert_close(model.network.dp_process.gamma_1,torch.ones(classes))
    torch.testing.assert_close(model.network.dp_process.gamma_2,torch.full((classes,),10.))


def test_cdpmil_eval_restores_rng_and_cache_is_content_addressed():
    model = CDPMILSource(4,2,clusters=2,cache_bags=1).eval()
    x = torch.randn(13,4)
    np.random.seed(7); before = copy.deepcopy(np.random.get_state())
    p = model(x)['logits']
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1],after[1])
    torch.testing.assert_close(p,model(x)['logits'],rtol=0,atol=0)
    first = next(iter(model.cache))
    model(x+1)
    assert len(model.cache)==1 and first not in model.cache
    torch.testing.assert_close(p,model(x)['logits'],rtol=0,atol=0)
    model.train()
    assert not torch.equal(model(x)['logits'],model(x)['logits'])


@pytest.mark.parametrize('profile', ['tpmil','cdpmil_source'])
@pytest.mark.parametrize('classes', [2,3])
def test_prototype_dp_exact_resume_and_evaluation(tmp_path,profile,classes):
    data = create_demo(tmp_path/'data',patients=classes*10,classes=classes,dimension=8)
    cfg = {'epochs':3,'threads':1,'model':{'hidden':16,'attention_dim':8} if profile=='tpmil' else {'clusters':3}}
    full = train(data,profile,tmp_path/'full',config=cfg)
    part = train(data,profile,tmp_path/'part',config=cfg,stop_after=1)
    train(data,profile,part,config=cfg,resume=True)
    a,b = [torch.load(p/'last.pt',weights_only=False) for p in [full,part]]
    assert a['history']==b['history']
    for key in a['model']:
        if key=='_extra_state':
            assert a['model'][key]==b['model'][key]
        else:
            torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert not (full/'test_predictions.csv').exists()
    assert evaluate(full)==evaluate(part)
    first = pd.read_csv(full/'test_predictions.csv')
    evaluate(full)
    pd.testing.assert_frame_equal(first,pd.read_csv(full/'test_predictions.csv'))
    protocol=read_json(full/'method_protocol.json')
    assert protocol['scope']


@pytest.mark.parametrize('profile', ['tpmil','cdpmil_source'])
def test_prototype_dp_patient_concatenation_rejected(tmp_path,profile):
    bundle = Bundle(create_demo(tmp_path/'data',patients=20))
    bundle.config['training_unit']='patient'
    with pytest.raises(ValueError,match='individual slide'):
        check_compatibility(require_runnable(profile),bundle)


@pytest.mark.parametrize('profile,config', [('tpmil',{'classes':1}),('tpmil',{'dropout':.3}),
    ('tpmil',{'hidden':0}),('cdpmil_source',{'clusters':0}),('cdpmil_source',{'concentration':0}),
    ('cdpmil_source',{'classes':1}),('cdpmil_source',{'cache_bags':-1}),
    ('cdpmil_source',{'concentration':float('nan')}),('cdpmil_source',{'clusters':2.5}),
    ('cdpmil_source',{'inference_seed':-1})])
def test_invalid_prototype_dp_model_options(profile,config):
    config = dict(config)
    classes = config.pop('classes',2)
    with pytest.raises(ValueError):
        build_model(profile,8,classes,config)


def test_bad_inputs_and_missing_labels():
    tpmil=TPMIL(8,2,hidden=16,attention_dim=8)
    for x,y in [(torch.randn(3,8),None),(torch.randn(3,8),torch.tensor([2])),
                (torch.randn(3,8),torch.tensor([.5])),(torch.empty(0,8),torch.tensor([0]))]:
        with pytest.raises(ValueError):
            tpmil(x,label=y)
    cdpmil=CDPMILSource(8,2,clusters=3)
    for x in [torch.randn(2,8),torch.randn(5,7),torch.full((5,8),float('nan'))]:
        with pytest.raises(ValueError):
            cdpmil(x)


@pytest.mark.parametrize('profile', ['tpmil','cdpmil_source'])
def test_invalid_clip_is_rejected(profile):
    model = build_model(profile,8,2)
    for clip in [0,-1,float('nan')]:
        with pytest.raises(ValueError,match='grad_clip'):
            model.configure_training({'epochs':3,'grad_clip':clip})


def test_changed_scipy_blocks_checkpoint_evaluation(tmp_path,monkeypatch):
    import importlib.metadata
    data = create_demo(tmp_path/'data',patients=20,dimension=4)
    run = train(data,'cdpmil_source',tmp_path/'run',config={'epochs':1,'model':{'clusters':2}})
    request = read_json(run/'request.json')
    assert request['software']['versions']['scipy'] == importlib.metadata.version('scipy')
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda package: 'changed-test-version' if package=='scipy' else original(package))
    with pytest.raises(ValueError,match='software differs'):
        evaluate(run)
