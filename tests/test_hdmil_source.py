"""Independent author objectives/graphs at native widths, not forward-only checks."""
import ast
import copy
import math
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.engine import restore_rng, rng_state, seed_everything
from milplus.hdmil_source import RECIPES, arguments, build_dmin, build_lipn, dmin_loss, lipn_loss
from milplus.upstream import verified_source


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def reference(path, scope):
    root, sha = verified_source('hdmil')
    text = subprocess.check_output(['git', 'show', sha+':'+path], cwd=root, text=True)
    nodes = [node for node in ast.parse(text).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    scope = dict(scope)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)
    return scope


class SourceBinarySVM(nn.Module):
    def forward(self, scores, labels):
        # Independent closed form of the original binary SmoothTop1SVM with
        # alpha=tau=1 and native hard/soft split at tau*log(1000).
        loss = []
        for row, label in zip(scores, labels):
            augmented = torch.stack([row[c]+float(c != int(label)) for c in range(2)])
            value = augmented.max() if (row.max()-row.min()).detach() >= math.log(1000) else torch.logsumexp(augmented, 0)
            loss.append(value-row[label])
        return torch.stack(loss).mean()


def test_binary_svm_hard_soft_and_tied_score_objective_gradients():
    from milplus.mammoth import BinarySmoothTop1
    scores = torch.tensor([[0.,0.],[.2,-.8],[10.,-20.],[-40.,30.],
                           [math.log(1000),0.]],requires_grad=True)
    labels = torch.tensor([0,1,0,0,1])
    original = scores.detach().clone().requires_grad_()
    actual, expected = BinarySmoothTop1()(scores,labels), SourceBinarySVM()(original,labels)
    actual.backward();expected.backward()
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(scores.grad,original.grad,rtol=1e-6,atol=1e-7)


def original_models(profile, dim):
    scope = dict(torch=torch, nn=nn, F=F, np=np, math=math)
    scope.update(reference('models/mobilenetv4.py', scope))
    scope.update(reference('models/hdmil.py', scope))
    args = arguments(profile, dim)
    kw = dict(I=dim, dropout=True, n_classes=args.n_classes, subtyping=args.subtyping,
              instance_loss_fn=SourceBinarySVM(), k_sample=args.k_sample, args=args)
    return scope['KAN_CLAM_MB_v4'](**kw), scope['KAN_CLAM_MB_v5'](**kw), scope['ImageProcessor'](args)


def original_runner(stage, network, processor, profile, dim):
    cls = reference('DMIN.py' if stage == 'dmin' else 'LIPN.py',
        dict(torch=torch, F=F, np=np, Dataset=torch.utils.data.Dataset))['DMINMIL' if stage == 'dmin' else 'LIPNMIL']
    runner = cls.__new__(cls)
    runner.args = arguments(profile, dim); runner.args.device = 'cpu'
    runner.model, runner.img_processor = network, processor
    runner.loss = nn.CrossEntropyLoss()
    runner.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
    runner.l2_loss, runner.l1_loss = nn.MSELoss(), nn.L1Loss()
    return runner


@pytest.mark.parametrize('profile,dim', [('hdmil_camelyon16_source',1024), ('hdmil_nsclc_source',1536),
                                      ('hdmil_brca_source',1024), ('hdmil_rcc_source',1536)])
def test_native_dmin_complete_author_objective_and_every_gradient(profile, dim):
    seed_everything(2026)
    port = build_dmin(profile, dim)
    original, _, _ = original_models(profile, dim); original.load_state_dict(port.state_dict())
    x = torch.randn(1, 40, dim); label = torch.tensor([RECIPES[profile]['classes']-1])
    rng = rng_state(); loss, terms = dmin_loss(port, x, label, RECIPES[profile]['mask_ratio']); loss.backward()
    after = rng_state(); restore_rng(rng)
    runner = original_runner('dmin', original, None, profile, dim)
    expected = runner.train_inference(x, label); expected.backward()
    torch.testing.assert_close(loss, expected, rtol=2e-6, atol=1e-7)
    for (name, p), (other, q) in zip(port.named_parameters(), original.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=2e-7)
    equal(after, rng_state())
    assert port.attention_net[0].out_features == 512
    assert port.attention_net[-1].attention_a[0].out_features == 256
    assert port.classifiers[0].degree == RECIPES[profile]['degree']
    assert port.instance_classifiers[int(label)].act_fun[0].cheby_coeffs.grad.abs().sum() > 0
    assert set(terms) == {'classification','kl','representation','retention','mask_rate'}
    port.eval(); original.eval()
    equal(port(x), original(x))


@pytest.mark.parametrize('profile', ['hdmil_brca_source','hdmil_rcc_source'])
def test_native_lipn_full_author_distillation_train_teacher_dropout_and_inference(profile):
    seed_everything(410)
    teacher = build_dmin(profile, 1024, 'lipn'); processor = build_lipn(profile, 1024)
    _, original_teacher, original_processor = original_models(profile, 1024)
    original_teacher.load_state_dict(teacher.state_dict()); original_processor.load_state_dict(processor.state_dict())
    teacher.requires_grad_(False); original_teacher.requires_grad_(False)
    x, rgb = torch.randn(1, 40, 1024), torch.randint(256, (1,40,16,16,3), dtype=torch.uint8)
    before = copy.deepcopy(teacher.state_dict()); rng = rng_state()
    loss, parts = lipn_loss(teacher, processor, x, rgb); loss.backward(); after = rng_state()
    restore_rng(rng)
    expected, rate, score_rate = original_runner('lipn', original_teacher, original_processor, profile, 1024).train_inference(x, torch.tensor([0]), rgb)
    expected.backward(); equal(loss, expected); equal(parts['mask_rate'], rate); equal(parts['score_rate'], score_rate)
    equal(after, rng_state()); equal(before, teacher.state_dict())
    assert all(p.grad is None for p in teacher.parameters())
    for (name,p),(other,q) in zip(processor.named_parameters(), original_processor.named_parameters()):
        assert name == other; equal(p.grad, q.grad)
    assert processor.extractor.features[0].block[0].weight.grad.abs().sum() > 0
    assert processor.extractor.classifier.weight.grad.abs().sum() > 0
    teacher.eval(); processor.eval(); original_teacher.eval(); original_processor.eval()
    equal(teacher(x, processor(rgb)), original_teacher(x, original_processor(rgb)))


def test_no_nonphysical_or_incomplete_recipe_fallback():
    with pytest.raises(ValueError):
        build_dmin('hdmil_unknown', 1024)
    with pytest.raises(ValueError):
        build_dmin('hdmil_brca_source', 1024, 'attention_only')


def test_actual_author_two_train_loops_weighted_draws_all_adams_and_stage_reset(tmp_path):
    """Execute DMINMIL.train and LIPNMIL.train, not a mirror of the adapter loop."""
    import logging
    from sklearn import metrics
    from milplus.data import Bundle
    from milplus.engine import train
    from milplus.hdmil_demo import create_hdmil_demo
    from milplus.mammoth import BinarySmoothTop1
    from torch.utils.data import DataLoader, Dataset, SequentialSampler, WeightedRandomSampler

    data = create_hdmil_demo(tmp_path/'data',patients=20,dimension=1024)
    bundle = Bundle(data)
    cfg = dict(epochs=4,threads=1,model=dict(dmin_epochs=2,lipn_epochs=2,dmin_selection_start=1))
    train(data,'HDMIL',tmp_path/'run',seed=2026,config=cfg)
    actual = torch.load(tmp_path/'run/last.pt',weights_only=False)
    bindings = dict(torch=torch,F=F,np=np,Dataset=Dataset,logging=logging,tqdm=lambda x:x)
    bindings.update({k:getattr(metrics,k) for k in ['roc_auc_score','accuracy_score','precision_score','recall_score','f1_score']})
    scope = dict(torch=torch,nn=nn,F=F,np=np,math=math)
    scope.update(reference('models/mobilenetv4.py',scope));scope.update(reference('models/hdmil.py',scope))

    class SourceDataset(Dataset):
        def __init__(self,subset,stage):self.rows,self.stage=bundle.records(subset),stage
        def __len__(self):return len(self.rows)
        def __getitem__(self,index):
            row=self.rows[index]; x,images=bundle.hdmil.load(row)
            return (x,row['target'],row['slide_id']) if self.stage=='dmin' else (x,row['target'],row['slide_id'],images,row['slide_id'])

    runners=[]; rngs=[]
    for stage in ['dmin','lipn']:
        seed_everything(2026)
        args=arguments('hdmil_brca_source',1024);args.device='cpu';args.n_epochs=2
        kw=dict(I=1024,dropout=True,n_classes=2,subtyping=True,k_sample=32,
                instance_loss_fn=BinarySmoothTop1(),args=args)
        network=scope['KAN_CLAM_MB_v4' if stage=='dmin' else 'KAN_CLAM_MB_v5'](**kw)
        processor=scope['ImageProcessor'](args) if stage=='lipn' else None
        if stage=='lipn':
            network.load_state_dict(torch.load(tmp_path/'dmin_selected.pt',weights_only=True))
            network.requires_grad_(False)
        cls=reference('DMIN.py' if stage=='dmin' else 'LIPN.py',bindings)['DMINMIL' if stage=='dmin' else 'LIPNMIL']
        runner=cls.__new__(cls);runner.args=args;runner.model=network;runner.img_processor=processor
        runner.loss=nn.CrossEntropyLoss();runner.kl_loss=nn.KLDivLoss(reduction='batchmean',log_target=True)
        runner.l1_loss=nn.L1Loss();runner.l2_loss=nn.MSELoss()
        runner.optimizer=torch.optim.Adam(network.parameters() if stage=='dmin' else processor.parameters(),
                                          lr=3e-4 if stage=='dmin' else 1e-4,weight_decay=1e-5)
        rows=bundle.records('train');counts=np.bincount([r['target'] for r in rows])
        weights=torch.DoubleTensor([len(rows)/counts[r['target']] for r in rows])
        trainset=SourceDataset('train',stage);validset=SourceDataset('val',stage)
        runner.train_loader=DataLoader(trainset,batch_size=1,sampler=WeightedRandomSampler(weights,len(weights),replacement=True))
        runner.valid_loader=DataLoader(validset,batch_size=1,sampler=SequentialSampler(validset))
        runner.counter=0;runner.patience=20 if stage=='dmin' else 5;runner.stop_epoch=1 if stage=='dmin' else 0
        runner.best_loss=np.inf;runner.flag=1;runner.best_valid_metrics=None
        runner.ckpt_name=str(tmp_path/'dmin_selected.pt')
        runner.ckpt_name1=str(tmp_path/'lipn_teacher.pt');runner.ckpt_name2=str(tmp_path/'lipn_processor.pt')
        runner.train();runners.append(runner);rngs.append(rng_state())
    equal(actual['model']['_extra_state']['best_dmin'],torch.load(tmp_path/'dmin_selected.pt',weights_only=True))
    for prefix,module in [('dmin',runners[0].model),('teacher',runners[1].model),('processor',runners[1].img_processor)]:
        equal({k[len(prefix)+1:]:v for k,v in actual['model'].items() if k.startswith(prefix+'.')},module.state_dict())
    equal(actual['optimizers'],[r.optimizer.state_dict() for r in runners])
    equal(actual['rng'],rngs[-1])
    selected=torch.load(tmp_path/'run/best.pt',weights_only=False)['model']
    equal({k[len('processor.'):]:v for k,v in selected.items() if k.startswith('processor.')},
          torch.load(tmp_path/'lipn_processor.pt',weights_only=True))
