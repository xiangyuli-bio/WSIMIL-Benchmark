"""Mechanism, state restoration and end-to-end checks for the expanded adapters."""
import copy

import pandas as pd
import pytest
import torch

from milplus.augmentation import phenotype_partition, remix_augment
from milplus.batched import padded_subbags
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, train
from milplus.io import read_json
from milplus.models import build_model
from milplus.prototype_models import PrototypeMIL, training_patch_sample
from milplus.recipes import classification_step, make_optimizers, probabilities
from milplus.upstream import source_root

NEW = ['frmil','focusmil','dgrmil','chowder','varpool','varpool_cox','h2t','panther',
       'ibmil','additive_mil','linmil','fouriermil','retmil','pneamil','mrblock',
       'carmil','fcmil','asmil','attrimil','nnmil','madmil','litemil','rrt',
       'psemix_abmil','prodiv_abmil','remix_abmil']


def config(method, epochs=2):
    result = {'epochs':epochs,'threads':1,'model':{'hidden':32}}
    if method in {'h2t','panther','frmil'}:
        result['model'] = {}
    if method in {'ibmil','dgrmil'}:
        result['model']['warmup_epochs'] = 1
    if method == 'retmil':
        result['model']['chunk_len'] = 16
    if method == 'rrt':
        result['model']['region_num'] = 2
    if method in {'psemix_abmil','prodiv_abmil'}:
        result['model'].update(pseudo_bags=3,phenotypes=2,batch_size=4)
    return result


def needs_source(method):
    if method in {'frmil','rrt','dgrmil'} and not source_root(method).is_dir():
        pytest.skip(f'Run milplus setup --method {method} to enable author-source integration test')


@pytest.mark.parametrize('method', NEW)
def test_expanded_train_reload(tmp_path, method):
    needs_source(method)
    data = create_demo(tmp_path/'data', patients=20, survival=method.endswith('_cox'))
    output = train(data,method,tmp_path/'run',config=config(method))
    assert read_json(output/'status.json')['state'] == 'completed'
    assert not (output/'test_predictions.csv').exists()
    evaluate(output)
    first = pd.read_csv(output/'test_predictions.csv')
    evaluate(output)
    pd.testing.assert_frame_equal(first,pd.read_csv(output/'test_predictions.csv'))


@pytest.mark.parametrize('method',['asmil','attrimil','ibmil','nnmil','frmil','dgrmil','psemix_abmil','remix_abmil'])
def test_method_state_resume_exact(tmp_path, method):
    needs_source(method)
    data = create_demo(tmp_path/'data',patients=20)
    cfg = config(method,epochs=3)
    full = train(data,method,tmp_path/'full',config=cfg)
    split = train(data,method,tmp_path/'split',config=cfg,stop_after=1)
    train(data,method,split,config=cfg,resume=True)
    a,b = [torch.load(p/'last.pt',weights_only=False) for p in (full,split)]
    assert a['history'] == b['history']
    def equivalent(left,right):
        if isinstance(left,torch.Tensor):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        elif isinstance(left,dict):
            assert left.keys() == right.keys()
            for k in left: equivalent(left[k],right[k])
        elif isinstance(left,(tuple,list)):
            assert len(left)==len(right)
            for x,y in zip(left,right):equivalent(x,y)
        else: assert left==right
    equivalent(a['model'],b['model'])


def test_prototypes_never_read_holdout(tmp_path):
    data = create_demo(tmp_path/'data',patients=20)
    bundle = Bundle(data)
    original = bundle.load
    train_ids = {row['slide_id'] for row in bundle.records('train')}
    def guarded(row, *args, **kwargs):
        assert row['slide_id'] in train_ids, 'Holdout read during prototype preparation'
        return original(row,*args,**kwargs)
    bundle.load = guarded
    assert training_patch_sample(bundle,17).shape == (17,16)
    for method in ['h2t','panther','remix_abmil','psemix_abmil']:
        model = build_model(method,16,2,config(method)['model'])
        model.prepare(bundle,2026,'cpu')


def test_panther_repeated_instances_preserve_valid_mixture():
    model = PrototypeMIL(4,2,'panther',prototypes=3)
    model.prototypes.copy_(torch.randn(3,4))
    model.prepared.fill_(True)
    vector = model.representation(torch.ones(80,4))
    assert torch.isfinite(vector).all()
    torch.testing.assert_close(vector[0,:3].sum(),torch.tensor(1.))
    assert (vector[0,-12:] > 0).all()


def test_nnmil_padding_and_probability_ensemble():
    model = build_model('nnmil',16,2,{'hidden':4,'dropout':0.}).eval()
    x = torch.randn(9,16)
    unpadded = model(x)
    padded = torch.cat((x,torch.randn(5,16)*100))
    mask = torch.tensor([[True]*9+[False]*5])
    with_padding = model(padded,valid_mask=mask)
    torch.testing.assert_close(probabilities('nnmil',unpadded),probabilities('nnmil',with_padding))
    torch.testing.assert_close(unpadded['probabilities'],unpadded['chunk_probabilities'].mean(0))
    bags, valid = padded_subbags([torch.randn(2,16),torch.randn(10,16)],5)
    assert valid.sum(1).tolist()==[2,5]
    assert torch.count_nonzero(bags[0,2:])==0


def test_asmil_anchor_frozen_but_updated_after_step():
    model=build_model('asmil',16,2,{'hidden':32,'ema_decay':.5}).train()
    anchor=copy.deepcopy(model.model.anchor.state_dict())
    assert not model.model.anchor.training
    classification_step(model,'asmil',torch.randn(24,16),None,torch.tensor([1]),make_optimizers(model,'asmil',.01,0))
    assert all(p.grad is None for p in model.model.anchor.parameters())
    assert any(not torch.equal(anchor[k],v) for k,v in model.model.anchor.state_dict().items())


def test_attrimil_requires_physical_coords():
    model=build_model('attrimil',16,2)
    with pytest.raises(ValueError,match='coordinates'):
        model(torch.randn(24,16),label=torch.tensor([1]))


def test_binary_only_methods_refuse_multiclass():
    for method in ['frmil','focusmil','dgrmil']:
        with pytest.raises(ValueError,match='binary'):
            build_model(method,16,3)


def test_division_and_replacement_are_not_generic_mixup():
    x=torch.randn(80,8)
    partition=phenotype_partition(x,4,3,refinements=2)
    assert partition.shape==(80,) and set(partition.tolist())==set(range(4))
    y=x+20
    mixed=remix_augment(x,y,torch.zeros(80,2,8),rate=1,strength=.3,mode='replace')
    # Replacement picks actual donor instances; it is not a convex label/feature mix.
    assert (torch.cdist(mixed,y).min(-1).values<1e-3).all()


def test_full_matrix_retains_unimplemented_and_incompatible(tmp_path):
    from milplus.execution import matrix
    from milplus.registry import methods
    data=create_demo(tmp_path/'data',patients=30,classes=3)
    plan=matrix(data,['all'],['toy'],[2026],tmp_path/'matrix.json')
    assert len(plan)==len(methods())==164
    # Family identity persists when an admitted alias resolves to a concrete
    # execution profile. Check every planned family, not an obsolete alias key.
    rows={row['family']:row for row in plan}
    assert set(rows)=={method['id'] for method in methods()}
    for method in methods():
        if method['status']=='pending_adapter':
            assert rows[method['id']]['state']=='blocked' and rows[method['id']]['reason']
    assert rows['zoommil']['state']=='blocked'
    assert rows['zoommil']['method']=='zoommil_bright_source_repaired'
    assert 'three-scale' in rows['zoommil']['reason']
    assert rows['frmil']['state']=='blocked' and '2 classes' in rows['frmil']['reason']
    assert rows['abmil']['state']=='ready'


def test_default_catalog_is_full_scope(capsys):
    from milplus.cli import main
    assert main(['methods'])==0
    output=capsys.readouterr().out
    assert 'zoommil' in output and 'qpmilvl' in output and '164 admission records' in output
