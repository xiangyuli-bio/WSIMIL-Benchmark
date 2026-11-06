import torch

from milplus.demo import create_demo
from milplus.engine import evaluate, train
from milplus.io import read_json
from milplus.teacher import MHIM


def test_teacher_mask_and_ema():
    model=MHIM(16,2,hidden=16,warmup_epochs=1,momentum=.5,dropout=0,high_mask_decay=False)
    model.epoch=1
    model.initialize_teacher()
    model.train()
    assert not model.teacher.training
    assert all(not p.requires_grad for p in model.teacher.parameters())
    out=model(torch.randn(30,16))
    assert 0 < out['kept_instances'] < 30
    assert out['regularization_loss'].requires_grad
    old=next(model.teacher.parameters()).clone()
    with torch.no_grad():
        next(model.student.parameters()).add_(1)
    model.update_teacher()
    torch.testing.assert_close(next(model.teacher.parameters()),old+.5)


def test_teacher_full_recipe_resume(tmp_path):
    data=create_demo(tmp_path/'task',patients=20)
    cfg={'epochs':3,'threads':1,'model':{'hidden':16,'warmup_epochs':1,'dropout':.1}}
    a=train(data,'mhim_abmil',tmp_path/'a',config=cfg)
    b=train(data,'mhim_abmil',tmp_path/'b',config=cfg,stop_after=1)
    assert not (b/'best.pt').exists()  # warmup model cannot become the final MHIM checkpoint
    train(data,'mhim_abmil',b,config=cfg,resume=True)
    sa=torch.load(a/'last.pt',weights_only=False); sb=torch.load(b/'last.pt',weights_only=False)
    for k in sa['model']:
        torch.testing.assert_close(sa['model'][k],sb['model'][k],rtol=0,atol=0)
    h=read_json(a/'history.json')
    assert 'regularization_loss' not in h[0]['train']
    assert h[1]['train']['regularization_loss'] > 0
    assert evaluate(a)['macro_auc'] >= 0
