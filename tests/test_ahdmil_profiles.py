"""Execute the remaining task recipes at native widths; short software budgets."""
import pytest
import torch

from milplus.ahdmil_source import RECIPES
from milplus.data import Bundle
from milplus.engine import evaluate, train
from milplus.hdmil_demo import create_hdmil_demo
from milplus.io import read_json


@pytest.mark.parametrize('task',['camelyon16','nsclc','rcc'])
def test_native_task_profile_full_stage_routes_and_class_count(tmp_path,task):
    torch.set_num_threads(1)
    profile='ahdmil_'+task+'_source_repaired';recipe=RECIPES[profile]
    data=create_hdmil_demo(tmp_path/'data',patients=10*recipe['classes'],classes=recipe['classes'],
                           dimension=1536 if task=='nsclc' else 1024)
    out=train(data,profile,tmp_path/'run',seed=2026,
        config=dict(epochs=5,threads=1,model=dict(dmin_epochs=2,joint_epochs=3,dmin_selection_start=1)))
    saved=torch.load(out/'last.pt',weights_only=False);extra=saved['model']['_extra_state']
    assert extra['updates']==dict(pretrain=2*len(Bundle(data).records('train')),
                                 lipn=3*len(Bundle(data).records('train')),dmin=3*len(Bundle(data).records('train')))
    assert extra['ema_calls']==4 and extra['stage']=='joint' and extra['stage_best']['joint'] is not None
    protocol=read_json(out/'method_protocol.json');assert protocol['native']==recipe
    evaluate(out)
    assert (out/'test_predictions.csv').exists()
