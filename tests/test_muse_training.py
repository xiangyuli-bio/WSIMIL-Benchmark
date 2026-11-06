"""Full20-view source equivalence, actual unified recovery and real CLI."""
import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from milplus.data import Bundle
from milplus.engine import evaluate,seed_everything,train
from milplus.io import digest_file
from milplus.muse import MUSEMIL
from milplus.muse_demo import create_muse_demo
from milplus.muse_source import diagnostic_rng_port
from milplus.muse_training import author_training

WEIGHTS='/mnt/xiangyuli-private-gpfs/mil_baseline_puls_work/model_assets/vlsa_conch/conch.pth'


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    return create_muse_demo(tmp_path_factory.mktemp('muse_training')/'data',WEIGHTS,patients=6,patches=257)


def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_two_full_epochs_match_original_twenty_view_loop_weights_moments_and_RNG(data):
    torch.set_num_threads(8);seed_everything(2026);bundle=Bundle(data)
    owner=MUSEMIL(bundle)
    cfg={'epochs':2,'lr':.0001,'weight_decay':.00001,'patience':20,'threads':8,
         'selection_metric':'muse_val_error','max_instances':None,'grad_clip':None}
    owner.configure_training(cfg);actual_opt=owner.make_optimizer(.0001,.00001)
    reference=copy.deepcopy(owner.network)
    # Restore the class's actual source method, rather than copying the owner's
    # instrumentation closure into an ostensibly independent reference model.
    del reference.retrieval_k
    ref_opt=torch.optim.Adam(reference.parameters(),lr=.0001,weight_decay=.00001)
    source=author_training();initial_rng=torch.get_rng_state()
    for epoch in range(2):
        owner.start_epoch(epoch);owner.epoch(bundle,actual_opt,torch.device('cpu'))
        owner.predict_subset(bundle,'val',torch.device('cpu'),cfg)
    final_rng=torch.get_rng_state();torch.set_rng_state(initial_rng)
    for epoch in range(2):
        source.train_loop(SimpleNamespace(model_type='Text_retrevial'),epoch,reference,
                          owner.loader('train')[2],ref_opt,2)
        reference.eval();reference.is_training=False
        with torch.no_grad():
            for _,_,feature,_,label in owner.loader('val')[2]:reference(x=feature,label=label)
        diagnostic_rng_port(reference)
    equal(owner.network.state_dict(),reference.state_dict())
    equal(actual_opt.state_dict(),ref_opt.state_dict())
    assert torch.equal(final_rng,torch.get_rng_state())
    assert owner.optimizer_updates==80
    owner.validate_optimizer_state(actual_opt)
    assert len(owner.retrieval_history)==2 and all(len(rows)==2 for rows in owner.retrieval_history)


def test_real_full_state_interruption_resume_and_two_independent_reloads(data,tmp_path):
    config={'epochs':2,'threads':8}
    full=train(data,'muse_released_source',tmp_path/'full',seed=2026,config=config)
    partial=train(data,'muse_released_source',tmp_path/'partial',seed=2026,config=config,stop_after=1)
    train(data,'muse_released_source',partial,seed=2026,config=config,resume=True)
    a=torch.load(full/'last.pt',map_location='cpu',weights_only=False)
    b=torch.load(partial/'last.pt',map_location='cpu',weights_only=False)
    for key in ['model','optimizers','history','best','bad']:equal(a[key],b[key])
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    assert a['rng']['python']==b['rng']['python']
    assert a['model']['_extra_state']['optimizer_updates']==80
    del a,b
    assert not (partial/'test_predictions.csv').exists()
    metrics=evaluate(partial);prediction=digest_file(partial/'test_predictions.csv')
    assert evaluate(partial)==metrics and digest_file(partial/'test_predictions.csv')==prediction


def test_real_CLI_uses_dataset_path_and_original_method_name(data,tmp_path):
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps({'epochs':1,'threads':8}))
    result=subprocess.run([sys.executable,'-m','milplus','run','--data',str(data),'--method','MUSE',
                           '--output',str(tmp_path/'run'),'--config',str(cfg)],
                          env=os.environ.copy(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    assert result.returncode==0,result.stdout[-4000:]
    checkpoint=torch.load(tmp_path/'run/last.pt',map_location='cpu',weights_only=False)
    assert checkpoint['model']['_extra_state']['completed_epochs']==1
    assert checkpoint['model']['_extra_state']['optimizer_updates']==40
