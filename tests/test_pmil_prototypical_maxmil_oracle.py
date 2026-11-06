"""Execute author's raw-CNN train/pred bodies with a CPU-only transfer port."""
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from milplus.data import Bundle
from milplus.engine import seed_everything,rng_state
from milplus.icmil_coupled_training import load,Progress
from milplus.pmil_prototypical_demo import create_demo
from milplus.pmil_prototypical_source import encoder_definitions
from milplus.pmil_prototypical_training import Patches,encoder_phase
from milplus.upstream import load_definitions
from test_pmil_prototypical_components import equal

WEIGHTS=Path(os.environ.get('MILPLUS_PMIL_WEIGHTS',Path.home()/'.cache/milplus/pmil/resnet34-333f7ec4.pth'))


class CPUTransfer:
    def __init__(self,loader):self.loader=iter(loader)
    def next(self):
        value=next(self.loader,None)
        return None if value is None else value[0]


def test_full_original_raw_top1_training_Adam_scheduler_BN_and_no_grad_validation(tmp_path):
    data=create_demo(tmp_path/'data',WEIGHTS);bundle=Bundle(data);assets=bundle.pmil_prototypical
    torch.set_num_threads(2);seed_everything(24)
    encoder_phase(bundle,assets,tmp_path/'adapted',1,torch.device('cpu'),Progress(tmp_path,None))
    actual=load(tmp_path/'adapted/last.pt');actual_rng=rng_state()
    original=load_definitions('pmil_prototypical','max-mil.py',
        ['inference','train','pred','group_argtopk','group_max'],
        {'torch':torch,'np':np,'nn':nn,'F':F,'device':torch.device('cpu'),
         'args':SimpleNamespace(nepochs=1,batch_size=256),'data_prefetcher':CPUTransfer},
        replacements=[('                    for i, slide in enumerate(train_dset.slidenames):\n                    slidename =',
                       '                    for i, slide in enumerate(train_dset.slidenames):\n                        slidename ='),
                      ('input = input.cuda()',"input = input.to('cpu')"),
                      ('target = target.cuda()',"target = target.to('cpu')")])
    seed_everything(24);source=encoder_definitions()
    model=source.ResNet(source.BasicBlock,[3,4,6,3]);model.load_state_dict(torch.load(WEIGHTS,weights_only=True),strict=True)
    model.fc=nn.Linear(512,2)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20],gamma=.1)
    train=Patches(assets,bundle.records('train'))
    probabilities,_=original.inference(0,DataLoader(train,batch_size=256,shuffle=False),model)
    groups=np.concatenate([np.full(assets.count(row),i,dtype=np.int64) for i,row in enumerate(bundle.records('train'))])
    top=original.group_argtopk(groups,probabilities,1)
    selected=[train.references[i] for i in top];selected=random.sample(selected,len(selected))
    chosen=Patches(assets,bundle.records('train'),selected)
    train_loss=original.train(0,DataLoader(chosen,batch_size=128,shuffle=True),model,nn.CrossEntropyLoss(),optimizer)
    scheduler.step()
    val=Patches(assets,bundle.records('val'))
    probabilities,_=original.inference(0,DataLoader(val,batch_size=256,shuffle=False),model)
    groups=np.concatenate([np.full(assets.count(row),i,dtype=np.int64) for i,row in enumerate(bundle.records('val'))])
    top=original.group_argtopk(groups,probabilities,1)
    chosen=Patches(assets,bundle.records('val'),[val.references[i] for i in top])
    val_loss=original.pred(0,DataLoader(chosen,batch_size=128,shuffle=True),model,nn.CrossEntropyLoss(),optimizer)
    equal(actual['model'],model.state_dict());equal(actual['optimizer'],optimizer.state_dict())
    equal(actual['scheduler'],scheduler.state_dict());equal(actual_rng,rng_state())
    assert train_loss==actual['history'][0]['train_loss']
    assert val_loss==actual['history'][0]['validation']['patch_ce']
    assert actual['updates']==1 and actual['validation_backprop'] is False
