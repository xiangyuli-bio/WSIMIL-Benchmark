"""DT-MIL CLI and validation of actual TRAIN-only cached-view contracts."""
import h5py
import numpy as np
import pytest
import torch

from milplus.cli import dispatch, parser
from milplus.data import Bundle
from milplus.dtmil import DTMIL
from milplus.dtmil_demo import create_dtmil_demo


def test_cli_creates_real_declared_synthetic_grid_and_refuses_mixed_formats(tmp_path):
    root=dispatch(parser().parse_args(['demo','--dtmil','--patients','20','--output',str(tmp_path/'demo')]))['created']
    b=Bundle(root);assert b.config['synthetic'] and b.dim==1280
    assert b.encoder['spatial']['patch_stride']==256
    x,c=b.load(b.records('train')[0],require_coords=True)
    assert x.shape==(42,1280) and c.unique(dim=0).shape==(42,2)
    for flag in ['--pamil-policy','--focus','--raw-images','--survival','--omics']:
        with pytest.raises(ValueError,match='mixed input'):
            dispatch(parser().parse_args(['demo','--dtmil',flag,'--output',str(tmp_path/'bad')]))
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('kind',['missing','row_count','integer','nonfinite','empty_views'])
def test_bad_cached_train_views_rejected_during_prepare_before_any_updates(tmp_path,kind):
    b=Bundle(create_dtmil_demo(tmp_path/'data',dimension=8,cached_views=True))
    p=b.root/b.records('train')[0]['path']
    with h5py.File(p,'r+') as f:
        old=f['train_views'][:];del f['train_views']
        if kind=='row_count':f['train_views']=old[:-1]
        elif kind=='integer':f['train_views']=old.astype('int64')
        elif kind=='nonfinite':old[0,0,0]=np.nan;f['train_views']=old
        elif kind=='empty_views':f['train_views']=old[:,:0]
    b=Bundle(b.root);model=DTMIL(b,map_size=8,batch_size=4)
    with pytest.raises(ValueError,match='TRAIN views|TRAIN-only|TRAIN|views'):
        model.prepare(b,2026,'cpu')
    assert model.optimizer_steps==0


def test_bad_heldout_views_are_not_read_and_prepare_does_not_consume_randomness(tmp_path):
    b=Bundle(create_dtmil_demo(tmp_path/'data',dimension=8,cached_views=True))
    with h5py.File(b.root/b.records('test')[0]['path'],'r+') as f:
        del f['train_views'];f['train_views']=np.array([np.nan])
    model=DTMIL(Bundle(b.root),map_size=8,batch_size=4)
    np.random.seed(39);torch.manual_seed(41)
    a=np.random.get_state();t=torch.get_rng_state().clone()
    model.prepare(model.bundle,2026,'cpu')
    later=np.random.get_state()
    assert all(np.array_equal(x,y) for x,y in zip(a,later))
    assert torch.equal(t,torch.get_rng_state())
