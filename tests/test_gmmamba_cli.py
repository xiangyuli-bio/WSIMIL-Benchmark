"""Actual path+method CLI for native and explicitly repaired multi-FM routes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import pytest
import torch

from milplus.data import Bundle
from milplus.gmmamba_demo import create_gmmamba_demo
from milplus.io import digest_file


def cli(*args):
    root=Path(__file__).resolve().parents[1]
    return subprocess.run([sys.executable,'-m','milplus',*map(str,args)],cwd=root,
        env={**os.environ,'PYTHONPATH':str(root/'src')+':'+os.environ.get('PYTHONPATH',''),
             'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)


@pytest.mark.parametrize('method,dimension,classes',[
    ('GMMamba',512,2),('gmmamba_fm_projection_repaired',1024,3)])
def test_actual_original_method_and_full_FM_repair_train_select_and_reload(tmp_path,method,dimension,classes):
    input_root=Path(os.environ.get('MILPLUS_GMMAMBA_CLI_INPUT_ROOT',tmp_path))
    parent=input_root/('cli_'+str(dimension)+'_'+str(classes));parent.mkdir(parents=True,exist_ok=False)
    data=create_gmmamba_demo(parent/'data',patients=3*classes,dimension=dimension,classes=classes)
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps({'epochs':7,'threads':2}))
    run=tmp_path/'run';result=cli('run','--data',data,'--method',method,'--config',cfg,'--output',run)
    assert result.returncode==0,result.stdout[-5000:]
    request=json.loads((run/'request.json').read_text());state=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    expected='gmmamba_mambamil_bimamba_source' if method=='GMMamba' else method
    assert request['method']==expected and request['config']['selection_metric']=='gmmamba_val_auc_f1'
    assert state['model']['_extra_state']['completed_epochs']==7
    assert state['model']['_extra_state']['optimizer_updates']==7*classes
    protocol=json.loads((run/'method_protocol.json').read_text())
    assert protocol['dimension']==dimension and protocol['groups']==10 and protocol['dependency']['source']=='mambamil'
    assert not (run/'test_predictions.csv').exists()
    result=cli('evaluate','--run',run);assert result.returncode==0,result.stdout[-4000:]
    prediction=digest_file(run/'test_predictions.csv')
    result=cli('evaluate','--run',run);assert result.returncode==0 and digest_file(run/'test_predictions.csv')==prediction


def test_actual_native_spatial_demo_and_mixed_task_rejection(tmp_path):
    data=tmp_path/'data';result=cli('demo','--gmmamba','--patients',6,'--output',data)
    assert result.returncode==0,result.stdout
    bundle=Bundle(data);assert bundle.dim==512 and bundle.config['synthetic'] is True
    assert all(bundle.load(row,require_coords=True)[0].shape==(257,512) for row in bundle.rows.to_dict('records'))
    result=cli('demo','--gmmamba','--muse','--output',tmp_path/'mixed')
    assert result.returncode!=0 and not (tmp_path/'mixed').exists()


def test_missing_actual_coordinates_are_rejected_before_generic_training(tmp_path):
    data=create_gmmamba_demo(tmp_path/'data',patients=6)
    for path in (data/'features').glob('*.h5'):
        with h5py.File(path,'r+') as archive:del archive['coords']
    result=cli('run','--data',data,'--method','GMMamba','--output',tmp_path/'run')
    assert result.returncode!=0 and 'coordinates' in result.stdout.lower()
