"""Real public generative CLI, resumable training and independent selected reloads."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def test_public_method_name_resume_equals_uninterrupted_full_model_and_two_reloads(tmp_path):
    from milplus.taxomil_demo import create_taxomil_demo
    from milplus.engine import train
    from milplus.io import digest_file
    assets=os.environ.get('MILPLUS_TAXOMIL_ASSETS','/dev/shm/milplus_taxomil_assets_r2/gpt2')
    data=create_taxomil_demo(tmp_path/'data',assets,patients=42,patches=257)
    config=tmp_path/'recipe.json';config.write_text(json.dumps({'epochs':3,'threads':2}))
    uninterrupted=tmp_path/'uninterrupted';resumed=tmp_path/'resumed'
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
        'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    def cli(*args):
        with (tmp_path/'cli.log').open('a') as log:
            result=subprocess.run([sys.executable,'-m','milplus',*map(str,args)],env=env,
                stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        assert result.returncode==0,(tmp_path/'cli.log').read_text()[-9000:]
    cli('run','--data',data,'--method','TaxoMIL','--output',uninterrupted,'--config',config)
    train(data,'TaxoMIL',resumed,seed=2026,config={'epochs':3,'threads':2},stop_after=1)
    assert json.loads((resumed/'status.json').read_text())['state']=='interrupted'
    partial_sha=digest_file(resumed/'last.pt')
    cli('run','--data',data,'--method','TaxoMIL','--output',resumed,'--config',config,'--resume')
    assert digest_file(resumed/'last.pt')!=partial_sha
    left=torch.load(uninterrupted/'last.pt',map_location='cpu',weights_only=False)
    right=torch.load(resumed/'last.pt',map_location='cpu',weights_only=False)
    for name,a in left['model'].items():
        b=right['model'][name]
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        else:assert a==b
    assert left['optimizers'][0]['param_groups']==right['optimizers'][0]['param_groups']
    for number,a in left['optimizers'][0]['state'].items():
        b=right['optimizers'][0]['state'][number]
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert left['rng']['python']==right['rng']['python']
    torch.testing.assert_close(left['rng']['torch'],right['rng']['torch'],rtol=0,atol=0)
    assert left['history']==right['history'] and len(right['history'])==3
    assert right['model']['_extra_state']['optimizer_updates']==3
    del left,right
    assert not (resumed/'test_predictions.csv').exists()
    cli('evaluate','--run',resumed,'--subset','test')
    metrics=json.loads((resumed/'test_metrics.json').read_text());prediction=digest_file(resumed/'test_predictions.csv')
    cli('evaluate','--run',resumed,'--subset','test')
    assert json.loads((resumed/'test_metrics.json').read_text())==metrics
    assert digest_file(resumed/'test_predictions.csv')==prediction
    assert metrics['prediction_type']=='unconstrained_generated_text_exact_match'
    assert metrics['n']==14 and not any('auc' in key for key in metrics)
