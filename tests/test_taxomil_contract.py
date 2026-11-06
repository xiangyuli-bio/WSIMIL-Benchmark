"""Meaningful taxonomy/metric/source-preservation and full author-oracle checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from milplus.registry import inspect_method,profiles
from milplus.taxomil import text_metrics
from milplus.taxomil_source import author_components


def test_generated_unknowns_prefixes_and_swapped_parents_are_strictly_incorrect():
    frame=pd.DataFrame({'taxomil_coarse_target':['benign']*4,
        'taxomil_fine_target':['normal']*4,
        'taxomil_coarse_text':['benign','benign','malignant','benign'],
        'taxomil_fine_text':['normal','normal extra','normal','unknown']})
    values=text_metrics(frame)
    assert values['taxomil_coarse_accuracy']==75
    assert values['taxomil_fine_accuracy']==50
    assert values['taxomil_holistic_accuracy']==25
    assert values['taxomil_score']==62.5
    assert not any('auc' in key for key in values)


def test_text_outputs_require_nonempty_literal_predictions():
    with pytest.raises(ValueError,match='empty'):text_metrics(pd.DataFrame())
    frame=pd.DataFrame({'taxomil_coarse_target':['benign'],'taxomil_fine_target':['normal'],
        'taxomil_coarse_text':[1],'taxomil_fine_text':['normal']})
    with pytest.raises(ValueError,match='literal text'):text_metrics(frame)


def test_original_fatal_alignment_lookup_is_reproduced_and_named_repair_is_separate():
    import torch
    _,_,_,_,_,original=author_components(repair_alignment_keys=False)
    _,_,_,_,_,repaired=author_components()
    criterion=torch.nn.CrossEntropyLoss()
    class Alignment(torch.nn.Module):
        def forward(self,features,text,labels):return criterion(features@text.T,labels)
    criteria={'c_ital':Alignment()};features=torch.eye(2);bank={'proj_coarse':torch.eye(2)};ids=torch.tensor([0,1])
    with pytest.raises(KeyError,match='coarse_ital'):
        original.compute_ital_loss(criteria,'coarse',features,bank,ids,torch.device('cpu'))
    torch.testing.assert_close(repaired.compute_ital_loss(criteria,'coarse',features,bank,ids,torch.device('cpu')),
        criterion(features,ids),rtol=0,atol=0)


def test_method_alias_binds_complete_objective_and_both_explicit_LR_profiles():
    selected=inspect_method('TaxoMIL')
    assert selected['id']=='taxomil_gpt2_all_losses_key_repaired'
    assert selected['external_source']=='taxomil'
    assert 'hierarchical_labels' in selected['required_inputs']
    assert {p['id'] for p in profiles() if p['family']=='taxomil'}=={
        'taxomil_gpt2_all_losses_key_repaired','taxomil_gpt2_paper_lr_key_repaired'}


def test_every_prior293_execution_contract_preserved():
    fixture=json.loads((Path(__file__).parent/'fixtures/taxomil_v86_reference_profiles293.json').read_text())
    current={p['id']:p for p in profiles()}
    assert len(fixture)==293
    from milplus.registry import methods
    assert set(current)=={name for method in methods() for name in method['profiles']}
    assert len(current)>=295
    for entry in fixture:assert current[entry['id']]==entry


def test_full_pretrained_author_model_train_VAL_gradients_moments_RNG_and_checkpoint(tmp_path):
    assets=Path(os.environ.get('MILPLUS_TAXOMIL_ASSETS','/dev/shm/milplus_taxomil_assets_r2/gpt2'))
    assert (assets/'milplus_assets.json').is_file(),'Full original GPT2 assets are required; no skipped or small-model replacement'
    script=Path(__file__).resolve().parents[1]/'scripts/validate_taxomil_components.py'
    with (tmp_path/'oracle.log').open('w') as log:
        result=subprocess.run([sys.executable,str(script),'--assets',str(assets),'--output',str(tmp_path/'actual')],
            stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    assert result.returncode==0,(tmp_path/'oracle.log').read_text()[-8000:]
    report=json.loads((tmp_path/'actual/validation.json').read_text())
    assert report['state']=='passed' and len(report['checks'])==7
    assert report['parameters']>200_000_000 and report['optimizer_updates']==1
    assert report['checkpoint_bytes']>2_000_000_000
