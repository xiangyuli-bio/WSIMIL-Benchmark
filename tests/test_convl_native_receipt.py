"""Receipt schema checks only: constructed JSON is never native execution evidence."""
import copy
import importlib.util
from pathlib import Path

import pytest

from milplus.io import write_json


def verifier():
    spec = importlib.util.spec_from_file_location('_convl_receipt_test',
        Path(__file__).parents[1]/'scripts/convl_native_receipt.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.verify_native_receipt


def schema_fixture():
    # This is deliberately only a parser fixture, not a GPU validation report.
    rows = []
    for profile in ['convl_yaml_fp16_source_repaired', 'convl_yaml_fp32_reference']:
        for cohort in ['BRCA', 'COADREAD']:
            rows.append({'profile':profile,'cohort':cohort,'state':'passed','device_type':'cuda',
                'max_epochs':200,'patience':8,'epochs':9,'patients':20,'patches_per_slide':257,
                'feature_dimension':512,'parameters':112353976+(8 if cohort=='COADREAD' else 0),
                'text_layers':12,'context_tokens':16,'gradient_accumulation':2,'optimizer_updates':54,
                'all_native_mechanism_markers_updated':True,'source_full_fast_slow_and_scaler_state_verified':True,
                'source_NLL_only_selection_and_stopping_verified':True,'selected_checkpoint_reloaded_twice':True,
                'synthetic':True})
    return {'state':'passed','synthetic':True,'scope':'Constructed parser fixture; no GPU execution claim',
            'software':{'source_sha256':'fixture_source','sources_lock_sha256':'fixture_lock'},'profiles':rows}


def test_complete_schema_fixture_can_be_parsed_but_is_not_execution_evidence(tmp_path):
    report = schema_fixture(); path = tmp_path/'schema_only.json'; write_json(path, report)
    result = verifier()(path, report['software'])
    assert result['report']['scope'] == 'Constructed parser fixture; no GPU execution claim'
    assert result['sha256']


@pytest.mark.parametrize('mutation', ['source', 'missing_precision', 'cpu', 'epochs', 'patches', 'width', 'layers', 'reload'])
def test_old_incomplete_reduced_or_cpu_claim_is_rejected(tmp_path, mutation):
    report = schema_fixture(); identity = copy.deepcopy(report['software'])
    if mutation == 'source':
        report['software']['source_sha256'] = 'another_revision'
    elif mutation == 'missing_precision':
        report['profiles'].pop()
    else:
        key, value = {'cpu':('device_type','cpu'),'epochs':('max_epochs',2),'patches':('patches_per_slide',17),
            'width':('parameters',1123),'layers':('text_layers',1),'reload':('selected_checkpoint_reloaded_twice',False)}[mutation]
        report['profiles'][0][key] = value
    path = tmp_path/'invalid_schema.json'; write_json(path, report)
    with pytest.raises(ValueError, match='ConVL'):
        verifier()(path, identity)
