"""Catalog labels resolve existing recipes without changing explicit profiles."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from milplus import registry
from milplus.cli import main
from milplus.demo import create_demo
from milplus.pneamil_spec import PROFILES as PNEA_PROFILES, VAL_PROFILE
from milplus.asmil_spec import PROFILES as ASMIL_PROFILES, DEFAULT_PROFILE as ASMIL_DEFAULT
from milplus.attrimil_spec import PROFILES as ATTRI_PROFILES, DEFAULT_PROFILE as ATTRI_DEFAULT
from milplus.carmil_spec import PROFILES as CAR_PROFILES, DEFAULT_PROFILE as CAR_DEFAULT
from milplus.mrblock_spec import PROFILES as MR_PROFILES, DEFAULT_PROFILE as MR_DEFAULT
from milplus.nnmil_spec import PROFILES as NN_PROFILES, DEFAULT_PROFILE as NN_DEFAULT
from milplus.fcmil_spec import PROFILES as FC_PROFILES, DEFAULT_PROFILE as FC_DEFAULT
from milplus.fouriermil_spec import PROFILES as FOURIER_PROFILES
from milplus.dyko_spec import PROFILES as DYKO_PROFILES
from milplus.dspamil_spec import PROFILES as DSPA_PROFILES
from milplus.feather_spec import PROFILES as FEATHER_PROFILES

PARENT = json.loads((Path(__file__).parent/'fixtures/catalog_parent120_routes.json').read_text())
FIXED = {
    'frmil': 'frmil', 'varmil': 'varmil', 'camil_channel': 'camil_channel',
    'smmil': 'smmil_early', 'camil_context': 'camil_context', 'mrblock': MR_DEFAULT,
    'rrt': 'rrt', 'campanella_rnn': 'campanella_rnn', 'scmil': 'scmil',
    'protosurv': 'protosurv_repaired', 'mggmmil': 'mggmmil_uaf_original_graph_patient_reference',
}
NEW_FAMILIES = {'dyko', 'dspamil', 'feather'}
NEW_PROFILES = DYKO_PROFILES + DSPA_PROFILES + FEATHER_PROFILES


def snapshot(function, *arguments):
    try:
        result = function(*arguments)
    except (ValueError, KeyError) as error:
        return {'error_type': type(error).__name__, 'error': str(error)}
    encoded = json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return {'id': result['id'], 'runnable': result.get('runnable'),
            'sha256': hashlib.sha256(encoded).hexdigest()}


def bundle(context):
    return SimpleNamespace(classes=context['classes'], task_type=context['task_type'], config={},
                           promptmil=None, macnet=None, slip=None)


def test_actual_parent_exhibits_exactly_eleven_implemented_label_failures():
    failures = {item['id'] for item in PARENT['families']
                if item['status'] != 'pending_adapter' and not item['inspect_display'].get('runnable')}
    assert failures == set(FIXED)
    assert PARENT['parent_version'] == '0.120.0'
    assert len(PARENT['families']) == 164
    assert len(PARENT['profiles']) == 352


def test_all_164_catalog_names_have_code_execution_routes():
    assert len(registry.methods()) == 164
    runnable = 0
    for item in registry.methods():
        spec = registry.inspect_method(item['name'])
        assert spec['runnable'] is True, item['id']
        expected = (VAL_PROFILE if item['id'] == 'pneamil' else
                    ASMIL_DEFAULT if item['id'] == 'asmil' else
                    ATTRI_DEFAULT if item['id'] == 'attrimil' else
                    CAR_DEFAULT if item['id'] == 'carmil' else
                    MR_DEFAULT if item['id'] == 'mrblock' else
                    NN_DEFAULT if item['id'] == 'nnmil' else
                    FC_DEFAULT if item['id'] == 'fcmil' else item['id'])
        assert registry.require_runnable(item['name']) == registry.require_runnable(expected), item['id']
        runnable += 1
    assert runnable == 164


@pytest.mark.parametrize('family,expected_profile', [
    pytest.param(family, expected, id=f"{family}-{'mrblock' if family == 'mrblock' else expected}")
    for family, expected in FIXED.items()
])
def test_newly_resolved_catalog_label_uses_existing_family_recipe(family, expected_profile):
    item = next(record for record in registry.methods() if record['id'] == family)
    spec = registry.require_runnable(item['name'])
    assert spec['id'] == expected_profile
    if family == 'mrblock':
        original = next(p for p in PARENT['profiles'] if p['id'] == 'mrblock')
        assert snapshot(registry.require_runnable, 'mrblock') == original['spec']
        assert spec == registry.require_runnable(MR_DEFAULT)
    else:
        assert spec == registry.require_runnable(item['id'])


def test_all_352_explicit_execution_profile_specs_are_identical_to_parent():
    old_profiles = [p for p in registry.profiles() if p['family'] not in NEW_FAMILIES]
    assert len([p for p in old_profiles if p['id'] not in
                {a['id'] for a in ASMIL_PROFILES + ATTRI_PROFILES + CAR_PROFILES + MR_PROFILES + NN_PROFILES + FC_PROFILES + FOURIER_PROFILES}]) == 354
    assert len([p for p in old_profiles if p['id'] not in
                {a['id'] for a in ATTRI_PROFILES + CAR_PROFILES + MR_PROFILES + NN_PROFILES + FC_PROFILES + FOURIER_PROFILES}]) == 358
    assert len([p for p in old_profiles if p["id"] not in {n["id"] for n in CAR_PROFILES + MR_PROFILES + NN_PROFILES + FC_PROFILES + FOURIER_PROFILES}]) == 360
    assert len([p for p in old_profiles if p["id"] not in {n["id"] for n in MR_PROFILES + NN_PROFILES + FC_PROFILES + FOURIER_PROFILES}]) == 368
    assert len([p for p in old_profiles if p["id"] not in {n["id"] for n in NN_PROFILES + FC_PROFILES + FOURIER_PROFILES}]) == 388
    assert len([p for p in old_profiles if p["id"] not in {n["id"] for n in FC_PROFILES + FOURIER_PROFILES}]) == 392
    assert len([p for p in old_profiles if p["id"] not in {n["id"] for n in FOURIER_PROFILES}]) == 394
    assert len(old_profiles) == 396 and len(registry.profiles()) == 415
    for item in PARENT['profiles']:
        assert snapshot(registry.require_runnable, item['id']) == item['spec'], item['id']
    old_ids = {item['id'] for item in PARENT['profiles']}
    assert [item for item in registry.profiles() if item['id'] not in old_ids] == PNEA_PROFILES + ASMIL_PROFILES + ATTRI_PROFILES + CAR_PROFILES + MR_PROFILES + NN_PROFILES + FC_PROFILES + FOURIER_PROFILES + NEW_PROFILES


def test_all_canonical_family_inspection_and_execution_contracts_are_unchanged():
    for item in PARENT['families']:
        if item['id'] in NEW_FAMILIES:
            continue  # Their newly integrated code routes are tested independently.
        assert snapshot(registry.inspect_method, item['id']) == item['inspect_id'], item['id']
        assert snapshot(registry.require_runnable, item['id']) == item['require_id'], item['id']
    family = registry.inspect_method('icmil_coupled')
    assert family['kind'] == 'family' and family['runnable'] is False
    assert len(family['execution_profiles']) == 3


def test_all_previously_working_display_names_keep_the_complete_parent_spec():
    for item in PARENT['families']:
        if item['id'] in FIXED or item['id'] in NEW_FAMILIES:
            continue
        if item['id'] == 'fcmil':
            assert registry.inspect_method(item['name']) == registry.inspect_method(FC_DEFAULT)
            assert registry.require_runnable(item['name']) == registry.require_runnable(FC_DEFAULT)
            continue
        if item['id'] == 'nnmil':
            assert registry.inspect_method(item['name']) == registry.inspect_method(NN_DEFAULT)
            assert registry.require_runnable(item['name']) == registry.require_runnable(NN_DEFAULT)
            continue
        if item['id'] == 'pneamil':
            assert registry.inspect_method(item['name']) == registry.inspect_method(VAL_PROFILE)
            assert registry.require_runnable(item['name']) == registry.require_runnable(VAL_PROFILE)
            continue
        if item['id'] == 'asmil':
            assert registry.inspect_method(item['name']) == registry.inspect_method(ASMIL_DEFAULT)
            assert registry.require_runnable(item['name']) == registry.require_runnable(ASMIL_DEFAULT)
            continue
        if item['id'] == 'carmil':
            assert registry.inspect_method(item['name']) == registry.inspect_method(CAR_DEFAULT)
            assert registry.require_runnable(item['name']) == registry.require_runnable(CAR_DEFAULT)
            continue
        if item['id'] == 'attrimil':
            assert registry.inspect_method(item['name']) == registry.inspect_method(ATTRI_DEFAULT)
            assert registry.require_runnable(item['name']) == registry.require_runnable(ATTRI_DEFAULT)
            continue
        assert snapshot(registry.inspect_method, item['name']) == item['inspect_display'], item['name']
        assert snapshot(registry.require_runnable, item['name']) == item['require_display'], item['name']


def test_all_family_and_display_routes_preserve_data_dependent_recipe_choices():
    for context in PARENT['contexts']:
        task = bundle(context)
        for item in PARENT['families']:
            if item['id'] in NEW_FAMILIES:
                continue  # Original pending errors were intentionally replaced by code routes.
            actual = snapshot(registry.resolve_for_bundle, item['id'], task)
            assert actual == item['bundle_id'][context['name']], (context['name'], item['id'])
            display = snapshot(registry.resolve_for_bundle, item['name'], task)
            if item['id'] == 'fcmil':
                assert display == snapshot(registry.require_runnable, FC_DEFAULT)
            elif item['id'] == 'nnmil':
                assert display == snapshot(registry.require_runnable, NN_DEFAULT)
            elif item['id'] == 'pneamil':
                assert display == snapshot(registry.require_runnable, VAL_PROFILE)
            elif item['id'] == 'asmil':
                assert display == snapshot(registry.require_runnable, ASMIL_DEFAULT)
            elif item['id'] == 'carmil':
                assert display == snapshot(registry.require_runnable, CAR_DEFAULT)
            elif item['id'] == 'attrimil':
                assert display == snapshot(registry.require_runnable, ATTRI_DEFAULT)
            elif item['id'] == 'mrblock':
                assert display == snapshot(registry.require_runnable, MR_DEFAULT)
            elif item['id'] in FIXED:
                assert display == actual, (context['name'], item['name'])
            else:
                assert display == item['bundle_display'][context['name']], (context['name'], item['name'])


@pytest.mark.parametrize('family', ['longmil', 'iibmil', 'stablemil', 'smmile', 'attention2majority',
                                    'conmil', 'c2c', 'pamil_prototype'])
def test_descriptive_catalog_names_use_task_aware_canonical_route(monkeypatch, family):
    records = registry.methods()
    label = family+' registered full descriptive method name'
    changed = [{**record, 'name': label} if record['id'] == family else record for record in records]
    monkeypatch.setattr(registry, 'methods', lambda: changed)
    for context in PARENT['contexts']:
        task = bundle(context)
        assert registry.resolve_for_bundle(label, task) == registry.resolve_for_bundle(family, task)


def test_explicit_profile_wins_over_a_catalog_label_collision(monkeypatch):
    records = registry.methods()
    explicit = 'clam_mb'
    changed = [{**record, 'name': explicit} if record['id'] == 'frmil' else record for record in records]
    monkeypatch.setattr(registry, 'methods', lambda: changed)
    assert registry.canonical_catalog_display_name(explicit) == explicit
    assert registry.resolve_for_bundle(explicit, bundle(PARENT['contexts'][0]))['id'] == explicit


def test_ambiguous_display_label_is_rejected(monkeypatch):
    records = registry.methods()
    changed = [{**record, 'name': 'Shared descriptive label'} if record['id'] in {'frmil', 'varmil'}
               else record for record in records]
    monkeypatch.setattr(registry, 'methods', lambda: changed)
    with pytest.raises(ValueError, match='Ambiguous catalog method'):
        registry.resolve_for_bundle('Shared descriptive label', bundle(PARENT['contexts'][0]))


def test_unknown_method_still_fails():
    with pytest.raises(ValueError, match='Unknown/ambiguous method'):
        registry.resolve_for_bundle('no_such_original_MIL', bundle(PARENT['contexts'][0]))


@pytest.mark.parametrize('family', list(FIXED))
def test_public_cli_inspect_accepts_the_actual_catalog_display_label(family, capsys):
    item = next(record for record in registry.methods() if record['id'] == family)
    assert main(['inspect', '--method', item['name']]) == 0
    actual = json.loads(capsys.readouterr().out)
    assert actual['id'] == FIXED[family]
    assert actual['runnable'] is True


@pytest.mark.parametrize('family', ['frmil', 'varmil', 'camil_channel', 'mrblock'])
def test_public_dataset_path_and_full_method_name_reach_the_same_compatibility_checks(family, tmp_path, capsys):
    task = create_demo(tmp_path/'task', patients=20)
    item = next(record for record in registry.methods() if record['id'] == family)
    canonical_method = MR_DEFAULT if family == 'mrblock' else family
    assert main(['check', '--data', str(task), '--method', canonical_method, '--deep']) == 0
    canonical = json.loads(capsys.readouterr().out)
    assert main(['check', '--data', str(task), '--method', item['name'], '--deep']) == 0
    descriptive = json.loads(capsys.readouterr().out)
    assert descriptive == canonical
    if family == 'mrblock':
        assert main(['check', '--data', str(task), '--method', 'mrblock', '--deep']) == 0
        assert json.loads(capsys.readouterr().out) == canonical
