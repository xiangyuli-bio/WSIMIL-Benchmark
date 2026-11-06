from milplus.fouriermil_spec import PROFILES as FOURIER_PROFILES
from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
from milplus.nnmil_spec import PROFILES as NNMIL_PROFILES
"""Every public profile exposes the same complete execution metadata schema."""
from copy import deepcopy

import pytest
from milplus.mrblock_spec import PROFILES as MRBLOCK_PROFILES

from milplus.catalog_contracts import native_profile
from milplus.registry import profiles


def test_all352_profiles_define_required_execution_fields_without_missing_defaults():
    rows = profiles()
    historical = rows[:396]
    from milplus.carmil_spec import PROFILES as carmil_profiles
    from milplus.asmil_spec import PROFILES as new_profiles
    from milplus.attrimil_spec import PROFILES as newest_profiles
    assert len([p for p in historical if p['id'] not in {n['id'] for n in new_profiles + newest_profiles + carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES + FOURIER_PROFILES}]) == 354
    assert len([p for p in historical if p['id'] not in {n['id'] for n in newest_profiles + carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES + FOURIER_PROFILES}]) == 358
    assert len([p for p in historical if p["id"] not in {n["id"] for n in carmil_profiles + MRBLOCK_PROFILES + NNMIL_PROFILES + FCMIL_PROFILES + FOURIER_PROFILES}]) == 360
    assert len([p for p in historical if p["id"] not in {n["id"] for n in NNMIL_PROFILES + FCMIL_PROFILES + FOURIER_PROFILES}]) == 388
    assert len([p for p in historical if p["id"] not in {n["id"] for n in FCMIL_PROFILES + FOURIER_PROFILES}]) == 392
    assert len([p for p in historical if p["id"] not in {n["id"] for n in FOURIER_PROFILES}]) == 394
    assert len(historical) == 396 and len(rows) == 415
    for row in rows:
        for field in ['family', 'recipe', 'tasks', 'required_inputs', 'class_count', 'external_source']:
            assert field in row, (row['id'], field)
        # Existing families also expose reviewed structured recipe mappings.
        # Preserve those original contracts instead of coercing them to text.
        assert isinstance(row['recipe'], (str, dict)) and bool(row['recipe']), row['id']
        if isinstance(row['recipe'], str):
            assert row['recipe'].strip(), row['id']
        assert row['class_count'] is None or type(row['class_count']) is int and row['class_count'] >= 2, row['id']


def test_native_description_preserves_full_recipe_and_existing_metadata_values():
    source = {'id': 'explicit_reference', 'family': 'test', 'tasks': ['classification'],
        'required_inputs': ['complete_bags'], 'external_source': 'test',
        'native_recipe': {'epochs': 1000, 'instances': 10000, 'auxiliary_loss': 1.}}
    original = deepcopy(source)
    value = native_profile(source, class_count=3)
    assert source == original
    for key in source:
        assert value[key] == source[key]
    assert value['class_count'] == 3 and '1000' in value['recipe']
    assert '10000' in value['recipe'] and 'auxiliary_loss' in value['recipe']
    value['native_recipe']['epochs'] = 1
    assert source == original


def test_native_class_metadata_cannot_override_existing_fixed_output_contract():
    source = {'id': 'explicit_reference', 'family': 'test', 'tasks': ['classification'],
        'required_inputs': ['complete_bags'], 'external_source': 'test',
        'native_recipe': {'epochs': 1000}, 'class_count': 2}
    with pytest.raises(ValueError, match='override'):
        native_profile(source, class_count=3)
