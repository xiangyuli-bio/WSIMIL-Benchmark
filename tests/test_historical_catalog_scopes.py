"""Keep every historic assertion's exact scope and retain the expanded catalog."""
import json
from pathlib import Path

from milplus.registry import methods, profiles


def test_every_historical_profile_snapshot_is_retained_with_every_field_exact():
    current = {p['id']: p for p in profiles()}
    fixtures = Path(__file__).parent / 'fixtures'
    names = ['comel_prior307_profiles.json', 'akdpmp_prior309_profiles.json',
        'qpmilvl_prior313_profiles.json', 'kmpmil_prior315_profiles.json',
        'protomil_prior317_profiles.json', 'macnet_prior321_profiles.json',
        'dspamil_prior327_profiles.json', 'mggmmil_prior327_profiles.json',
        'hybrid_prior328_profiles.json']
    for name in names:
        old = json.loads((fixtures / name).read_text())
        assert len({row['id'] for row in old}) == len(old), name
        for row in old:
            assert current[row['id']] == row, (name, row['id'])


def test_all352_frozen_current_execution_contracts_retained_without_budget_or_input_changes():
    current = {p['id']: p for p in profiles()}
    frozen = json.loads((Path(__file__).parent / 'fixtures/current352_execution_contracts.json').read_text())
    assert len(frozen) == 352 and set(frozen) <= current.keys()
    for name, contract in frozen.items():
        for field, value in contract.items():
            assert current[name][field] == value, (name, field)
    assert len(methods()) == 164
    assert len({row['id'] for row in methods()}) == 164
