"""Preserve every parent profile while adding explicitly named FC experiments."""

import copy
import json
from pathlib import Path
import subprocess
import sys

from milplus.fcmil_spec import DEFAULT_PROFILE, PROFILES
from milplus.fouriermil_spec import PROFILES as FOURIER_PROFILES
from milplus.registry import methods, profiles, require_runnable


def test_all392_parent_profile_dictionaries_and164_entries_survive_regeneration():
    parent = json.loads(
        (Path(__file__).parent / "fixtures/fcmil_parent127_profiles392.json").read_text())
    before = copy.deepcopy(profiles())
    assert len(parent) == 392 and len(before) == 415
    assert before[:392] == parent
    assert before[392:394] == PROFILES
    assert before[394:396] == FOURIER_PROFILES
    assert len({p["id"] for p in before[:394]}) == 394
    assert len({p["id"] for p in before}) == 415
    script = Path(__file__).resolve().parents[1] / "scripts/build_catalog.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert profiles() == before
    assert len(methods()) == 164
    assert sum(bool(m["profiles"]) for m in methods()) == 164
    assert {m["id"] for m in methods() if not m["profiles"]} == set()


def test_display_selects_original_mechanism_and_literal_controlled_port_keeps_identity():
    assert require_runnable("FC-MIL")["id"] == DEFAULT_PROFILE
    assert require_runnable("fcmil")["id"] == "fcmil"
    entry = next(m for m in methods() if m["id"] == "fcmil")
    assert entry["source_commit"] == "315ac88808d7517429ca0b55073e75dce03ddfd9"
    assert entry["license"] == "Apache-2.0_original_fixed_tree_read"
    assert entry["status"] == "adapted_under_verification"
    assert set(entry["profiles"]) == {"fcmil", *[p["id"] for p in PROFILES]}
    assert len(PROFILES) == 2
    for profile in PROFILES:
        assert profile["author_training_configuration_recovered"] is False
        assert profile["benchmark_recipe"]["epochs"] == 100
