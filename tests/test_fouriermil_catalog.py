"""Append original-source recipes without changing any existing profile."""

import json
from pathlib import Path
import subprocess
import sys

from milplus.fouriermil_spec import PROFILES, SOURCE_COMMIT
from milplus.registry import methods, profiles, require_runnable, resource
from milplus.setup_sources import plan_setup


def test_all394_parent_profiles164_families_and181_source_pins_preserved():
    fixtures = Path(__file__).parent / "fixtures"
    previous = json.loads((fixtures / "fouriermil_parent129_profiles394.json").read_text())
    previous_sources = json.loads((fixtures / "fouriermil_parent129_sources181.json").read_text())
    current = profiles()
    assert len(previous) == 394 and current[:394] == previous
    assert current[394:396] == PROFILES and len(current) == 415
    sources = resource("sources.lock.json")
    assert len(previous_sources) == 181
    assert all(sources[k] == v for k, v in previous_sources.items())
    assert sources["fouriermil_author"]["commit"] == SOURCE_COMMIT
    assert len(methods()) == 164 and sum(bool(m["profiles"]) for m in methods()) == 164
    old_profile = next(p for p in previous if p["id"] == "fouriermil")
    assert old_profile["external_source"] is None  # vendored third-party port
    assert require_runnable("fouriermil")["external_source"] is None
    family = next(m for m in methods() if m["id"] == "fouriermil")
    assert require_runnable(family["name"]) == require_runnable("fouriermil")
    for profile in PROFILES:
        assert require_runnable(profile["id"])["id"] == profile["id"]
        plan = plan_setup(profile["id"])
        assert any(item.get("id", item.get("source")) == "fouriermil_author"
                   or item.get("commit") == SOURCE_COMMIT
                   for item in plan["sources"])


def test_catalog_rebuild_is_byte_reproducible():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "src/milplus/resources").glob("*.json")) + list((root / "method_specs").glob("*.json"))
    previous = {p: p.read_bytes() for p in paths}
    result = subprocess.run([sys.executable, str(root / "scripts/build_catalog.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {p: p.read_bytes() for p in paths} == previous
