"""First-use code preparation, independent of cached author repositories."""
import ast
import json
import subprocess
from pathlib import Path

import pytest

from milplus import registry
from milplus import setup_sources as setup
from milplus.cli import main


def cli_json(capsys, arguments):
    assert main(arguments) == 0
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("method", [record["id"] for record in registry.methods()])
def test_every164_family_has_a_pure_complete_pinned_preparation_plan(method, monkeypatch, tmp_path):
    source_root = tmp_path / "not-created"
    monkeypatch.setenv("MILPLUS_SOURCE_ROOT", str(source_root))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Planning must not execute a child"))
    plan = setup.plan_setup(method)
    assert plan["family"] == method
    assert len(plan["sources"]) == len({entry["id"] for entry in plan["sources"]})
    for entry in plan["sources"]:
        assert Path(entry["path"]) == source_root / entry["id"]
        assert len(entry["commit"]) == 40
        assert set(entry["dependencies"]).issubset({item["id"] for item in plan["sources"]})
    assert not source_root.exists()


@pytest.mark.parametrize("profile", [record["id"] for record in registry.profiles()])
def test_every352_execution_profile_can_prepare_its_actual_source_without_loading_models(profile, monkeypatch, tmp_path):
    monkeypatch.setenv("MILPLUS_SOURCE_ROOT", str(tmp_path / "not-created"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Planning must not execute a child"))
    expected = next(record for record in registry.profiles() if record["id"] == profile)
    plan = setup.plan_setup(profile)
    assert plan["method_id"] == profile
    assert plan["family"] == expected["family"]
    assert plan["profile_available"] is True
    if expected["external_source"]:
        assert plan["source"] == expected["external_source"]
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("name,expected", [
    ("ProtoMIL", ["protomil", "protomil_dtn"]),
    ("KMP-MIL", ["kmpmil", "conch_kmp"]),
    ("RMDL", ["rmdl", "keras_contrib_legacy"]),
    ("AHDMIL", ["ahdmil", "hdmil"]),
    ("GAT-Mamba", ["gatmamba", "gatmamba_ssm", "lifelines_metric"]),
])
def test_actual_setup_fetches_complete_required_author_code(name, expected, monkeypatch, tmp_path, capsys):
    seen = []
    monkeypatch.setenv("MILPLUS_SOURCE_ROOT", str(tmp_path))
    monkeypatch.setattr("milplus.execution.fetch_source", lambda source, path: seen.append(source) or path)
    result = cli_json(capsys, ["setup", "--method", name])
    assert seen == expected
    assert list(result["prepared_paths"]) == expected
    assert result["state"] == "source_preparation_completed"
    assert result["training_or_gated_weight_download_performed"] is False


@pytest.mark.parametrize("method", ["MeanMIL", "MaxMIL"])
def test_packaged_pooling_setup_does_not_clone_a_nonexistent_repository(method, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MILPLUS_SOURCE_ROOT", str(tmp_path / "not-created"))
    monkeypatch.setattr("milplus.execution.fetch_source", lambda *a: pytest.fail("Packaged pooling needs no Git source"))
    result = cli_json(capsys, ["setup", "--method", method])
    assert result["packaged_code_only"] is True
    assert result["source"] is None and result["path"] is None
    assert result["prepared_paths"] == {}
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("method", ["DyKo", "FEATHER", "DSPA-MIL"])
def test_integrated_profiles_keep_original_asset_and_validation_requirements_in_plan(method, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MILPLUS_SOURCE_ROOT", str(tmp_path / "not-created"))
    result = cli_json(capsys, ["setup", "--method", method, "--plan"])
    assert result["method_status"] == "adapted_under_verification"
    assert result["profile_available"] is True
    assert result["admission_blocker"] is None
    assert result["prerequisites"] and result["validation_blockers"]
    if method == 'DSPA-MIL':
        assert result['unsupported_variants'] == ['DSPA-U-MIL']
    assert result["public_asset_actions"] == []
    assert result["state"] == "plan_only"
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("method,module", [
    ("C2C", "milplus.c2c_assets"), ("center_embeddings", "milplus.center_embeddings_assets"),
])
def test_plan_lists_legacy_public_asset_action_without_downloading(method, module, monkeypatch, tmp_path, capsys):
    import importlib
    assets = importlib.import_module(module)
    monkeypatch.setattr(assets, "fetch", lambda: pytest.fail("Plan must not download public weights"))
    monkeypatch.setenv("MILPLUS_SOURCE_ROOT", str(tmp_path / "not-created"))
    result = cli_json(capsys, ["setup", "--method", method, "--plan"])
    assert len(result["public_asset_actions"]) == 1
    assert result["network_or_filesystem_writes_performed"] is False
    assert not (tmp_path / "not-created").exists()


def test_missing_dependency_pin_fails_before_fetching_primary(monkeypatch, tmp_path):
    lock = registry.resource("sources.lock.json")
    del lock["protomil_dtn"]
    monkeypatch.setattr(setup, "resource", lambda name: lock)
    monkeypatch.setenv("MILPLUS_SOURCE_ROOT", str(tmp_path / "not-created"))
    monkeypatch.setattr("milplus.execution.fetch_source", lambda *a: pytest.fail("Validate all pins before first fetch"))
    with pytest.raises(ValueError, match="protomil_dtn"):
        setup.prepare_sources("ProtoMIL")
    assert not (tmp_path / "not-created").exists()


def test_recursive_shared_dependency_is_fetched_once_and_cycles_are_rejected():
    lock = {name: {"url": "https://example.invalid/" + name, "commit": "a" * 40} for name in ["a", "b", "c", "d"]}
    assert setup.source_closure("a", lock, {"a": ["b", "c"], "b": ["d"], "c": ["d"]}) == ["a", "b", "d", "c"]
    with pytest.raises(ValueError, match="cycle"):
        setup.source_closure("a", lock, {"a": ["b"], "b": ["a"]})
    lock["a"]["commit"] = "main"
    with pytest.raises(ValueError, match="full fixed Git commit"):
        setup.source_closure("a", lock)


def test_unknown_method_is_rejected_before_fetch(monkeypatch):
    monkeypatch.setattr("milplus.execution.fetch_source", lambda *a: pytest.fail("Unknown method must not fetch"))
    with pytest.raises((ValueError, KeyError)):
        setup.prepare_sources("not-a-registered-MIL-method")


def test_preparation_plan_covers_literal_dependencies_of_current_private_author_loaders():
    package = Path(setup.__file__).parent
    lock = registry.resource("sources.lock.json")
    owner_keys = sorted(lock, key=len, reverse=True)
    loader_names = {"verified_source", "load_definitions", "load_single", "definitions_from", "root", "source_root",
                    "_root", "module", "_module", "load_module", "load_class", "_load"}
    methods = {item["id"] for item in registry.methods()}
    checked = set()
    for file in package.iterdir():
        if file.suffix != ".py" or file.name in {"upstream.py", "setup_sources.py"}:
            continue
        owner = next((key for key in owner_keys if file.stem == key or file.stem.startswith(key + "_")), None)
        if owner not in methods:
            continue
        referenced = set()
        for node in ast.walk(ast.parse(file.read_text())):
            if not isinstance(node, ast.Call) or not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            function = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if function in loader_names and node.args[0].value in lock:
                referenced.add(node.args[0].value)
        if referenced:
            planned = {entry["id"] for entry in setup.plan_setup(owner)["sources"]}
            assert referenced.issubset(planned), (file.name, sorted(referenced - planned))
            checked.add(owner)
    # This guards the omissions that motivated the fix; it does not assert that
    # arbitrary dynamic Python imports can be exhaustively inferred by AST.
    assert {"protomil", "kmpmil", "rmdl", "akdpmp", "sparseconvmil"}.issubset(checked)


def test_actual_local_Git_fetch_prepares_both_pinned_trees_and_repeats_cleanly(tmp_path, monkeypatch, capsys):
    lock = {}
    for source in ["protomil", "protomil_dtn"]:
        origin = tmp_path / (source + "-origin")
        origin.mkdir()
        subprocess.run(["git", "init", "--quiet", str(origin)], check=True)
        (origin / "original.txt").write_text("explicit local setup mechanism fixture: " + source)
        subprocess.run(["git", "add", "original.txt"], cwd=origin, check=True)
        subprocess.run(["git", "-c", "user.name=Setup fixture", "-c", "user.email=fixture@example.invalid",
                        "commit", "--quiet", "-m", "Original setup fixture"], cwd=origin, check=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
        lock[source] = {"url": str(origin), "commit": commit}
    original_resource = registry.resource
    monkeypatch.setattr(registry, "resource", lambda name: lock if name == "sources.lock.json" else original_resource(name))
    monkeypatch.setattr(setup, "resource", lambda name: lock if name == "sources.lock.json" else original_resource(name))
    destination = tmp_path / "prepared"
    monkeypatch.setenv("MILPLUS_SOURCE_ROOT", str(destination))
    plan = cli_json(capsys, ["setup", "--method", "ProtoMIL", "--plan"])
    assert not destination.exists()
    result = cli_json(capsys, ["setup", "--method", "ProtoMIL"])
    assert list(result["prepared_paths"]) == ["protomil", "protomil_dtn"]
    for entry in plan["sources"]:
        checkout = Path(entry["path"])
        assert (checkout / "original.txt").read_text().endswith(entry["id"])
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() == entry["commit"]
        assert subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True) == ""
    assert cli_json(capsys, ["setup", "--method", "ProtoMIL"])["prepared_paths"] == result["prepared_paths"]
