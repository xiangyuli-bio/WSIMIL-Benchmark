"""Verify public config dispatch before invoking costly method trainers."""

import json

import pytest
import yaml

from milplus import cli
from milplus.akdpmp_runtime import desired_policy
from milplus.config_io import load_training_config


@pytest.mark.parametrize("suffix", [".json", ".JSON", ".yaml", ".yml"])
def test_public_run_passes_numeric_exponents_and_nested_values_to_trainer(
    tmp_path, monkeypatch, suffix
):
    from milplus import engine

    expected = {
        "epochs": 100, "lr": 2e-4, "weight_decay": 1e-5,
        "max_instances": None, "model": {"dropout": .1},
    }
    path = tmp_path / ("training" + suffix)
    path.write_text(
        json.dumps(expected) if suffix.lower() == ".json" else yaml.safe_dump(expected)
    )
    received = {}

    def train(*args, **kwargs):
        received["config"] = args[7]
        return tmp_path / "run"

    monkeypatch.setattr(engine, "train", train)
    cli.main([
        "run", "--data", str(tmp_path), "--method", "linmil",
        "--config", str(path), "--output", str(tmp_path / "run"),
    ])
    assert received["config"] == expected
    assert type(received["config"]["weight_decay"]) is float


def test_invalid_public_configuration_is_rejected_before_training(tmp_path, monkeypatch, capsys):
    from milplus import engine

    def unexpected(*args, **kwargs):
        raise AssertionError("trainer must not execute")

    monkeypatch.setattr(engine, "train", unexpected)
    path = tmp_path / "config.json"
    path.write_text("null")
    with pytest.raises(SystemExit) as error:
        cli.main([
            "run", "--data", str(tmp_path), "--method", "linmil",
            "--config", str(path), "--output", str(tmp_path / "run"),
        ])
    assert error.value.code == 2
    assert "object/mapping" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_AKD_startup_uses_identical_JSON_configuration(tmp_path):
    from argparse import Namespace

    path = tmp_path / "config.json"
    path.write_text('{"threads": 2, "weight_decay": 1e-05}')
    args = Namespace(command="run", device="cpu", method="akdpmp", config=str(path))
    policy = desired_policy(args)
    assert policy["MKL_NUM_THREADS"] == str(load_training_config(path)["threads"])
    assert type(load_training_config(path)["weight_decay"]) is float


def test_DSPA_subprocess_driver_uses_same_numeric_JSON_loader(tmp_path, monkeypatch):
    import sys
    from milplus import dspamil_driver

    path = tmp_path / "config.json"
    path.write_text('{"epochs": 50, "weight_decay": 1e-05}')
    received = {}

    def train(*args, **kwargs):
        received["config"] = kwargs["config"]
        return tmp_path / "run"

    monkeypatch.setattr(dspamil_driver, "train", train)
    monkeypatch.setattr(sys, "argv", [
        "dspamil_driver", "train", "--data", str(tmp_path),
        "--method", next(iter(dspamil_driver.PROFILES)),
        "--config", str(path), "--output", str(tmp_path / "run"),
    ])
    dspamil_driver.main()
    assert received["config"] == {"epochs": 50, "weight_decay": 1e-5}
    assert type(received["config"]["weight_decay"]) is float
