"""Concrete public-configuration failure and accepted input format boundaries."""

import json

import pytest
import yaml

from milplus.config_io import load_training_config as load


def test_JSON_exponent_stays_numeric_where_current_YAML_only_CLI_fails(tmp_path):
    path = tmp_path / "training.json"
    path.write_text('{"epochs": 100, "lr": 2e-4, "weight_decay": 1e-05}')
    assert isinstance(yaml.safe_load(path.read_text())["weight_decay"], str)
    actual = load(path)
    assert actual == {"epochs": 100, "lr": .0002, "weight_decay": .00001}
    assert type(actual["lr"]) is float and type(actual["weight_decay"]) is float


@pytest.mark.parametrize("suffix", [".json", ".JSON", ".yaml", ".yml"])
def test_equivalent_JSON_and_YAML_preserve_nested_types_and_nulls(tmp_path, suffix):
    expected = {
        "epochs": 100, "lr": .0002, "weight_decay": .00001,
        "max_instances": None, "model": {"dropout": .1, "enabled": True},
    }
    path = tmp_path / ("training" + suffix)
    path.write_text(json.dumps(expected) if suffix.lower() == ".json" else yaml.safe_dump(expected))
    assert load(path) == expected


@pytest.mark.parametrize("text", ["[]", "2", '"config"', "null", "true"])
def test_JSON_non_mapping_configuration_fails_before_training(tmp_path, text):
    path = tmp_path / "training.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="object/mapping"):
        load(path)


def test_invalid_JSON_is_not_silently_reinterpreted_as_YAML(tmp_path):
    path = tmp_path / "training.json"
    path.write_text("epochs: 100\n")
    with pytest.raises(json.JSONDecodeError):
        load(path)


def test_empty_YAML_and_absent_config_keep_existing_default_behavior(tmp_path):
    path = tmp_path / "training.yaml"
    path.write_text("")
    assert load(path) == {}
    assert load(None) == {}
