"""Read training options without converting JSON exponents into YAML strings."""

import json
from pathlib import Path

import yaml


def load_training_config(path):
    if path is None:
        return {}
    file = Path(path)
    text = file.read_text()
    value = json.loads(text) if file.suffix.lower() == ".json" else yaml.safe_load(text)
    if value is None and file.suffix.lower() != ".json":
        return {}
    if not isinstance(value, dict):
        raise ValueError("Training configuration must contain an object/mapping")
    return value
