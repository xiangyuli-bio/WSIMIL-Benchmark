"""Verify offline original-code import closure, without loading invented weights."""
import sys
from types import ModuleType

import pytest

from milplus.dyko_assets import OriginalTITANAssets
from milplus.io import digest_file


@pytest.fixture
def asset(tmp_path, monkeypatch):
    root = tmp_path / 'original-TITAN'
    root.mkdir()
    sources = {'model.py': 'from .tower import Tower\n',
               'tower.py': 'from .layers import Layer\nclass Tower: pass\n',
               'layers.py': 'class Layer: pass\n'}
    for name, code in sources.items():
        (root / name).write_text(code)
    # Only exercise the checked source-code cache, not model/weight verification.
    item = object.__new__(OriginalTITANAssets)
    item.root = root
    item.files = {name: {'sha256': digest_file(root / name)} for name in sources}
    module = ModuleType('transformers.dynamic_module_utils')
    module.HF_MODULES_CACHE = str(tmp_path / 'cache')
    destination = tmp_path / 'cache' / 'transformers_modules' / root.name
    module.create_dynamic_module = lambda name: destination.mkdir(parents=True, exist_ok=True)
    package = ModuleType('transformers')
    package.__path__ = []
    monkeypatch.setitem(sys.modules, 'transformers', package)
    monkeypatch.setitem(sys.modules, 'transformers.dynamic_module_utils', module)
    return item, destination


def test_transitive_original_modules_are_copied_exactly_and_reused(asset):
    item, destination = asset
    item._prepare_module_cache()
    item._prepare_module_cache()
    assert {p.name for p in destination.iterdir()} == set(item.files)
    for name in item.files:
        assert (destination / name).read_bytes() == (item.root / name).read_bytes()


def test_changed_original_module_rejects_before_cache_creation(asset):
    item, destination = asset
    (item.root / 'layers.py').write_text('changed code\n')
    with pytest.raises(ValueError, match='Original TITAN module changed'):
        item._prepare_module_cache()
    assert not destination.exists()


def test_unbound_transitive_relative_module_cannot_be_imported(asset):
    item, _destination = asset
    path = item.root / 'tower.py'
    path.write_text('from .unbound import Layer\n')
    item.files[path.name]['sha256'] = digest_file(path)
    with pytest.raises(ValueError, match='Unbound original TITAN relative import'):
        item._prepare_module_cache()


@pytest.mark.parametrize('symlink', [False, True])
def test_conflicting_or_symlinked_cache_never_overwrites_existing_code(asset, symlink):
    item, destination = asset
    destination.mkdir(parents=True)
    path = destination / 'layers.py'
    if symlink:
        path.symlink_to(item.root / 'layers.py')
    else:
        path.write_text('unrelated cached code\n')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='cached module'):
        item._prepare_module_cache()
    assert path.read_bytes() == before
