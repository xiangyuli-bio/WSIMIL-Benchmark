import json
import subprocess
import sys

from milplus.c2c_demo import create_c2c_demo
from milplus.data import Bundle
from milplus.registry import resolve_for_bundle


def cli(*arguments):
    return subprocess.run([sys.executable, '-m', 'milplus', *map(str, arguments)],
                          text=True, capture_output=True, timeout=180)


def test_public_alias_reports_full_cluster_route_instead_of_topk_ablation():
    result = cli('inspect', '--method', 'C2C')
    assert result.returncode == 0, result.stdout+result.stderr
    value = json.loads(result.stdout)
    assert value['id'] == 'c2c_cluster_source' and value['runnable'] is True
    assert 'FAISS8group' in value['recipe']


def test_public_synthetic_rgb_mode_and_explicit_multiclass_resolution(tmp_path):
    result = cli('demo', '--c2c', '--output', tmp_path/'data', '--patients', 15, '--classes', 3)
    assert result.returncode == 0, result.stdout+result.stderr
    bundle = Bundle(tmp_path/'data')
    assert bundle.config['synthetic'] is True and bundle.dim == 3 and bundle.c2c is not None
    assert resolve_for_bundle('C2C', bundle)['id'] == 'c2c_cluster_multiclass_repaired'


def test_prepare_cli_round_trips_complete_rgb_and_original_assets(tmp_path):
    root = create_c2c_demo(tmp_path/'data', patients=10, patches=8)
    result = cli('prepare-c2c', '--data', root, '--schema', root/'c2c_schema.json', '--output', root/'new.json')
    assert result.returncode == 0, result.stdout+result.stderr
    assert json.loads((root/'new.json').read_text()) == json.loads((root/'c2c/manifest.json').read_text())


def test_raw_and_frozen_or_other_input_modes_cannot_be_mixed(tmp_path):
    result = cli('demo', '--c2c', '--sammil', '--output', tmp_path/'invalid')
    assert result.returncode != 0 and not (tmp_path/'invalid').exists()
