"""The executable control fixture keeps its full native dimensions and clear scope."""
from milplus.data import Bundle
from milplus.dspamil_demo import create_demo
from milplus.io import read_json


def test_full1025_instance_fixture_is_explicitly_synthetic_and_does_not_admit_a_method(tmp_path):
    root = create_demo(tmp_path / 'data')
    bundle = Bundle(root)
    assert bundle.dim == 1024 and len(bundle.rows) == 18
    assert bundle.config['synthetic'] is True
    for row in bundle.rows.to_dict('records'):
        assert bundle.dspamil.load_features(row).shape == (1025, 1024)
    proof = read_json(root / 'synthetic_control_provenance.json')
    assert proof['actual_patches_per_bag'] == 1025
    assert proof['original_expert_assets_verified'] is proof['real_WSI_experiment'] is False
    assert proof['formal_admission'] is False
