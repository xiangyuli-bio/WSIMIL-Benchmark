"""Full native text/fast+slow optimizer state, actual disk round-trip recovery."""
import copy
import gc

import pytest
import torch

from milplus.convl import ConVLMIL
from milplus.engine import restore_rng, rng_state, seed_everything
from test_convl_training import (assert_core_equal, assert_optimizer_equal,
    assert_rng_equal, config, full_native_bundle)


def test_complete_native_state_checkpoint_resume_retains_all_fast_slow_and_rng(tmp_path):
    torch.set_num_threads(1); seed_everything(2024)
    bundle = full_native_bundle(tmp_path/'data')
    actual = ConVLMIL(bundle, 'convl_yaml_fp32_reference'); actual.configure_training(config())
    opt = actual.make_optimizer(.0002, .00001)
    actual.train(); actual.start_epoch(0); actual.epoch(bundle, opt, 'cpu')
    actual.predict_subset(bundle, 'val', 'cpu', config())
    checkpoint = tmp_path/'complete.pt'
    torch.save({'model': actual.state_dict(), 'optimizer': opt.state_dict(), 'rng': rng_state()}, checkpoint)
    assert checkpoint.stat().st_size > 10**9  # Real full text and both optimizer states, no compact dummy checkpoint.
    actual.train(); actual.start_epoch(1); actual.epoch(bundle, opt, 'cpu')
    expected_val = actual.predict_subset(bundle, 'val', 'cpu', config())
    expected_rng = rng_state()
    restored = ConVLMIL(bundle, 'convl_yaml_fp32_reference'); restored.configure_training(config())
    restored_opt = restored.make_optimizer(.0002, .00001)
    # mmap only changes I/O allocation; every original saved tensor is loaded.
    state = torch.load(checkpoint, mmap=True, map_location='cpu', weights_only=False)
    restored.load_state_dict(state['model']); restored_opt.load_state_dict(state['optimizer'])
    restored.validate_optimizer_state(); restore_rng(state['rng'])
    restored.train(); restored.start_epoch(1); restored.epoch(bundle, restored_opt, 'cpu')
    actual_val = restored.predict_subset(bundle, 'val', 'cpu', config())
    assert expected_val.equals(actual_val)
    assert_core_equal(actual.core, restored.core); assert_optimizer_equal(opt, restored_opt)
    assert_rng_equal(expected_rng, rng_state()); assert actual.get_extra_state() == restored.get_extra_state()
    assert restored.optimizer_updates == 4 and restored.completed_epochs == 2
    baseline = restored.get_extra_state()
    corruptions = [lambda s: s['train_orders'][0].append(s['train_orders'][0][0]),
        lambda s: s['update_traces'][0][-1].update(normalization=1),
        lambda s: s.update(optimizer_updates=s['optimizer_updates']+1),
        lambda s: s.update(scaler={'scale': 1.})]
    for mutate in corruptions:
        bad = copy.deepcopy(baseline); mutate(bad)
        with pytest.raises(ValueError, match='ConVL'):
            restored.set_extra_state(bad)
        assert restored.get_extra_state() == baseline
    del state, actual, restored, opt, restored_opt
    gc.collect()
