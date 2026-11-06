from milplus.fcmil_spec import PROFILES as FCMIL_PROFILES
"""Native nnMIL model, deterministic input, TRAIN-only planning and source contract."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from milplus.data import Bundle
from milplus.nnmil_data import SlideBags, train_only_plan
from milplus.nnmil_source import definitions
from milplus.nnmil_spec import DEFAULT_PROFILE, PROFILES
from milplus.registry import profiles, require_runnable
from nnmil_helpers import make_task, original_package


@pytest.mark.parametrize("dimension", [256, 768, 1536, 2048])
def test_literal_model_initialization_masked_TRAIN_RNG_and_ddof1_chunk_outputs(tmp_path, dimension):
    original_package()
    from nnMIL.network_architecture.model_factory import create_mil_model
    actual = definitions(lambda *args, **kwargs: None)
    torch.manual_seed(42)
    reference = create_mil_model("nnmil", dimension, max(256, dimension // 4), 2, dropout=.25)
    torch.manual_seed(42)
    model = actual.models.create_mil_model("nnmil", dimension, max(256, dimension // 4), 2, dropout=.25)
    for key, value in reference.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    features = torch.randn(2, 13, dimension, generator=torch.Generator().manual_seed(55))
    mask = torch.ones(2, 13, dtype=torch.bool)
    mask[1, -4:] = False
    for training in (True, False):
        model.train(training)
        reference.train(training)
        torch.manual_seed(63)
        a = model(features, valid_mask=mask, return_WSI_attn=True)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(63)
        b = reference(features, valid_mask=mask, return_WSI_attn=True)
        assert torch.equal(rng, torch.get_rng_state())
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True)
        if not training and dimension == 256:
            assert model.eval_chunks == 1 and torch.isnan(a["var_logits"]).all()


@pytest.mark.parametrize("metric", ["auc", "bacc", "f1", "kappa"])
def test_native_plan_and_dataset_padding_crop_match_independent_original(tmp_path, metric):
    original_package()
    from nnMIL.data.dataset import UnifiedMILDataset
    import h5py
    import pandas as pd
    task = make_task(tmp_path / "task")
    bundle = Bundle(task)
    plan = train_only_plan(bundle, metric, definitions(lambda *args, **kwargs: None), 2026)
    cfg = plan["training_configuration"]
    h5 = tmp_path / "h5"
    h5.mkdir()
    rows = bundle.records("train")
    for row in rows:
        values, _ = bundle.load(row)
        with h5py.File(h5 / (row["slide_id"] + ".h5"), "x") as handle:
            handle.create_dataset("features", data=values.numpy())
            handle.create_dataset("coords", data=np.zeros((len(values), 2), "f4"))
    csv = tmp_path / "native.csv"
    pd.DataFrame([{"slide_id": row["slide_id"], "patient_id": row["patient_id"],
                   "label": int(row["target"])} for row in rows]).to_csv(csv, index=False)
    reference = UnifiedMILDataset(str(csv), str(h5), task_type="classification",
                                 dataset_name="synthetic_nnmil", split="train",
                                 max_seq_length=cfg["max_seq_length"], use_original_length=False,
                                 skip_feature_validation=True, seed=42)
    actual = SlideBags(bundle, "train", cfg)
    for i in range(len(actual)):
        a, b = actual[i], reference[i]
        for j in range(4):
            assert torch.equal(a[j], b[j])
        assert a[4:] == b[4:]
    assert cfg["num_epochs"] == 100 and cfg["batch_size"] == 16
    assert cfg["hidden_dim"] == 256 and cfg["patience"] == 10
    original_plan = copy.deepcopy(plan)
    for row in bundle.records("test"):
        path = task / row["path"]
        values = np.load(path)
        np.save(path, np.concatenate([values] * 20))
    assert train_only_plan(Bundle(task), metric, definitions(lambda *args, **kwargs: None), 2026) == original_plan


def test_native_catalog_keeps_controlled_legacy_and_four_explicit_selectors():
    assert len(PROFILES) == 4 and len([p for p in profiles() if p["id"] not in {n["id"] for n in FCMIL_PROFILES}]) == 392
    assert len(profiles()) == 394
    assert require_runnable("nnMIL")["id"] == DEFAULT_PROFILE
    assert require_runnable("nnmil")["id"] == "nnmil"
    for profile in PROFILES:
        assert profile["native_recipe"]["epochs"] == 100
        assert profile["native_recipe"]["workers"] == 4
