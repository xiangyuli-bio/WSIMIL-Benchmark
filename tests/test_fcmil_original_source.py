"""Original FC-MIL architecture, intervention gradients and explicit repair scope."""

import inspect

import pytest
import torch

from milplus.fcmil_source import contract, definitions
from milplus.fcmil_spec import PROFILES
from fcmil_helpers import equal_tree, literal_sources


@pytest.mark.parametrize("patches", [7, 8, 17, 31])
@pytest.mark.parametrize("alpha", [.1, 0.])
def test_exact_native_model_causal_loss_gradients_Adam_and_RNG(patches, alpha):
    author, source = literal_sources(), definitions()
    torch.manual_seed(32)
    expected = author.models.FCMIL(n_classes=2)
    torch.manual_seed(32)
    actual = source.models.FCMIL(n_classes=2)
    equal_tree(actual.state_dict(), expected.state_dict())
    features = torch.randn(1, patches, 1024, generator=torch.Generator().manual_seed(57))
    label = torch.tensor([1])
    optimizers = [torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
                  for model in (actual, expected)]
    values = []
    for model, loss_function, optimizer in zip(
            (actual, expected), (source.loss.causal_mil_loss, author.loss.causal_mil_loss), optimizers):
        torch.manual_seed(43)
        logits, attention = model(features)
        cr = loss_function(model, logits, features, attention, topk_ratio=.03, lam=.5)
        ce = torch.nn.functional.cross_entropy(logits, label)
        (ce + alpha * cr).backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        values.append((logits, attention, ce, cr, gradients, torch.get_rng_state()))
    equal_tree(values[0], values[1])
    equal_tree(actual.state_dict(), expected.state_dict())
    equal_tree(optimizers[0].state_dict(), optimizers[1].state_dict())


def test_original_loss_detaches_soft_target_and_preserves_negative_drop_positive_replace():
    source = definitions()
    torch.manual_seed(67)
    model = source.models.FCMIL(n_classes=2).eval()
    features = torch.randn(1, 17, 1024)
    y_orig = torch.tensor([[.4, -.3]], requires_grad=True)
    attention = torch.arange(17, dtype=torch.float32).unsqueeze(0)
    loss = source.loss.causal_mil_loss(model, y_orig, features, attention, .03, .5)
    assert torch.autograd.grad(loss, y_orig, allow_unused=True, retain_graph=True)[0] is None
    changed = features.clone()
    changed[:, 16, :] = 0
    replaced = features.clone()
    replaced[:, 0, :] = features[:, 16, :]
    target = torch.softmax(y_orig, dim=1).detach()
    expected = -.5 * torch.nn.functional.kl_div(
        torch.log_softmax(model(changed)[0], 1), target, reduction="batchmean")
    expected += .5 * torch.nn.functional.kl_div(
        torch.log_softmax(model(replaced)[0], 1), target, reduction="batchmean")
    assert torch.equal(loss, expected)


def test_native_writer_mismatch_and_missing_auc_are_preserved_as_failure_evidence():
    author = literal_sources()
    assert "writer" not in inspect.signature(author.trainer._run_epoch).parameters
    with pytest.raises(TypeError, match="writer"):
        author.trainer._run_epoch(writer=None)
    source = contract()
    assert source["original_writer_keyword_mismatch"] and source["original_epoch_auc_missing"]
    assert source["original_dataset_logger_initialization_order_failure"]
    from types import SimpleNamespace
    with pytest.raises(AttributeError, match="logger"):
        author.datasets.MILDataset(
            [("slide", "negative", "not_read.pt")],
            SimpleNamespace(label_list=["negative", "positive"], mapping={}, data_root="."))
    assert source["original_configuration_files"] == []
    assert source["original_training_configuration_recovered"] is False
    assert source["license"].startswith("Apache-2.0")
    assert len(PROFILES) == 2
    for profile in PROFILES:
        assert profile["benchmark_recipe"]["epochs"] == 100
        assert profile["author_training_configuration_recovered"] is False
