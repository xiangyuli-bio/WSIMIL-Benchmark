"""Full original architecture components; no fake FEATHER pretrained weights."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from milplus.feather_source import COMMIT, architecture_classes, checked_source, original_configuration


@pytest.mark.parametrize('encoder,dimension',[('conch_v15',768),('uni',1024),('uni_v2',1536)])
def test_original_full_native_forward_loss_attention_and_all_gradients(encoder,dimension):
    torch.set_num_threads(1);torch.manual_seed(20261027)
    model = architecture_classes().ABMIL(**original_configuration(encoder,3)).eval()
    x = torch.randn(2,257,dimension,requires_grad=True)
    mask = torch.ones(2,257);mask[:,-17:] = 0
    label = torch.tensor([0,2])
    result,logs = model(x,loss_fn=nn.CrossEntropyLoss(),label=label,attn_mask=mask,
                      return_attention=True,return_slide_feats=True)
    # Independent literal equations from the native512/gated384 graph, with
    # no reduced dimensions, substituted aggregation or pretrained claim.
    hidden = F.relu(F.linear(x,model.patch_embed[0].weight,model.patch_embed[0].bias))
    branch_a = torch.tanh(F.linear(hidden,model.global_attn.attention_a[0].weight,
                                 model.global_attn.attention_a[0].bias))
    branch_b = torch.sigmoid(F.linear(hidden,model.global_attn.attention_b[0].weight,
                                    model.global_attn.attention_b[0].bias))
    score = F.linear(branch_a*branch_b,model.global_attn.attention_c.weight,
                     model.global_attn.attention_c.bias).transpose(-2,-1)
    score = score+(1-mask).unsqueeze(1)*torch.finfo(score.dtype).min
    bag = torch.bmm(score.softmax(-1),hidden).squeeze(1)
    logits = F.linear(bag,model.classifier.weight,model.classifier.bias)
    expected_loss = F.cross_entropy(logits,label)
    assert torch.equal(result['logits'],logits) and torch.equal(result['loss'],expected_loss)
    assert torch.equal(logs['attention'],score) and torch.equal(logs['slide_feats'],bag)
    assert model.patch_embed[0].weight.shape == (512,dimension)
    assert model.global_attn.attention_a[0].weight.shape == (384,512)
    assert model.global_attn.attention_b[0].weight.shape == (384,512)
    assert len(model.patch_embed) == 2 and not any(isinstance(part,nn.Dropout) for part in model.patch_embed)
    variables = (x,*model.parameters())
    actual_grad = torch.autograd.grad(result['loss'],variables,retain_graph=True)
    expected_grad = torch.autograd.grad(expected_loss,variables)
    for one,two in zip(actual_grad,expected_grad):
        assert torch.equal(one,two) and torch.isfinite(one).all()
    assert actual_grad[0][:,-17:].abs().sum() == 0


@pytest.mark.parametrize('encoder',['conch_v15','uni','uni_v2'])
def test_actual_classifier_reset_preserves_every_backbone_tensor(encoder):
    torch.manual_seed(451)
    model = architecture_classes().ABMIL(**original_configuration(encoder,2))
    before = copy.deepcopy(model.state_dict())
    model.initialize_classifier(num_classes=5)
    assert model.classifier.weight.shape == (5,512) and torch.count_nonzero(model.classifier.bias) == 0
    for key,value in before.items():
        if not key.startswith('classifier.'):
            assert torch.equal(value,model.state_dict()[key])


def test_original_source_revision_and_nonpretrained_status_remain_explicit():
    root,revision = checked_source()
    assert revision == COMMIT and (root/'LICENSE').is_file()
    from milplus.registry import inspect_method
    method = inspect_method('FEATHER / MIL-Lab models')
    assert method['runnable'] is True and method['family'] == 'feather'
    assert method['formal_admission'] is False and method['paper_reproduction'] == 'not_run'


@pytest.mark.parametrize('encoder',['ctranspath','virchow2','conch_v1'])
def test_other_encoders_cannot_be_presented_as_original_pretrained_FEATHER(encoder):
    with pytest.raises(ValueError,match='original FEATHER24K'):
        original_configuration(encoder,3)
