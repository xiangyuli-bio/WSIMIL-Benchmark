"""Inspect full native released components without calling SAT a full DSPA."""
import torch
from torch import nn
import pytest
from milplus import dspamil_source as source


def test_full_native151million_source_forward_backward_update_and_actual_missing_trainer_interface():
    torch.set_num_threads(2);torch.manual_seed(26)
    model=source.build_native_source_component()
    assert sum(p.numel() for p in model.parameters())==151141388
    assert len(model.transformer)==12 and all(b.attn.num_heads==8 for b in model.transformer)
    assert model.virtual_token_feature.shape==(1,3,1024)
    x=torch.randn(1,17,1024);output=model(x)
    assert output['fused_logits'].shape==(1,6)
    for i in [1,2,3]:
        assert output[f'val{i}'].shape==(1,2) and output[f'inp_cls_token_{i}'].shape==(1,1024)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
    # Explicit differentiability audit, not an invented full DSPA recipe.
    objective=nn.functional.cross_entropy(output['fused_logits'],torch.tensor([1]))
    objective+=.3*sum(nn.functional.cross_entropy(output[f'val{i}'],torch.tensor([int(i==1)])) for i in [1,2,3])/3
    optimizer.zero_grad();objective.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    before=model.virtual_token_feature.detach().clone();optimizer.step()
    assert not torch.equal(before,model.virtual_token_feature) and optimizer.state
    # Literal released loop fails before a loss/update: it unpacks a dict as
    # four outputs. Do not silently repair this and claim the missing method.
    batch=(x,torch.tensor([1]),torch.tensor([1]),torch.tensor([0]),torch.tensor([0]))
    with pytest.raises(ValueError,match='too many values to unpack'):
        source.original_training_loop()(torch.device('cpu'),model,[batch],nn.CrossEntropyLoss(),optimizer,None)


def test_original_native_attention_zeroes_logits_then_softmax_leaves_cross_token_probability():
    attention=source.author_definitions().Attention(dim=1024,num_heads=8,attn_drop=0.,proj_drop=0.)
    x=torch.zeros(1,20,1024);_,weights=attention(x)
    mask=attention.gen_mask(weights)
    assert torch.equal(mask[:,:,0,1],torch.zeros(1,8))
    assert torch.equal(weights[:,:,0,1],torch.full((1,8),1/20))
    assert torch.count_nonzero(weights[:,:,0,1])==8


def test_source_review_distinguishes_incomplete_author_trainer_from_integrated_reference():
    from milplus.registry import profiles
    review=source.source_review()
    assert review['commit']==source.COMMIT and not review['required_GS_dataset_released']
    assert 'WSI_Dataset' in review['declared_dataset_classes']
    assert len(review['missing_full_method_mechanisms'])==6
    assert not review['literal_author_trainer_runnable'] and review['runnable_profile_registered']
    assert len([p for p in profiles() if p['family']=='dspamil']) == 2


def test_catalog_preserves_all327_profiles_and164_families_and_reviewed_DSPA_metadata():
    import json,subprocess,sys
    from pathlib import Path
    from milplus.registry import profiles,methods
    root=Path(__file__).resolve().parents[1]
    prior=json.loads((root/'tests/fixtures/dspamil_prior327_profiles.json').read_text())
    # Freeze this admission's exact historical scope, while later families
    # retain their independently checked execution profiles in the full catalog.
    prior_ids={p['id'] for p in prior}
    current=[p for p in profiles() if p['id'] in prior_ids];assert len(prior)==len(current)==327
    assert {p['id']:p for p in prior}=={p['id']:p for p in current}
    assert len({p['family'] for p in current})==152 and len(methods())==164
    before=(root/'src/milplus/resources/methods.json').read_bytes()
    p=subprocess.run([sys.executable,str(root/'scripts/build_catalog.py')],cwd=root,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert before==(root/'src/milplus/resources/methods.json').read_bytes()
    row=next(r for r in json.loads(before) if r['id']=='dspamil')
    assert row['source_commit']==source.COMMIT and len(row['profiles'])==2
    assert row['status']=='adapted_under_verification' and row['DSPA_U_implemented'] is False
