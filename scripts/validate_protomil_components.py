"""Full native SAE/concept-MIL source probe; no complete method admission."""
from pathlib import Path
import argparse,time,copy,math,json
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from milplus import protomil_source as source
from milplus.io import write_json


def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(2026);d=source.definitions();s=d.sae;checks=[]
    args=d.parser().parse_args([])
    assert args.expansion_factor==4 and args.train_sae_bs==4096 and args.num_epochs==200 and args.resample_freq==500000
    assert args.lr==1e-4 and args.l1_coeff==3e-4
    ae=source.autoencoder();reference=source.autoencoder();reference.load_state_dict(ae.state_dict(),strict=True)
    assert len(ae.decoder._weight._backward_hooks or {})==len(reference.decoder._weight._backward_hooks or {})==1
    assert sum(p.numel() for p in ae.parameters())==2099712
    x=torch.nn.functional.normalize(torch.randn(4097,1,512),dim=-1)
    store=s.TensorActivationStore(4097,512,1);store.extend(x[:,0],component_idx=0)
    loss=source.loss();optimizer=source.optimizer(ae)
    pipeline=d.Pipeline(activation_resampler=None,autoencoder=ae,loss=loss,optimizer=optimizer,device='cpu',args=SimpleNamespace())
    before=torch.get_rng_state();firing=pipeline.train_autoencoder(store,train_batch_size=4096)
    assert firing.shape==(1,2048) and pipeline.total_activations_trained_on==8192
    torch.set_rng_state(before);ref_optimizer=source.optimizer(reference);ref_loss=source.loss();batch_sizes=[]
    for batch in DataLoader(store,batch_size=4096,shuffle=True):
        batch_sizes.append(len(batch));ref_optimizer.zero_grad();concepts,recon=reference(batch)
        objective,_=ref_loss.scalar_loss_with_log(batch,concepts,recon,component_reduction=s.LossReductionType.MEAN)
        objective.backward();ref_optimizer.step();reference.post_backwards_hook()
    assert batch_sizes==[4096,1];same(ae.state_dict(),reference.state_dict());same(optimizer.state_dict(),ref_optimizer.state_dict())
    for p,q in zip(ae.parameters(),reference.parameters()):same(p.grad,q.grad)
    checks.append('literal_source_full512_to2048_SAE_4096_plus1_updates_every_weight_gradient_Adam_moment_exact')
    decoder_norms=ae.decoder.weight.norm(dim=-2);assert torch.allclose(decoder_norms,torch.ones_like(decoder_norms),atol=2e-6,rtol=0)
    checks.append('actual_tied_bias_sparse_encoder_unit_norm_decoder_and_source_post_backwards_hook')
    learned,recon=ae(x[:19]);assert learned.shape==(19,1,2048) and torch.all(learned>=0)
    mil=source.model(classes=2);opt=torch.optim.Adam(mil.parameters(),lr=1e-4);target=torch.tensor([1]);features=learned[:,0].detach().unsqueeze(0)
    pred=mil(features);ce=torch.nn.functional.cross_entropy(pred['bag_logits'],target)
    penalty=torch.nn.functional.l1_loss(mil.classifier.weight.flatten(),torch.zeros_like(mil.classifier.weight.flatten()))
    objective=ce+penalty;objective.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mil.parameters());opt.step()
    pred=mil(features);contributions=pred['attn']*features[0]
    expected=torch.nn.functional.linear(contributions.sum(0,keepdim=True),mil.classifier.weight,mil.classifier.bias)
    assert torch.equal(pred['bag_logits'],expected)
    checks.append('full_original2048concept_MIL_gated_attention_CE_plus_mean_absolute_classifier_weight_and_exact_linear_explanation')
    artifact=out/'actual_full_state.pt';torch.save({'sae':ae.state_dict(),'optimizer':optimizer.state_dict(),'MIL':mil.state_dict(),'MIL_optimizer':opt.state_dict(),'rng':torch.get_rng_state()},artifact)
    loaded=torch.load(artifact,weights_only=False);same(ae.state_dict(),loaded['sae']);same(optimizer.state_dict(),loaded['optimizer'])
    checks.append('actual_full_component_checkpoint_serialization_no_admission_claim')
    write_json(out/'validation.json',{'state':'passed_components_only','checks':checks,'source_commit':source.COMMIT,'reviewed_DTN_dependency_commit':source.DTN_COMMIT,
        'SAE_parameters':sum(p.numel() for p in ae.parameters()),'actual_native_SAE_batch_sizes':batch_sizes,'native_SAE_input_dimension':512,'native_concepts':2048,
        'source_max_SAE_epochs':200,'paper_SAE_epochs':60,'full_training_pipeline_complete':False,'included_in_method_count':False,'formal_admission':False,'synthetic':True,'elapsed_seconds':time.monotonic()-start})

if __name__=='__main__':main()
