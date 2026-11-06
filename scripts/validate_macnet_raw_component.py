"""Full original pretrained raw Swin/GFK/head/CE+distance two-update oracle."""
import torch,random,copy,json,time,argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from milplus.macnet_source import FullMacNet,definitions,original_base,SWIN22K_SHA256
from milplus.engine import seed_everything,rng_state,restore_rng
from milplus.io import write_json


def same(a,b,path='state'):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu()),(path,float((a.cpu()-b.cpu()).abs().max()) if a.shape==b.shape else 'shape')
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
    else:assert a==b,(path,a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    torch.set_num_threads(2);seed_everything(0);d=definitions();base=original_base(a.weights,2);feature=d.MIL_Parallel_Feature(base_model=base,img_size=[96,96]).cuda();head=d.MIL_Parallel_Head_ub(base_model=base,class_num=2,batch_size=1,bags_len=2000,model_stats='train',seed=0).cuda()
    original_parameters=list(feature.parameters())+list(head.parameters());original_opt=torch.optim.RMSprop(original_parameters,lr=1e-5)
    seed_everything(0);adapter=FullMacNet(a.weights,classes=2,condition='public',adaptive='source_cached').cuda();adapt_opt=torch.optim.RMSprop(adapter.parameters(),lr=1e-5)
    same(feature.state_dict(),adapter.feature.state_dict(),'full_original_pretrained_Swin');same(head.state_dict(),adapter.head.state_dict(),'full_original_public_head');same(head.tgi_clustering_block.inst_adapt_w,adapter.initial_adaptive_diagonal,'actual_unregistered_original_cached_matrix')
    generator=torch.Generator().manual_seed(30);images=torch.rand(37,3,96,96,generator=generator).to('cuda')*2-1;label=torch.tensor([1],device='cuda');steps=[]
    for step in [1,2]:
        before=rng_state();feature.train();head.train();original_opt.zero_grad();reference=head(feature(images));loss=torch.nn.functional.cross_entropy(reference[0],label)+reference[1]-reference[2];loss.backward();original_opt.step();after=rng_state()
        restore_rng(before);adapter.train();adapt_opt.zero_grad();actual=adapter(images);actual_loss=torch.nn.functional.cross_entropy(actual[0],label)+actual[1]-actual[2];actual_loss.backward();adapt_opt.step()
        same(reference,actual,'actual_full_raw_GFK_proxy_logits_distances');same(loss,actual_loss,'source_CE_plus_min_minus_max');same(feature.state_dict(),adapter.feature.state_dict(),'all_learned_original_Swin_weights');same(head.state_dict(),adapter.head.state_dict(),'all_learned_head_and_adaptive_parameters')
        for x,y in zip(original_parameters,adapter.parameters()):same(x.grad,y.grad,'every_original_encoder_head_gradient')
        same(original_opt.state_dict(),adapt_opt.state_dict(),'all_original_RMSprop_moments');same(after,rng_state(),'whole_RNG')
        steps.append({'step':step,'state':'passed','loss':float(actual_loss.detach()),'native_feature_dimension':768,'raw_patch_count':37,'guide_count':5,'all_Swin_head_weights_gradients_RMSprop_RNG_exact':True})
    # A new actual complete constructor with original init and checked state,
    # retaining the cached matrix buffer, must recover the trained predictor.
    adapter.eval();seed_everything(0);expected=adapter(images)
    state=copy.deepcopy(adapter.state_dict());seed_everything(917);loaded=FullMacNet(a.weights,classes=2,condition='public',adaptive='source_cached').cuda();loaded.load_state_dict(state,strict=True);loaded.eval();seed_everything(0);same(expected,loaded(images),'fresh_full_raw_selected_predictor')
    write_json(out/'validation.json',{'state':'passed_full_original_raw_two_update_oracle','conditions':steps,'full_original_Swin_parameter_count':sum(p.numel() for p in feature.parameters()),'actual_pretrained22K_weight_sha256':SWIN22K_SHA256,'actual_full_GFK_adaptive_clusters_and_proxy_aggregation':True,'fresh_constructor_cached_state_recovery_exact':True,'scope':'Full raw component and two-step source comparison; not full method training recipe/native100epoch admission','synthetic':True,'included_in_method_count':False,'elapsed_seconds':time.monotonic()-start})

if __name__=='__main__':main()
