"""Full original KMP components with actual CONCH and source update oracle."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset

from milplus import kmpmil_source as source
from milplus.io import write_json


def same(a,b,path='state'):
    if torch.is_tensor(a):assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'.'+str(i))
    else:assert a==b,(path,a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);checks=[];started=time.monotonic()
    try:
        from milplus.engine import seed_everything
        seed_everything(2026);torch.set_num_threads(2)
        if not torch.cuda.is_available() or not a.device.startswith('cuda'):raise ValueError('Actual full-width component gate requires an allocated CUDA device')
        base=source.load_base(a.weights,a.sha256,a.device);d=source.definitions();cfg=source.config();cfg['task_num']=1
        assert base.text.width==768 and base.text.heads==12 and len(base.text.transformer.resblocks)==12
        assert sum(p.numel() for p in base.text.parameters())==110129664
        tc=nn.ParameterList([nn.Parameter(torch.zeros(n,512,device=a.device),requires_grad=False) for n in cfg['dataset_subtype_num']])
        d.utilities.activate_current_tc_residual(tc,1)
        texts,descriptors=source.source_text_assets();assert [len(descriptors[n]) for n in cfg['dataset_names']]==[10]*4
        def create(task,past):
            current=cfg['dataset_names'][task];cfg['task_num']=task+1;d.utilities.activate_current_tc_residual(tc,task+1)
            pools=source.task_pools(cfg,base,len(descriptors[current]))
            return d.model.KMPMIL(cfg,base,torch.device(a.device),pools[0],pools[1],tc,source.class_prompts(current),
                pfc_gamma_delta_pool=pools[2],pfc_beta_pool=pools[3],past_keys=past,current_descriptors=descriptors[current]).eval()
        model=create(0,[])
        tokens=source.conch_definitions().tokenizer.tokenize(source.conch_definitions().tokenizer.get_tokenizer(),descriptors['bracs']).to(a.device)
        with torch.no_grad():expected=torch.nn.functional.normalize(base.encode_text(tokens),dim=-1).to(base.dtype)
        assert torch.equal(torch.cat(list(model.key)),expected)
        assert model.use_pfc and len(model.key)==10 and len(model.prompt)==10 and model.prompt[0].shape==(1,24,768)
        checks.append('full_original_CONCH_and_actual_semantic_descriptor_key_initialization')
        clone=copy.deepcopy(model)
        def opt(m):return torch.optim.Adam(m.parameters(),lr=cfg['adam_lr'],eps=cfg['adam_eps'],weight_decay=cfg['adam_weight_decay'])
        optimizer=opt(model);reference_optimizer=opt(clone)
        bags=[torch.randn(1,17+i%3,512,device=a.device) for i in range(16)]
        labels=[torch.tensor([[i%3]],device=a.device) for i in range(16)]
        logits,indices,losses=source.original_update(model,optimizer,cfg,[x.clone() for x in bags],labels)
        raw_context=SimpleNamespace(model=clone,optimizer=reference_optimizer,cfg=cfg,loss_function=nn.CrossEntropyLoss())
        reference,reference_indices=d.training.TrainingMixin._update_network(raw_context,16,[x.clone() for x in bags],labels)
        same(logits,reference);same(indices,reference_indices);same(model.state_dict(),clone.state_dict());same(optimizer.state_dict(),reference_optimizer.state_dict())
        for (name,p),(other,q) in zip(model.named_parameters(),clone.named_parameters()):
            assert name==other;same(p.grad,q.grad,name)
        assert losses['routing_loss']==0 and losses['pfc_reg_loss']==0
        assert all(p.grad is None for p in base.parameters())
        checks.append('literal_source_update_every_parameter_gradient_Adam_moment_and_zero_gradient_rule_exact')
        _,_,second_losses=source.original_update(model,optimizer,cfg,[x.clone() for x in bags],labels)
        assert second_losses['pfc_reg_loss']>0
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.pfc.gamma_delta_pool)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.pfc.beta_pool)
        old_keys=torch.cat([p.detach().cpu() for p in model.key]);old_prompts=torch.cat([p.detach().cpu() for p in model.prompt])
        old_gamma=torch.cat([p.detach().cpu() for p in model.pfc.gamma_delta_pool]);old_beta=torch.cat([p.detach().cpu() for p in model.pfc.beta_pool])
        current=create(1,[old_keys]);optimizer2=opt(current)
        labels2=[torch.tensor([[3+i%2]],device=a.device) for i in range(16)]
        _,_,losses2=source.original_update(current,optimizer2,cfg,[x.clone() for x in bags],labels2)
        assert losses2['routing_loss']>0 and tc[0].grad is None and not tc[0].requires_grad and tc[1].requires_grad
        assert not any(k.requires_grad for k in current.past_keys)
        checks.append('nonzero_PFC_regularization_and_second_task_routing_current_TC_only_no_old_feature_replay')
        banks=[old_keys,torch.cat([p.detach().cpu() for p in current.key])]
        prompts=[old_prompts,torch.cat([p.detach().cpu() for p in current.prompt])]
        gammas=[old_gamma,torch.cat([p.detach().cpu() for p in current.pfc.gamma_delta_pool])]
        betas=[old_beta,torch.cat([p.detach().cpu() for p in current.pfc.beta_pool])]
        universal=d.builders.build_universal_kmp_mil(cfg,base,torch.device(a.device),base.dtype,banks,prompts,tc,source.class_prompts('tcga_lung'),gammas,betas)
        assert len(universal.key)==20 and len(universal.prompt)==20 and not any(p.requires_grad for p in universal.key)
        assert torch.equal(torch.cat(list(universal.key)).cpu(),torch.cat(banks))
        with torch.no_grad():pred,_=universal([bags[0]],eval=True)
        assert pred.shape==(1,5) and torch.isfinite(pred).all()
        checks.append('actual_growing_universal_selected_memory_bank_and_all_seen_class_inference')
        # Literal epoch, including a seventeenth-bag tail update. No mock model.
        class Bags(Dataset):
            def __len__(self):return 17
            def __getitem__(self,i):return torch.tensor([i],dtype=torch.int),bags[i%16][0].detach().cpu(),torch.tensor([i%2])
        count=[];context=SimpleNamespace(model=current,optimizer=optimizer2,cfg=cfg,device=torch.device(a.device),loss_function=nn.CrossEntropyLoss(),current_dataset='tcga_lung',data_loader={'tcga_lung':{'train':DataLoader(Bags(),batch_size=1,shuffle=False)}})
        def update(i,xs,ys):
            logits,indices,record=d.observed_update(context,i,xs,ys);count.append(len(xs));return logits,indices
        context._update_network=update
        epoch=d.training.TrainingMixin._train_each_epoch(context)
        assert count==[16,1] and len(epoch['pred']['y'])==17
        checks.append('literal_complete_epoch_flushes_actual16_plus1_tail_with_full_model')
        write_json(out/'validation.json',{'state':'passed_components_only','checks':checks,'source_commit':source.COMMIT,
            'runtime':source.runtime_identity(),'actual_original_model_pfc_tc_and_routing':True,'actual_descriptors_per_task':[10]*4,
            'pretrained_text_parameters':110129664,'prototype_prompt_length':24,'matching_keys':5,'tail_updates':count,
            'elapsed_seconds':time.monotonic()-started,'synthetic':True,'full_continual_trainer_passed':False,
            'included_in_method_count':False,'formal_admission':False})
    except BaseException as error:
        write_json(out/'validation.json',{'state':'failed_components_only','checks':checks,'error':repr(error),'included_in_method_count':False,'formal_admission':False});raise


if __name__=='__main__':main()
