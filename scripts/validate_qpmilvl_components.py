"""Real pretrained full text-tower, original16-bag update; component evidence only."""
import argparse
from collections import Counter
from pathlib import Path
import time

import torch
from milplus import qpmilvl_source as source
from milplus.io import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();checks=[]
    try:
        from milplus.engine import seed_everything
        seed_everything(2026);torch.set_num_threads(2)
        if not torch.cuda.is_available() or not a.device.startswith('cuda'):raise ValueError('Full native component probe requires an allocated GPU')
        base=source.load_base(a.weights,a.sha256,a.device);d=source.definitions();cfg=source.config()
        assert len(base.text.transformer.resblocks)==12 and base.text.width==768 and base.text.heads==12
        assert type(base.text.ln_final) is torch.nn.LayerNorm
        assert len(base.visual.trunk.blocks)==12 and base.text_decoder is not None
        assert base.text.context_length==128 and base.text.text_projection.shape==(768,512)
        assert not any(p.requires_grad for p in base.parameters())
        texts=['a histopathology image of lung adenocarcinoma.','a histopathology image of lung squamous cell carcinoma.']
        tokens=d.tokenizer.tokenize(d.tokenizer.get_tokenizer(),texts)[:,:-1].to(a.device)
        with torch.no_grad():
            literal=base.text(tokens)[0]
            port=d.model.CONCHPromptEncoder(base)(base.text.token_embedding(tokens),tokens)
        assert torch.equal(literal,port),'Full original causal+CLS prompt path differs from original CONCH text forward'
        checks.append('full_pretrained12layer768hidden12head128position_text_tower_literal_prompt_identity')
        cfg['task_num']=1;pools=source.parameter_pools(cfg,base);ensemble=source.ensemble_through('tcga_lung')
        assert ensemble['count']==[66,66] and len(ensemble['ensemble_classes'])==132
        before={name:p.detach().clone() for prefix,parameters in zip(['key','prompt','vector'],pools) for name,p in [(prefix+'.'+str(i),p) for i,p in enumerate(parameters)]}
        frequency={name:[] for name in cfg['dataset_names']}
        model=source.model(cfg,base,pools,ensemble,frequency).eval()
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['adam_lr'],eps=cfg['adam_eps'],weight_decay=cfg['adam_weight_decay'])
        bags=[torch.randn(1,17+i%3,512,device=a.device) for i in range(16)]
        labels=[torch.tensor([[i%2]],device=a.device) for i in range(16)]
        result,losses=source.original_update(model,optimizer,cfg,bags,labels)
        logits,indices,features=result
        assert logits.shape==(16,2) and indices.shape==(16,5) and features['image'].shape==(16,512)
        assert all(torch.equal(indices[0],row) for row in indices)
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
        matched=set(indices[0].tolist());assert len(matched)==5
        for i,(key,prompt) in enumerate(zip(pools[0],pools[1])):
            if i in matched:
                assert key.grad is not None and prompt.grad is not None
                assert key.grad.abs().sum()>0 and prompt.grad.abs().sum()>0
                assert not torch.equal(key,before['key.'+str(i)]) and not torch.equal(prompt,before['prompt.'+str(i)])
            else:
                assert key.grad is None and prompt.grad is None
                assert torch.equal(key,before['key.'+str(i)]) and torch.equal(prompt,before['prompt.'+str(i)])
        assert pools[2][0].grad is not None and pools[2][0].grad.abs().sum()>0
        assert all(p.grad is None for p in base.parameters())
        assert all(p.grad is None and not p.requires_grad for p in pools[2][1:])
        checks.append('literal_original16bag_update_CE_matching_class_similarity_clip_and_zero_gradient_exclusion')
        counts=torch.zeros(20,dtype=torch.int)
        for idx in indices.detach().cpu().flatten():counts[idx]+=1
        assert counts.sum()==80;frequency[cfg['dataset_names'][0]].append(counts)
        cfg['task_num']=2
        next_model=source.model(cfg,base,pools,source.ensemble_through('tcga_brca'),frequency).eval()
        assert torch.equal(next_model.penalty_table,(counts/counts.sum()).to(device=a.device,dtype=base.dtype))
        assert not pools[2][0].requires_grad and pools[2][0].grad is None and pools[2][1].requires_grad
        train_indices,match_loss=next_model._query_prototype_pool(bags,16,False)
        eval_indices,no_loss=next_model._query_prototype_pool(bags,16,True)
        assert train_indices.shape==(16,5) and eval_indices.shape==(16,5) and no_loss is None and torch.isfinite(match_loss)
        checks.append('second_task_actual_frequency_penalty_current_vector_only_and_unpenalized_evaluation')
        write_json(out/'validation.json',{'state':'passed_components_only','source_commit':source.COMMIT,
            'checks':checks,'frozen_pretrained_text_parameters':sum(p.numel() for p in base.text.parameters()),
            'full_original_CoCa_constructor_and_timm1011_RNG_order_preserved':True,
            'full_actual_training_minibatch':16,'pool_size':20,'prompt_length':24,'matched_prototypes':5,
            'real_pretrained_weights_sha256':a.sha256,'actual_original_loss_log':losses,
            'elapsed_seconds':time.monotonic()-start,'synthetic_bags':True,'full_continual_recipe_complete':False,
            'included_in_method_count':False,'formal_admission':False})
    except BaseException as e:
        write_json(out/'validation.json',{'state':'failed_components_only','checks':checks,'error':repr(e),
            'included_in_method_count':False,'formal_admission':False});raise


if __name__=='__main__':main()
