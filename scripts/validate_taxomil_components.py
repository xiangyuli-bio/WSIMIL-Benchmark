"""Full pretrained TaxoMIL versus literal released data/model/train/VAL oracles."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from milplus.data import Bundle
from milplus.engine import seed_everything,rng_state,restore_rng
from milplus.io import write_json
from milplus.taxomil import Bags,TaxoMIL
from milplus.taxomil_demo import create_taxomil_demo
from milplus.taxomil_source import author_components,COMMIT
from milplus.upstream import load_definitions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    report={'state':'running','source_commit':COMMIT,'formal_admission':False,'checks':[]}
    def check(name):
        report['checks'].append(name);write_json(out/'validation.json',report);print('PASS',name,flush=True)
    data=create_taxomil_demo(out/'data',args.assets,patches=257)
    bundle=Bundle(data);cfg={'epochs':100,'lr':1e-5,'weight_decay':.01,'patience':20,
        'max_instances':None,'grad_clip':None,'threads':2,'selection_metric':'taxomil_score'}
    seed_everything(8707);adapter=TaxoMIL(bundle);adapter.configure_training(cfg)
    assert adapter.network.decoder.shared_trunk.config.n_layer==12
    assert adapter.network.decoder.shared_trunk.config.n_embd==768
    assert adapter.network.decoder.coarse_unembedding.out_features==50259
    assert len(adapter.network.decoder.shared_trunk.h)==12
    assert len(adapter.network.tokenizer)==50259
    report['parameters']=sum(p.numel() for p in adapter.parameters())
    assert report['parameters']>200_000_000
    check('real_full_pretrained_GPT2_12layers_768D_dualheads_50259tokens')
    from safetensors import safe_open
    with safe_open(str(Path(args.assets)/'model.safetensors'),framework='pt',device='cpu') as f:
        weight=f.get_tensor('wte.weight')
        torch.testing.assert_close(adapter.network.decoder.shared_trunk.wte.weight[:50257],weight,rtol=0,atol=0)
    check('actual_original_pretrained_WTE_retained_byte_exact')
    decoder,losses,scheduler_cls,utils,collate,trainer=author_components(repair_alignment_keys=False)
    seed_everything(8707)
    original=decoder.build_model(model_path=args.assets,device='cpu',dataset='BRACS',emb_dim=1024,
        gated=True,output_len_coarse=5,output_len_fine=15,all_coarse_labels=bundle.hierarchy.coarse,all_fine_labels=bundle.classes)
    utils.patch_unembedding_to_tokenizer(original,original.tokenizer)
    assert adapter.network.state_dict().keys()==original.state_dict().keys()
    for name,value in original.state_dict().items():torch.testing.assert_close(adapter.network.state_dict()[name],value,rtol=0,atol=0)
    check('every_initialized_parameter_matches_literal_author_build_and_required_vocab_repair')
    source_ds=load_definitions('taxomil','dataloader/dataset.py',['UNI_PathDataset','_infer_patch_count'],
        {'torch':torch,'np':np,'pd':pd,'Dataset':torch.utils.data.Dataset,'os':__import__('os'),'tqdm':lambda it,**kw:it}).UNI_PathDataset
    rows=[]
    for row in bundle.rows.to_dict('records'):
        rows.append({'split':row['subset'],'coarse_label':row['coarse_label'],'fine_label':row['label'],
                     'feature_path':str(bundle.root/row['path'])})
    csv=out/'original.csv';pd.DataFrame(rows).to_csv(csv,index=False)
    literal={subset:source_ds('BRACS',str(csv),subset,original.tokenizer,verify_files=False) for subset in ['train','val','test']}
    for subset in literal:
        port=Bags(adapter,subset)
        for i in range(len(port)):
            for a,b in zip(port[i],literal[subset][i]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        a=collate([port[i] for i in range(4)]);b=collate([literal[subset][i] for i in range(4)])
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    check('all_full257_variable_bags_original_EOS_PAD_targets_and_unmasked_collator')
    # Original released criterion/optimizer definitions; all source auxiliary
    # branches are active, and each fine label occurs twice in this TRAIN batch.
    original_criteria={'gen':torch.nn.CrossEntropyLoss(ignore_index=original.tokenizer.pad_token_id),
        'hier':losses.TextHierarchicalLoss(bundle.classes,bundle.hierarchy.coarse,bundle.config['hierarchy']['fine_to_coarse'],tau=1.,margin=1.5),
        'c_ital':losses.AlignmentLoss(.07),'f_ital':losses.AlignmentLoss(.07),
        'c_cl':losses.ContrastiveLoss(.07),'f_cl':losses.ContrastiveLoss(.07)}
    original_criteria['coarse_ital']=original_criteria['c_ital']
    original_criteria['fine_ital']=original_criteria['f_ital']
    original_optimizer=utils.get_optimizer('adamw',original.parameters(),1e-5)
    original_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(original_optimizer,T_max=100,eta_min=1e-7)
    optimizer=adapter.make_optimizer(1e-5,.01)
    original_bank=utils.build_text_bank_for_losses(original,original.tokenizer,torch.device('cpu'),bundle.hierarchy.coarse,bundle.classes)
    # The source pre-TRAIN bank has no RNG operations; both DataLoader iterator
    # base seed and shuffled sampler consume the exact same parent stream.
    seed_everything(9911);before=rng_state()
    trainer.train_one_epoch(1,original,original_criteria,original_optimizer,
        DataLoader(literal['train'],batch_size=64,shuffle=True,collate_fn=collate,drop_last=False,num_workers=4),
        torch.device('cpu'),original.tokenizer,adapter.source_args(),adapter.label_maps,
        scheduler_cls(hier_max=.3,ital_max=.3,cl_max=.1,n_cycles=3).get_weights(0),bundle.hierarchy.coarse,bundle.classes)
    after_original=rng_state()
    restore_rng(before);adapter.start_epoch(0);values=adapter.epoch(bundle,optimizer,torch.device('cpu'))
    after_port=rng_state()
    assert after_original['python']==after_port['python']
    torch.testing.assert_close(after_original['torch'],after_port['torch'],rtol=0,atol=0)
    for name,param in original.named_parameters():
        port=dict(adapter.network.named_parameters())[name]
        torch.testing.assert_close(port,param,rtol=1e-6,atol=2e-6,msg=lambda msg:'parameter '+name+': '+msg)
        assert (port.grad is None)==(param.grad is None),name
        if param.grad is not None:torch.testing.assert_close(port.grad,param.grad,rtol=1e-6,atol=2e-6,msg=lambda msg:'gradient '+name+': '+msg)
        a=optimizer.state.get(port);b=original_optimizer.state.get(param)
        assert (a is None)==(b is None)
        if a:
            assert set(a)==set(b)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=1e-6,atol=2e-6)
    assert values[0]['hier']>0 and values[0]['ital']>0 and values[0]['cl']>0
    assert adapter.optimizer_updates==1
    check('literal_original_full_epoch_all_objectives_every_gradient_AdamW_moment_and_parent_RNG')
    # Compare source generation, CE, exact-match selector and LR ordering.
    before=rng_state()
    ref=trainer.validate(1,original,original_criteria,
        DataLoader(literal['val'],batch_size=64,shuffle=False,collate_fn=collate,drop_last=False,num_workers=4),
        torch.device('cpu'),original.tokenizer,adapter.source_args(),adapter.label_maps,adapter.weights(),original_bank)
    after_original=rng_state();restore_rng(before)
    predictions=adapter.predict_subset(bundle,'val',torch.device('cpu'),cfg)
    measured=adapter.validation_metrics(predictions)
    assert measured['taxomil_score']==ref[0] and measured['taxomil_coarse_accuracy']==ref[2] and measured['taxomil_fine_accuracy']==ref[3]
    np.testing.assert_allclose(measured['taxomil_generation_ce'],ref[1],rtol=1e-6,atol=2e-6)
    torch.testing.assert_close(rng_state()['torch'],after_original['torch'],rtol=0,atol=0)
    original_scheduler.step();adapter.selection_objective(measured,0,measured['taxomil_score'])
    assert adapter.scheduler.state_dict()==original_scheduler.state_dict()
    assert adapter.selected_epoch==0 and adapter.best_score==ref[0]
    check('literal_dualhead_autoregressive_fullVAL_generation_losses_exactmatch_LR_and_selection')
    predictions.to_csv(out/'val_predictions.csv',index=False)
    # A genuine full model/optimizer/RNG state is serialized and restored.
    checkpoint=out/'checkpoint.pt'
    torch.save({'model':adapter.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state()},checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    with torch.no_grad():next(adapter.network.parameters()).add_(.25)
    adapter.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    adapter.validate_optimizer_state()
    assert adapter.get_extra_state()==saved['model']['_extra_state']
    for name,param in original.named_parameters():torch.testing.assert_close(dict(adapter.network.named_parameters())[name],param,rtol=1e-6,atol=2e-6)
    report['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest();report['checkpoint_bytes']=checkpoint.stat().st_size
    report['losses']=values;report['optimizer_updates']=adapter.optimizer_updates
    check('actual_full_parameter_optimizer_scheduler_rng_checkpoint_serialized_and_restored')
    report.update(state='passed',software_fixture='synthetic_features_real_full_GPT2',paper_reproduction='not_run',real_wsi_validation='not_run')
    write_json(out/'validation.json',report)


if __name__=='__main__':main()
