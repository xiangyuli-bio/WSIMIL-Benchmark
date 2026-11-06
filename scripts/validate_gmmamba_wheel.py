"""Three complete100max-epoch GMMamba installed-wheel native conditions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

PROGRAM=r'''
import gc,sys
from pathlib import Path
target,out,inputs=map(Path,sys.argv[1:4]);profile=sys.argv[4];dimension=int(sys.argv[5]);classes=int(sys.argv[6]);device=sys.argv[7]
sys.path.insert(0,str(target))
import torch,milplus
from milplus.gmmamba_demo import create_gmmamba_demo
from milplus.gmmamba import SourceSelection
from milplus.data import Bundle
from milplus.engine import train,evaluate,seed_everything
from milplus.io import read_json,write_json,digest_file
from milplus.models import build_bundle_model
from milplus.provenance import software_identity
assert milplus.__version__=='0.86.0' and Path(milplus.__file__).resolve().is_relative_to(target)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
if device.startswith('cuda'):assert torch.cuda.is_available()
out.mkdir(parents=True,exist_ok=False)
data=create_gmmamba_demo(inputs,patients=12*classes,dimension=dimension,classes=classes,patches=257)
bundle=Bundle(data);count=len(bundle.records('train'));assert count==4*classes
assert all(sum(row['target']==label for row in bundle.records('train'))==4 for label in range(classes))
seed_everything(2026);model=build_bundle_model(profile,bundle)
original=model.network.imm.source if hasattr(model.network.imm,'source') else model.network.imm
for block in [original.BiMamba1[1],original.BiMamba2[1]]:
 assert (block.d_model,block.d_state,block.d_inner,block.d_conv)==(512,16,1024,4)
assert model.network.groups==10 and model.network.mask_ratio==.3
assert model.network.css.stoken_refine.num_heads==8
prefix='imm.source.' if hasattr(model.network.imm,'source') else 'imm.'
names=[prefix+'BiMamba1.1.in_proj.weight',prefix+'BiMamba2.1.A_b_log',
 'css.stoken_refine.in_proj_weight','classifier.cls_token','classifier.fc2.weight']
initial={name:value.detach().cpu().clone() for name,value in model.network.named_parameters() if name in names}
assert len(initial)==len(names);parameters=sum(p.numel() for p in model.parameters())
del model;gc.collect()
run=train(data,profile,out/'run',seed=2026,device=device,config={'threads':2},stop_after=3)
assert read_json(run/'status.json')['state']=='interrupted' and not (run/'best.pt').exists()
partial=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
assert partial['model']['_extra_state']['optimizer_updates']==3*count
partial_sha=digest_file(run/'last.pt');del partial;gc.collect()
train(data,profile,run,seed=2026,device=device,config={'threads':2},resume=True)
cfg=read_json(run/'request.json')['config']
assert cfg['epochs']==100 and cfg['lr']==1e-4 and cfg['weight_decay']==1e-5 and cfg['patience']==20
assert cfg['max_instances'] is None and cfg['grad_clip'] is None and cfg['selection_metric']=='gmmamba_val_auc_f1'
assert not (run/'test_predictions.csv').exists() and read_json(run/'status.json')['state']=='completed'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
extra=last['model']['_extra_state'];epochs=extra['completed_epochs'];updates=extra['optimizer_updates']
assert 52<=epochs<=100 and last['epoch']==epochs-1 and len(last['history'])==epochs
assert updates==epochs*count and len(extra['train_orders'])==epochs and len(extra['val_orders'])==epochs
assert all(len(ids)==count and len(set(ids))==count for ids in extra['train_orders'])
selection=SourceSelection();events=[]
for row in last['history']:
 values=row['validation'];selection.observe(row['epoch'],values['gmmamba_val_auc'],values['gmmamba_val_f1'])
 if selection.improved:events.append(row['epoch'])
assert selection.state_dict()==extra['selection'] and best['epoch']==selection.selected_epoch
assert events==extra['deferred_test_rng_epochs'] and events[0]>=6
if epochs<100:assert selection.done and selection.counter>=20 and selection.epoch>50
assert last['optimizers'][0]['param_groups'][0]['lr']==1e-4
for name,value in initial.items():assert not torch.equal(value,last['model']['network.'+name]),name
counts=extra['gradient_updates']
assert counts[prefix+'mask_attention.0.weight']==0
assert counts[prefix+'_fc1.0.weight']==(0 if profile=='gmmamba_mambamil_bimamba_source' else updates)
assert counts[prefix+'BiMamba1.1.A_log']==updates and counts[prefix+'BiMamba2.1.A_b_log']==updates
restored=build_bundle_model(profile,bundle);restored.configure_training(cfg);optimizer=restored.make_optimizer(cfg['lr'],cfg['weight_decay'])
restored.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizers'][0]);restored.validate_optimizer_state()
assert restored.completed_epochs==epochs and restored.optimizer_updates==updates
assert len(optimizer.state)==sum(n>0 for n in counts.values())
write_json(out/'full_parameter_gradient_updates.json',counts)
del restored,optimizer,last,best,extra,initial;gc.collect()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target),name
write_json(out/'validation.json',{'state':'passed','profile':profile,'dimension':dimension,'classes':classes,
 'parameters':parameters,'full_hidden_width':512,'state_size':16,'conv_width':4,'expansion':2,
 'groups':10,'mask_ratio':.3,'CSS_heads':8,'patches':257,'train_slides':count,'shots_per_class':4,
 'max_epochs':100,'completed_epochs':epochs,'actual_optimizer_updates':updates,
 'actual_interruption_checkpoint_sha256':partial_sha,'full_AdamW_sparse_moments_restored':True,
 'exact_source_selection_and_stopping':True,'original_test_RNG_effect_replayed_without_test_access':True,
 'selected_checkpoint_independently_reloaded_twice':True,'prediction_sha256':prediction,'metrics':metrics,
 'device':device,'backend':'explicit_complete_torch_recurrence','dependency_condition':'explicit_pinned_MambaMIL_BiMamba_port_original_GMMamba_dependency_unpublished',
 'persistent_native_inputs':str(data),'software':software_identity(),'synthetic':True,
 'formal_admission':False,'real_wsi_validation':'not_run','paper_reproduction':'not_run'})
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['wheel','source-tree','output','persistent-inputs']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();wheel,source,out,inputs=[Path(v).resolve() for v in
        [args.wheel,args.source_tree,args.output,args.persistent_inputs]]
    out.mkdir(parents=True,exist_ok=False);inputs.mkdir(parents=True,exist_ok=False)
    target=out/'wheel_import';target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        if any(not (target/name).resolve().is_relative_to(target) for name in archive.namelist()):
            raise ValueError('Unsafe installed-wheel member')
        archive.extractall(target)
    rows=[]
    for profile,dimension,classes in [('gmmamba_mambamil_bimamba_source',512,2),
        ('gmmamba_fm_projection_repaired',896,3),('gmmamba_fm_projection_repaired',1024,2)]:
        name=str(dimension)+'_'+str(classes);case=out/('case_'+name)
        with (out/(name+'.log')).open('x') as log:
            code=subprocess.run([sys.executable,'-c',PROGRAM,str(target),str(case),str(inputs/('case_'+name)),
                profile,str(dimension),str(classes),args.device],cwd=source,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
                     'CUBLAS_WORKSPACE_CONFIG':':4096:8'},stdout=log,stderr=subprocess.STDOUT).returncode
        if code:raise RuntimeError('Full native GMMamba installed-wheel budget failed: '+name)
        rows.append(json.loads((case/'validation.json').read_text()))
        (out/'process.json').write_text(json.dumps({'state':'running','completed_conditions':len(rows),'total_conditions':3},indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'state':'passed','profiles':rows,'synthetic':True,
        'formal_admission':False,'scope':'Three full100max/epoch>50 native512/896/1024 conditions, full257patch/10groups/512hidden, actual saved-state recovery and double reload. Real WSI, missing original backend provenance and complete release remain separate.'},indent=2)+'\n')


if __name__=='__main__':main()
