"""Four complete100max native TaxoMIL budgets from the actual installed wheel."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

PROGRAM=r'''
import gc,math,sys
from pathlib import Path
target,out,inputs,assets=map(Path,sys.argv[1:5]);profile=sys.argv[5];dataset=sys.argv[6];device=sys.argv[7]
sys.path.insert(0,str(target))
import torch,milplus
from milplus.taxomil_demo import create_taxomil_demo
from milplus.data import Bundle
from milplus.engine import train,evaluate,seed_everything
from milplus.io import read_json,write_json,digest_file
from milplus.models import build_bundle_model
from milplus.provenance import software_identity
assert milplus.__version__=='0.88.0' and Path(milplus.__file__).resolve().is_relative_to(target)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
if device.startswith('cuda'):assert torch.cuda.is_available()
out.mkdir(parents=True,exist_ok=False)
classes=7 if dataset=='BRACS' else 6
if not inputs.exists():create_taxomil_demo(inputs,assets,dataset=dataset,patients=(16 if dataset=='PANDA' else 14)*classes,patches=257,train_per_class=12 if dataset=='PANDA' else 10)
bundle=Bundle(inputs);count=len(bundle.records('train'));assert count==(12 if dataset=='PANDA' else 10)*classes and count>64
assert len(bundle.records('val'))==len(bundle.records('test'))==2*classes
seed_everything(2026);model=build_bundle_model(profile,bundle)
assert len(model.network.decoder.shared_trunk.h)==12
assert model.network.decoder.shared_trunk.config.n_embd==768
assert len(model.network.tokenizer)==50259
parameters=sum(p.numel() for p in model.parameters());assert parameters>200_000_000
names=['decoder.shared_trunk.h.0.attn.c_attn.weight','decoder.coarse_head.attn.c_attn.weight',
       'decoder.fine_head.attn.c_attn.weight','aggregator.feature.0.weight','coarse_adaptor.0.weight',
       'fine_adaptor.0.weight','text_projector.weight','decoder.coarse_unembedding.weight','decoder.fine_unembedding.weight']
initial={name:value.detach().cpu().clone() for name,value in model.network.named_parameters() if name in names}
assert len(initial)==len(names)
del model;gc.collect()
run=train(inputs,profile,out/'run',seed=2026,device=device,config={'threads':2},stop_after=3)
assert read_json(run/'status.json')['state']=='interrupted'
partial=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
assert partial['model']['_extra_state']['optimizer_updates']==3*2
partial_sha=digest_file(run/'last.pt');del partial;gc.collect()
train(inputs,profile,run,seed=2026,device=device,config={'threads':2},resume=True)
cfg=read_json(run/'request.json')['config'];lr=1e-4 if 'paper_lr' in profile else 1e-5
assert cfg['epochs']==100 and cfg['lr']==lr and cfg['weight_decay']==.01 and cfg['patience']==20
assert cfg['max_instances'] is None and cfg['grad_clip'] is None and cfg['selection_metric']=='taxomil_score'
assert not (run/'test_predictions.csv').exists() and read_json(run/'status.json')['state']=='completed'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
extra=last['model']['_extra_state'];epochs=extra['completed_epochs'];updates=extra['optimizer_updates']
assert 21<=epochs<=100 and last['epoch']==epochs-1 and len(last['history'])==epochs
assert updates==epochs*2 and len(extra['train_orders'])==len(extra['val_orders'])==epochs
assert all(len(ids)==count and len(set(ids))==count for ids in extra['train_orders'])
score=-1e9;bad=0;selected=None
for row in last['history']:
 current=row['validation']['taxomil_score']
 if current>score:score=current;bad=0;selected=row['epoch']
 else:bad+=1
assert score==extra['best_score'] and bad==extra['bad'] and selected==extra['selected_epoch']==best['epoch']
assert epochs==100 or bad>=20
assert extra['scheduler']['last_epoch']==epochs
for name,value in initial.items():assert not torch.equal(value,last['model']['network.'+name]),name
assert extra['gradient_updates']['text_projector.weight']==updates
restored=build_bundle_model(profile,bundle);restored.configure_training(cfg);optimizer=restored.make_optimizer(lr,.01)
restored.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizers'][0]);restored.validate_optimizer_state()
assert restored.optimizer_updates==updates and restored.completed_epochs==epochs
write_json(out/'full_gradient_updates.json',extra['gradient_updates'])
del restored,optimizer,last,best,extra,initial;gc.collect()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target),name
write_json(out/'validation.json',{'state':'passed','profile':profile,'dataset':dataset,'classes':classes,
 'parameters':parameters,'dimension':1024,'pretrained_GPT2_layers':12,'hidden':768,'vocabulary':50259,
 'batch_size':64,'actual_batches_each_epoch':2,'patches':257,'train_slides':count,'shots_per_class':12 if dataset=='PANDA' else 10,
 'max_epochs':100,'completed_epochs':epochs,'actual_optimizer_updates':updates,
 'actual_interruption_checkpoint_sha256':partial_sha,'full_AdamW_and_cosine_state_restored':True,
 'exact_source_selection_and_stopping':True,'selected_checkpoint_independently_reloaded_twice':True,
 'prediction_sha256':prediction,'metrics':metrics,'device':device,'software':software_identity(),
 'persistent_native_inputs':str(inputs),'synthetic':True,'formal_admission':False,
 'real_wsi_validation':'not_run','paper_reproduction':'not_run'})
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['wheel','source-tree','output','persistent-inputs','assets']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args()
    wheel,source,out,inputs,assets=[Path(v).resolve() for v in [args.wheel,args.source_tree,args.output,args.persistent_inputs,args.assets]]
    out.mkdir(parents=True,exist_ok=False);inputs.mkdir(parents=True,exist_ok=False);target=out/'wheel_import';target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        if any(not (target/name).resolve().is_relative_to(target) for name in archive.namelist()):raise ValueError('Unsafe installed-wheel member')
        archive.extractall(target)
    rows=[]
    for profile in ['taxomil_gpt2_all_losses_key_repaired','taxomil_gpt2_paper_lr_key_repaired']:
        for dataset in ['BRACS','PANDA']:
            name=profile+'_'+dataset;case=out/name
            with (out/(name+'.log')).open('x') as log:
                result=subprocess.run([sys.executable,'-c',PROGRAM,str(target),str(case),str(inputs/dataset),str(assets),
                    profile,dataset,args.device],cwd=source,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',
                    'PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','HF_HUB_OFFLINE':'1',
                    'TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'},stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Full native TaxoMIL installed-wheel budget failed: '+name)
            rows.append(json.loads((case/'validation.json').read_text()))
            (out/'process.json').write_text(json.dumps({'state':'running','completed_conditions':len(rows),'total_conditions':4},indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'state':'passed','profiles':rows,'synthetic':True,
        'formal_admission':False,'scope':'Four real full-pretrained100max/patience20 native source/paperLR BRACS/PANDA budgets; full257patch/1024D, actual64+remainder batches, real interruption/fullstate restore and double selected reload. No real WSI/paper score claim.'},indent=2)+'\n')


if __name__=='__main__':main()
