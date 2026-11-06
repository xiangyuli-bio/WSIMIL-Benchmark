"""Full released MUSE budgets from an installed wheel, never a tiny text model.

Four explicitly synthetic conditions retain all600 original descriptions,
the complete12-layer text tower,8 experts,257patch bags and20 optimizer
updates per visited TRAIN bag. Source/paper discrepancies remain separate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


COLD_CLI = r'''
import json,runpy,sys
from pathlib import Path
target,data,run,receipt=map(Path,sys.argv[1:5]);device=sys.argv[5]
sys.path.insert(0,str(target))
import torch,milplus
assert milplus.__version__=='0.85.0' and Path(milplus.__file__).resolve().is_relative_to(target)
assert device=='cuda:0' and torch.cuda.is_available() and not torch.cuda.is_initialized()
original=torch.cuda.reset_peak_memory_stats;observations=[]
def checked(*args,**kwargs):
 assert torch.cuda.is_initialized(),'Public CLI reset memory statistics before CUDA context initialization'
 observations.append(True);return original(*args,**kwargs)
torch.cuda.reset_peak_memory_stats=checked
config=receipt.with_suffix('.config.json');config.write_text(json.dumps({'epochs':1,'threads':2}))
sys.argv=['milplus','run','--data',str(data),'--method','MUSE','--output',str(run),
          '--device',device,'--config',str(config)]
try:runpy.run_module('milplus',run_name='__main__')
except SystemExit as error:
 assert error.code in {None,0},error.code
assert observations==[True]
status=json.loads((run/'status.json').read_text());assert status['state']=='completed'
state=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
assert state['model']['_extra_state']['optimizer_updates']==160
receipt.write_text(json.dumps({'state':'passed','cold_CUDA_at_entry':True,
 'context_initialized_by_public_engine_before_memory_statistics':True,
 'actual_original_method_alias_CLI':True,'full20view_updates':160,'device':device},indent=2)+'\n')
'''


PROGRAM = r'''
import gc,json,subprocess,sys
from pathlib import Path
target,out=map(Path,sys.argv[1:3]);cohort=sys.argv[3];dimension=int(sys.argv[4]);device=sys.argv[5]
sys.path.insert(0,str(target))
import torch,milplus
from milplus.muse_demo import create_muse_demo
from milplus.conmil_text import default_weights
from milplus.data import Bundle
from milplus.engine import train,evaluate,seed_everything
from milplus.models import build_bundle_model
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.85.0' and Path(milplus.__file__).resolve().is_relative_to(target)
assert device=='cuda:0' and torch.cuda.is_available() and not torch.cuda.is_initialized()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
out.mkdir(parents=True,exist_ok=False)
data=create_muse_demo(out/'data',default_weights(),patients=24,patches=257,cohort=cohort,dimension=dimension)
bundle=Bundle(data)
assert {s:len(bundle.records(s)) for s in ['train','val','test']}=={'train':8,'val':8,'test':8}
assert all(sum(r['target']==label for r in bundle.records('train'))==4 for label in [0,1])
# A separate fresh interpreter exercises the actual public CLI with a cold CUDA
# context. The assertion is inside reset_peak_memory_stats itself, not after a
# validator-side set_device/forward that could hide the engine initialization bug.
cold=out/'cold_CUDA_CLI.json'
subprocess.run([sys.executable,'-c',COLD_CLI,str(target),str(data),str(out/'cold_CLI_run'),str(cold),device],check=True)
assert read_json(cold)['state']=='passed'
assert not torch.cuda.is_initialized()
seed_everything(2026);model=build_bundle_model('muse_released_source',bundle)
parameters=sum(p.numel() for p in model.parameters())
assert parameters==88985107+(dimension-512)*512
assert model.network.text_base.shape==(600,512) and model.network.text_base_0.shape==(300,512)
assert model.network.prompt_learner.ctx.shape==(16,768)
assert len(model.network.text_encoder.transformer.resblocks)==12
assert len(model.network.moe.experts)==8 and model.network.moe.top_k==2
markers=['prompt_learner.ctx','text_encoder.transformer.resblocks.0.mlp.c_fc.weight',
 'text_encoder.transformer.resblocks.11.mlp.c_proj.weight','adapter.0.weight',
 'classifier.0.weight','moe.router.topkroute_linear.weight','moe.router.noise_linear.weight']
initial={name:value.detach().cpu().clone() for name,value in model.network.named_parameters() if name in markers}
assert len(initial)==len(markers)
del model;gc.collect()
cfg={'threads':2}
run=train(data,'muse_released_source',out/'run',seed=2026,device=device,config=cfg,stop_after=1)
assert read_json(run/'status.json')['state']=='interrupted'
partial=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
assert partial['model']['_extra_state']['completed_epochs']==1
assert partial['model']['_extra_state']['optimizer_updates']==160
partial_sha=digest_file(run/'last.pt');del partial;gc.collect()
train(data,'muse_released_source',run,seed=2026,device=device,config=cfg,resume=True)
request=read_json(run/'request.json');config=request['config']
assert config['epochs']==200 and config['patience']==20 and config['lr']==1e-4 and config['weight_decay']==1e-5
assert config['grad_clip'] is None and config['max_instances'] is None
assert config['selection_metric']=='muse_val_error'
assert read_json(run/'status.json')['state']=='completed' and not (run/'test_predictions.csv').exists()
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
extra=last['model']['_extra_state'];epochs=extra['completed_epochs'];updates=extra['optimizer_updates']
assert 82<=epochs<=200 and last['epoch']==epochs-1 and len(last['history'])==epochs
assert updates==epochs*8*20 and len(extra['train_orders'])==epochs and len(extra['retrieval_history'])==epochs
train_ids={row['slide_id'] for row in bundle.records('train')}
assert all(len(ids)==8 and set(ids)==train_ids for ids in extra['train_orders'])
assert all(len(rows)==8 and all(len(ids)==20 and len(set(ids))==20 for ids in rows) for rows in extra['retrieval_history'])
# Reconstruct exact strict first-minimum VAL-error selection and stopping.
selected=None;minimum=float('inf');bad=0;expected_stop=None
for row in last['history']:
 epoch=row['epoch'];error=row['validation']['muse_val_error']
 if error<minimum:minimum=error;bad=0;selected=epoch
 else:bad+=1
 if bad>=20 and epoch>80:
  expected_stop=epoch;break
assert best['epoch']==selected and last['bad']==bad
assert (expected_stop==epochs-1) if epochs<200 else (expected_stop is None or expected_stop==199)
assert last['optimizers'][0]['param_groups'][0]['lr']==1e-4
for name,value in initial.items():assert not torch.equal(value,last['model']['network.'+name]),name
counts=extra['gradient_updates']
assert counts['q_proj.weight']==0 and counts['agg.0.weight']==0
assert counts['prompt_learner.ctx']==updates and counts['text_encoder.transformer.resblocks.11.mlp.c_proj.weight']==updates
assert all(0<=n<=updates for n in counts.values())
restored=build_bundle_model('muse_released_source',bundle);restored.configure_training(config)
optimizer=restored.make_optimizer(config['lr'],config['weight_decay'])
restored.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizers'][0]);restored.validate_optimizer_state()
assert restored.completed_epochs==epochs and restored.optimizer_updates==updates
assert len(optimizer.state)==sum(n>0 for n in counts.values())
write_json(out/'complete_sparse_gradient_updates.json',counts)
del restored,optimizer,last,best,extra,initial;gc.collect()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target),name
write_json(out/'validation.json',{'state':'passed','profile':'muse_released_source','cohort':cohort,
 'feature_dimension':dimension,'parameters':parameters,'patients':24,'train_slides':8,'shots_per_class':4,
 'patches_per_slide':257,'text_layers':12,'context_tokens':16,'knowledge_per_class':300,
 'experts':8,'selected_experts':2,'separate_updates_per_TRAIN_visit':20,
 'max_epochs':200,'patience':20,'minimum_stop_epoch_index':81,'completed_epochs':epochs,
 'actual_optimizer_updates':updates,'actual_epoch1_interruption_checkpoint_sha256':partial_sha,
 'actual_full_Adam_sparse_moments_and_source_caches_restored':True,
 'actual_cold_CUDA_CLI_context_fix':True,'selected_checkpoint_independently_reloaded_twice':True,
 'all20views_and_complete_TRAIN_visits_retained':True,'exact_strict_VAL_error_selection_and_stopping':True,
 'device':device,'prediction_sha256':prediction,'metrics':metrics,'software':software_identity(),
 'synthetic':True,'formal_admission':False,'released_code_vs_paper_reconciliation':'pending',
 'real_wsi_validation':'not_run','paper_reproduction':'not_run'})
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0'])
    args = parser.parse_args()
    wheel, source, out = [Path(s).resolve() for s in [args.wheel,args.source_tree,args.output]]
    out.mkdir(parents=True,exist_ok=False);target=out/'wheel_import';target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        if any(not (target/name).resolve().is_relative_to(target) for name in archive.namelist()):
            raise ValueError('Unsafe wheel member')
        archive.extractall(target)
    reports=[]
    for cohort,dimension in [('tcga_brca',512),('tcga_nsclc',512),('camelyon_all',512),('tcga_brca',1024)]:
        name=cohort+'_'+str(dimension);case=out/name
        with (out/(name+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,'-c','COLD_CLI='+repr(COLD_CLI)+'\n'+PROGRAM,
                str(target),str(case),cohort,str(dimension),args.device],cwd=source,
                env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
                     'CUBLAS_WORKSPACE_CONFIG':':4096:8'},stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Complete installed-wheel MUSE budget failed: '+name)
        reports.append(json.loads((case/'validation.json').read_text()))
        (out/'process.json').write_text(json.dumps({'state':'running','completed_conditions':len(reports),'total_conditions':4},indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps({'state':'passed','profiles':reports,
        'formal_admission':False,'synthetic':True,
        'scope':'Four complete released-source200max/patience20/epoch>80 native MUSE budgets, all600descriptions/20separate updates, actual recovery/coldCUDA CLI/double reload. Whole gates and paper reconciliation remain separate.'},indent=2)+'\n')


if __name__=='__main__':
    main()
