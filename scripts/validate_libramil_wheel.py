"""Installed-wheel full20epoch native512/46description Libra-MIL references."""
import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

PROGRAM = r'''
import gc,json,sys
from pathlib import Path
target,out=map(Path,sys.argv[1:3]);profile=sys.argv[3];device=sys.argv[4]
sys.path.insert(0,str(target))
import torch,milplus
from milplus.libramil_demo import create_libramil_demo
from milplus.libramil_training import SourceSelection
from milplus.conmil_text import default_weights
from milplus.data import Bundle
from milplus.engine import train,evaluate,seed_everything
from milplus.models import build_bundle_model
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.84.0' and Path(milplus.__file__).resolve().is_relative_to(target)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
if device.startswith('cuda:'):assert torch.cuda.is_available()
out.mkdir(parents=True,exist_ok=False)
data=create_libramil_demo(out/'data',default_weights(),patients=30,patches=257)
bundle=Bundle(data);count=len(bundle.records('train'));assert count==27
seed_everything(2026);model=build_bundle_model(profile,bundle)
prototypes=4 if profile=='libramil_source_p4' else 6
assert model.network.proto_vis.shape==(prototypes,512)
assert model.network.prompt_inst.shape==(46,512) and model.network.prompt_bag.shape==(3,512)
assert model.network.num_heads==8 and model.network.ot_iter==20 and model.network.ot_epsilon==.05
expected_parameters=1869315 if prototypes==4 else 1870339
assert sum(p.numel() for p in model.parameters())==expected_parameters
initial={name:value.clone() for name,value in model.network.named_parameters()}
del model;gc.collect()
run=train(data,profile,out/'run',seed=2026,device=device,config={'threads':2})
cfg=read_json(run/'request.json')['config']
assert cfg['epochs']==20 and cfg['lr']==.0001 and cfg['weight_decay']==.01
assert cfg['max_instances'] is None and cfg['grad_clip'] is None and cfg['selection_metric']=='libramil_val_accuracy'
assert not (run/'test_predictions.csv').exists()
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
extra=last['model']['_extra_state'];opt=last['optimizers'][0]
assert last['epoch']==19 and len(last['history'])==20
assert extra['completed_epochs']==20 and extra['optimizer_updates']==20*count==540
assert len(extra['train_orders'])==20 and all(len(x)==len(set(x))==count for x in extra['train_orders'])
assert extra['scheduler']['last_epoch']==540 and extra['scheduler']['_step_count']==541
assert opt['param_groups'][0]['lr']==0 and opt['param_groups'][0]['weight_decay']==.01
assert len(opt['state'])==len(initial)-2
assert all(int(value['step'])==540 for value in opt['state'].values())
for name,value in initial.items():
 actual=last['model']['network.'+name]
 if name in {'temp_struct','temp_vis'}:assert torch.equal(value,actual)
 else:assert not torch.equal(value,actual),name
selection=SourceSelection()
for row in last['history']:
 metrics=row['validation'];selection.observe(row['epoch'],metrics['libramil_val_accuracy'],metrics['libramil_val_auc'])
assert selection.state_dict()==extra['selection'] and best['epoch']==selection.selected_epoch
restored=build_bundle_model(profile,bundle);restored.configure_training(cfg)
optimizer=restored.make_optimizer(cfg['lr'],cfg['weight_decay'])
restored.load_state_dict(last['model']);optimizer.load_state_dict(opt)
assert restored.scheduler.state_dict()==extra['scheduler']
assert len(optimizer.state)==len(initial)-2
del restored,optimizer,last,best,extra,opt,initial;gc.collect()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target),name
write_json(out/'validation.json',{'state':'passed','profile':profile,'prototypes':prototypes,'parameters':expected_parameters,
 'epochs':20,'actual_optimizer_updates':540,'train_slides':count,'native_dimension':512,'patches':257,
 'original_structural_descriptions':46,'class_descriptions':3,'attention_heads':8,'sinkhorn_iterations':20,
 'actual_original_pretrained_CONCH_text':True,'full_source_AdamW_cosine_and_selection_restored':True,
 'every_active_parameter_updated_and_unused_temperatures_unchanged':True,'selected_checkpoint_independently_reloaded_twice':True,
 'prediction_sha256':prediction,'metrics':metrics,'device':device,'software':software_identity(),
 'synthetic':True,'formal_admission':False,'real_wsi_validation':'not_run','paper_reproduction':'not_run'})
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    wheel, source, out = map(lambda s: Path(s).resolve(), [args.wheel, args.source_tree, args.output])
    out.mkdir(parents=True, exist_ok=False); target = out/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        if any(not (target/name).resolve().is_relative_to(target) for name in archive.namelist()):
            raise ValueError('Unsafe wheel member')
        archive.extractall(target)
    reports = []
    for profile in ['libramil_source_p4', 'libramil_readme_p6']:
        case = out/profile
        with (out/(profile+'.log')).open('x') as log:
            result = subprocess.run([sys.executable, '-c', PROGRAM, str(target), str(case), profile, args.device],
                cwd=source, env={**os.environ, 'PYTHONUNBUFFERED':'1', 'PYTHONDONTWRITEBYTECODE':'1',
                                'CUBLAS_WORKSPACE_CONFIG':':4096:8'}, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Full original installed-wheel Libra-MIL budget failed: '+profile)
        reports.append(json.loads((case/'validation.json').read_text()))
    if not all(row['state'] == 'passed' for row in reports):
        raise RuntimeError('Incomplete actual Libra-MIL native reports')
    (out/'validation.json').write_text(json.dumps({'state':'passed', 'profiles':reports,
        'formal_admission':False, 'synthetic':True,
        'scope':'Both complete installed-wheel20epoch native512/46description profiles; complete regression and realWSI/paper evidence separate'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
