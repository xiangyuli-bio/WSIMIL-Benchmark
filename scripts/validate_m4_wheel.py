"""Two complete original32epoch M4 budgets, native2048 and explicit1024 FM port."""
import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

PROGRAM = r'''
import gc,hashlib,sys,time
from pathlib import Path
target,out=map(Path,sys.argv[1:3]);dimension=int(sys.argv[3]);device=sys.argv[4]
sys.path.insert(0,str(target))
import torch,milplus
from milplus.data import Bundle
from milplus.m4_demo import create_m4_demo
from milplus.models import build_bundle_model
from milplus.engine import train,evaluate,seed_everything
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.83.0' and torch.cuda.is_available()
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
data=create_m4_demo(out/'data',patients=20,dimension=dimension,endpoints=10,patches=257);bundle=Bundle(data)
seed_everything(42);model=build_bundle_model('m4_multilabel_source',bundle)
parameters=7462613 if dimension==2048 else 4316885
assert sum(p.numel() for p in model.parameters())==parameters
assert len(model.network.experts)==5 and len(model.network.towers)==10
markers=[]
for index in range(5):
 for suffix in ['fc.0.weight','cpe.proj.weight','hierarchical.agg1.conv1.weight',
   'hierarchical.agg2.pointwise_conv.weight','hierarchical.agg3.conv1.weight','attention_net.attention_c.weight']:
  markers.append('network.experts.'+str(index)+'.'+suffix)
for index in range(10):markers += ['network.gate1_fc2.'+str(index)+'.0.weight','network.towers.'+str(index)+'.fc2.weight']
markers.append('network.gate1_fc1.0.weight')
state=model.state_dict();before={name:state[name].clone() for name in markers}
def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
unused={name:sha(value) for name,value in state.items() if name.startswith('network.experts.') and '.classifier.' in name}
assert len(unused)==10
del model,state;gc.collect()
run=train(data,'m4_multilabel_source',out/'run',seed=42,device=device,config={'threads':2})
assert read_json(run/'status.json')['completed_epochs']==32 and not (run/'test_predictions.csv').exists()
settings=read_json(run/'request.json')['config']
assert settings['epochs']==32 and settings['lr']==.0001 and settings['weight_decay']==.00004
assert settings['max_instances'] is None and settings['grad_clip'] is None and settings['selection_metric']=='m4_val_mean_auc'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
extra=last['model']['_extra_state'];train_slides=len(bundle.records('train'));val_slides=len(bundle.records('val'))
assert len(last['history'])==extra['completed_epochs']==32
assert extra['optimizer_updates']==32*train_slides
assert len(extra['train_orders'])==len(extra['val_orders'])==32
assert all(len(order)==len(set(order))==train_slides for order in extra['train_orders'])
assert all(len(order)==len(set(order))==val_slides for order in extra['val_orders'])
assert all(not torch.equal(before[name],last['model'][name]) for name in markers)
assert all(sha(last['model'][name])==digest for name,digest in unused.items())
opt=last['optimizers'][0]
assert len(opt['param_groups'])==1 and opt['param_groups'][0]['weight_decay']==.00004
assert opt['param_groups'][0]['lr']==.00001 and opt['param_groups'][0]['betas']==(.9,.999)
assert all(int(value['step'])==32*train_slides for value in opt['state'].values())
assert extra['scheduler']['last_epoch']==32 and extra['scheduler']['_step_count']==33
scores=[row['validation']['m4_val_mean_auc'] for row in last['history']]
assert best['epoch']==max(range(32),key=lambda i:scores[i]) and scores[best['epoch']]>0
for row in bundle.records('test'):
 features,_=bundle.load(row);assert features.shape==(257,dimension)
del last,best,extra,opt;gc.collect();torch.cuda.empty_cache()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
gc.collect();torch.cuda.empty_cache()
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
assert metrics['classes']==10 and metrics['n']==len(bundle.records('test'))
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
write_json(out/'validation.json',{'state':'passed','profile':'m4_multilabel_source','dimension':dimension,
 'condition':'native_RetCCL2048_source_dimensions' if dimension==2048 else 'explicit_FM1024_input_port_full_original_internal_widths',
 'epochs':32,'patients':20,'train_slides':train_slides,'test_slides':len(bundle.records('test')),
 'experts':5,'expert_width':512,'gate_slices':4,'tower_hidden':32,'endpoints':10,'patches_per_slide':257,
 'parameters':parameters,'actual_optimizer_updates':32*train_slides,'all51_mechanism_markers_updated':True,
 'all_original_unused_expert_classifiers_unchanged':True,'source_Adam_cosine_and_sampler_state_verified':True,
 'source_strict_VALmeanAUROC_selection_verified':True,'selected_checkpoint_reloaded_twice':True,
 'prediction_sha256':prediction,'metrics':metrics,'software':software_identity(),'device_type':'cuda',
 'device':torch.cuda.get_device_name(device),'synthetic':True,'formal_admission':False,'wall_seconds':time.monotonic()-started})
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    out, source = Path(args.output).resolve(), Path(args.source_tree).resolve()
    out.mkdir(parents=True, exist_ok=False)
    target = out/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        if any(not (target/name).resolve().is_relative_to(target) for name in archive.namelist()):
            raise ValueError('Unsafe wheel member')
        archive.extractall(target)
    reports = []
    process = {'state':'running','cases':[],'formal_admission':False}
    def save():
        (out/'process.json').write_text(json.dumps(process,indent=2)+'\n')
    try:
        for dimension in [2048,1024]:
            case = out/f'case_{dimension}'
            with (out/f'case_{dimension}.log').open('x') as log:
                child = subprocess.Popen([sys.executable,'-c',PROGRAM,str(target),str(case),str(dimension),args.device],
                    cwd=source,env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT)
                item = {'dimension':dimension,'pid':child.pid,'state':'running'}
                process['cases'].append(item);save();code=child.wait()
                item.update(returncode=code,state='passed' if code==0 else 'failed');save()
            if code:
                raise RuntimeError('M4 complete native budget failed; retain original scope and evidence')
            reports.append(json.loads((case/'validation.json').read_text()))
        assert all(report['state']=='passed' and report['software']==reports[0]['software'] for report in reports)
        (out/'validation.json').write_text(json.dumps({'state':'passed','profiles':reports,'software':reports[0]['software'],
            'synthetic':True,'formal_admission':False,'scope':'Both full32epoch budgets, native2048 and explicit1024 port; full5experts/512width/10endpoint/257patches/two independent reloads; no real WSI or paper metrics'},indent=2)+'\n')
        process['state']='passed';save()
    except BaseException as error:
        process.update(state='failed',error=repr(error));save();raise


if __name__ == '__main__':
    main()
