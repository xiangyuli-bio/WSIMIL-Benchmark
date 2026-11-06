"""Four full native512RGB/25epoch independent-wheel C2C GPU references.

No frozen-feature, smaller-CNN, top-k or missing-loss fallback. Parallelism
changes only independent case placement, never model/input/recipe budgets.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path


PROGRAM = r'''
import gc,json,sys,time
from pathlib import Path
target,out=map(Path,sys.argv[1:3]);case=int(sys.argv[3]);device=sys.argv[4]
sys.path.insert(0,str(target))
import torch,milplus
from milplus.c2c_demo import create_c2c_demo
from milplus.data import Bundle
from milplus.engine import train,evaluate,seed_everything
from milplus.models import build_bundle_model
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.77.0' and torch.cuda.is_available()
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
torch.cuda.set_device(device);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
profile,classes=[('c2c_cluster_source',2),('c2c_cluster_default_bn_source',2),
 ('c2c_cluster_multiclass_repaired',3),('c2c_cluster_multiclass_repaired',7)][case]
out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
data=create_c2c_demo(out/'data',patients=5*classes,classes=classes,patches=144,patch_pixels=512)
bundle=Bundle(data);n=len(bundle.records('train'));assert n==3*classes
assert all(bundle.c2c.count(row)==144 for row in bundle.rows.to_dict('records'))
assert all(bundle.c2c.inventory(row)['shape']==[144,512,512,3] for row in bundle.rows.to_dict('records'))
seed_everything(2026);model=build_bundle_model(profile,bundle)
markers=['resnet_head.0.weight','resnet_head.7.1.conv2.weight','resnet_tail.0.weight',
 'attention.0.weight','classifier.0.weight','patch_classifier.0.weight']
initial=model.network.state_dict();before={name:initial[name].clone() for name in markers}
assert sum(p.numel() for p in model.network.parameters())==11326661+(classes-2)*130
assert all(module.track_running_stats==(profile=='c2c_cluster_default_bn_source')
 for module in model.network.modules() if isinstance(module,torch.nn.BatchNorm2d))
del model,initial;gc.collect()
run=train(data,profile,out/'run',seed=2026,device=device,config={'threads':2})
assert not (run/'test_predictions.csv').exists()
cfg=read_json(run/'request.json')['config']
assert cfg['epochs']==cfg['patience']==25 and cfg['lr']==1e-4 and cfg['weight_decay']==0.
assert cfg['model']=={} and cfg['max_instances'] is None and cfg['grad_clip'] is None
assert cfg['selection_metric']=='c2c_val_accuracy'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
state=last['model'];extra=state['_extra_state'];history=last['history']
assert len(history)==25 and last['epoch']==24 and extra['completed_epochs']==25
assert extra['optimizer_updates']==25*n and len(extra['cluster_history'])==25
assert all(len(order)==n and set(order)=={r['slide_id'] for r in bundle.records('train')} for order in extra['train_orders'])
for slide,cache in extra['clusters'].items():
 assert cache['labels'].shape==(144,) and cache['centers'].shape==(8,64) and cache['distances'].shape==(144,)
 assert 0<=cache['labels'].min() and cache['labels'].max()<8
assert state['network.resnet_head.0.weight'].shape==(64,3,7,7)
assert state['network.resnet_head.7.1.conv2.weight'].shape==(512,512,3,3)
assert state['network.resnet_tail.0.weight'].shape==(256,512)
assert state['network.resnet_tail.2.weight'].shape==(64,256)
assert state['network.attention.0.weight'].shape==(32,64)
assert state['network.classifier.0.weight'].shape==(classes,64)
assert state['network.patch_classifier.0.weight'].shape==(classes,64)
assert all(not torch.equal(before[name],state['network.'+name]) for name in markers)
counts=[int(v) for key,v in state.items() if key.endswith('num_batches_tracked')]
assert counts and all(count==(25*n if profile=='c2c_cluster_default_bn_source' else 0) for count in counts)
for record in history:
 assert 1<=record['train']['selected_patches']<=64 and record['train']['kl_active_groups']>=1
 assert all(__import__('math').isfinite(record['train'][key]) for key in ['loss','bag_ce','patch_ce','cluster_kl'])
assert any(record['train']['cluster_kl']>0 for record in history)
accuracy=[r['validation']['c2c_val_accuracy'] for r in history]
selected=max(i for i,value in enumerate(accuracy) if value==max(accuracy));assert best['epoch']==selected
assert len(last['optimizers'])==1
optimizer=last['optimizers'][0];assert len(optimizer['param_groups'])==1
assert optimizer['param_groups'][0]['lr']==1e-4 and optimizer['param_groups'][0]['weight_decay']==0.
assert all(int(v['step'])==25*n for v in optimizer['state'].values())
protocol=read_json(run/'method_protocol.json');assert protocol['optimizer_updates']==25*n
assert protocol['weights']=='5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8'
del last,best,state,extra,optimizer;gc.collect();torch.cuda.empty_cache()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
write_json(out/'validation.json',{'state':'passed','case':case,'profile':profile,'classes':classes,
 'pixels':512,'all_patches':144,'encoding_chunk':128,'actual_tail':16,'clusters':8,'samples_per_cluster':8,
 'epochs':25,'batch_size':1,'train_slides':n,'actual_optimizer_updates':25*n,
 'architecture':'full actual pretrained ResNet18/256/64/32attention','all_six_mechanism_markers_updated':True,
 'original_bn_counter_and_buffer_semantics_verified':True,'all_three_original_objectives_active':True,
 'selected_epoch':selected,'source_latest_tie_selection_verified':True,'selected_checkpoint_reloaded_twice':True,
 'prediction_sha256':prediction,'metrics':metrics,'software':software_identity(),'synthetic':True,
 'device':torch.cuda.get_device_name(device),'wall_seconds':time.monotonic()-started,'formal_admission':False,
 'scope':'Full native512RGB/25epoch end-to-end original cluster route with explicit historical assets and device/FAISS ports; no naturalWSI or historical paper-score claim'})
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--devices', default='cuda:0')
    parser.add_argument('--workers-per-device', type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    devices = args.devices.split(',')
    if not devices or len(set(devices)) != len(devices) or any(not re.fullmatch(r'cuda:\d+', d) for d in devices):
        raise ValueError('Specify unique actual CUDA devices without changing the platform GPU mapping')
    wheel, source, output = [Path(value).resolve() for value in [args.wheel, args.source_tree, args.output]]
    if output.exists():
        raise ValueError('Preserve native attempts; choose a new output root')
    output.mkdir(parents=True); target = output/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not (target/name).resolve().is_relative_to(target):
                raise ValueError('Unsafe installed wheel member')
        archive.extractall(target)
    cases = [{'case': index, 'state': 'pending'} for index in range(4)]
    slots = [device for device in devices for _ in range(args.workers_per_device)]
    active, handles, cursor = {}, [], 0
    def save():
        (output/'process.json').write_text(json.dumps({'state': 'running', 'cases': cases,
            'formal_admission': False, 'workers_per_device': args.workers_per_device}, indent=2)+'\n')
    try:
        while cursor < 4 or active:
            for slot, device in enumerate(slots):
                if slot in active or cursor >= 4:
                    continue
                index = cursor; cursor += 1
                log = (output/f'case_{index}.log').open('x'); handles.append(log)
                command = [sys.executable, '-c', PROGRAM, str(target), str(output/f'case_{index}'), str(index), device]
                child = subprocess.Popen(command, cwd=source,
                    env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
                    stdout=log, stderr=subprocess.STDOUT)
                cases[index].update(state='running', pid=child.pid, device=device)
                active[slot] = (index, child); save()
            for slot, (index, child) in list(active.items()):
                code = child.poll()
                if code is not None:
                    cases[index].update(state='passed' if code == 0 else 'failed', returncode=code)
                    del active[slot]; save()
            time.sleep(1)
        if any(case['state'] != 'passed' for case in cases):
            raise RuntimeError('One or more complete C2C native cases failed; preserve and inspect logs, no reduced fallback')
        reports = [json.loads((output/f'case_{index}/validation.json').read_text()) for index in range(4)]
        assert len(reports) == 4 and all(row['state'] == 'passed' and row['software'] == reports[0]['software'] for row in reports)
        assert [(r['profile'], r['classes']) for r in reports] == [
            ('c2c_cluster_source', 2), ('c2c_cluster_default_bn_source', 2),
            ('c2c_cluster_multiclass_repaired', 3), ('c2c_cluster_multiclass_repaired', 7)]
        (output/'validation.json').write_text(json.dumps({'state': 'passed', 'profiles': reports,
            'software': reports[0]['software'], 'formal_admission': False, 'synthetic': True,
            'independent_workers_per_device': args.workers_per_device, 'efficiency_benchmark': False,
            'scope': 'All four independent-wheel actual native512RGB/25epoch C2C references; whole regression and installed final package gates remain separate'}, indent=2)+'\n')
    finally:
        for handle in handles: handle.close()


if __name__ == '__main__':
    main()
