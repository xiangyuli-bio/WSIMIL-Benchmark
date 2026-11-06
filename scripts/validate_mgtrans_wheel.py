"""Four independent-wheel native200max MG-Trans budgets, full512 PAM and MI.

Device workers only parallelize the four fixed cases. Every case, the complete
source budget, native readers, real synthetic512/640-node inputs and all losses
remain mandatory. These are synthetic engineering references, not paper scores.
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
import gc,json,sys,time,shutil
from pathlib import Path
target,out=map(Path,sys.argv[1:3]);case=int(sys.argv[3]);device=sys.argv[4]
sys.path.insert(0,str(target))
import torch,milplus
import pandas as pd
from milplus.data import Bundle
from milplus.engine import train,evaluate,seed_everything
from milplus.models import build_bundle_model
from milplus.mgtrans_demo import create_mgtrans_demo
from milplus.io import write_json,read_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.71.0' and torch.cuda.is_available()
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
torch.cuda.set_device(device);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
assert shutil.disk_usage('/dev/shm').total>=512*1024**2,'Full native4-reader prefetch needs adequate shared memory'
dimension,classes,patients,roots=[(1024,2,40,512),(1024,3,36,640),(1536,2,40,512),(1536,3,36,640)][case]
out.mkdir(parents=True,exist_ok=False)
started=time.monotonic()
data=create_mgtrans_demo(out/'data',patients=patients,dimension=dimension,classes=classes,roots=roots)
bundle=Bundle(data);n=len(bundle.records('train'));assert n==36 and bundle.config['synthetic'] is True
assert all(bundle.mgtrans.inventory(row)['nodes_per_level']==[roots,roots*4] for row in bundle.records('train'))
seed_everything(2026);model=build_bundle_model('mgtrans_source_repaired',bundle)
assert model.num_workers is None and model._loader('train',device)[0].num_workers==4
assert model.network.transformer.embeddings.patch_embeddings[0].weight.shape==(192,dimension)
assert model.network.transformer.encoder.layer2.gcn.num_layers==3
markers=['head_low.weight','head_high.weight','IB_layer.weight','transformer.encoder.layer2.gcn.conv1.weight']
initial=model.network.state_dict();before={name:initial[name].clone() for name in markers}
del initial,model;gc.collect()
write_json(out/'progress.json',{'state':'running_complete_native_budget','case':case,'dimension':dimension,'classes':classes,'roots':roots})
run=train(data,'mgtrans_source_repaired',out/'run',device=device,seed=2026,config={'threads':2})
status=read_json(run/'status.json');request=read_json(run/'request.json');cfg=request['config']
assert status['state']=='completed' and cfg['epochs']==200 and cfg['patience']==20
assert cfg['lr']==.0001 and cfg['weight_decay']==1e-5 and cfg['model']=={}
assert cfg['max_instances'] is None and cfg['grad_clip'] is None and cfg['selection_metric']=='mgtrans_val_error'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
extra=last['model']['_extra_state'];epochs=extra['completed_epochs'];history=last['history']
assert epochs==len(history) and extra['optimizer_updates']==epochs*n
assert extra['mi_updates']==[n-31]*epochs and len(extra['train_orders'])==epochs
actual_slides={row['slide_id'] for row in bundle.records('train')}
assert all(len(order)==n and set(order)==actual_slides for order in extra['train_orders'])
assert epochs==200 or (epochs>=52 and extra['training_done'] and extra['early_bad']>=20 and last['epoch']>50)
assert all(any(abs(row['train'][key])>0 for row in history) for key in ['source_mib','source_semantic_and_ce'])
assert all(not torch.equal(before[name],last['model']['network.'+name]) for name in markers)
assert last['model']['network.transformer.embeddings.patch_embeddings.0.weight'].shape==(192,dimension)
assert last['model']['network.transformer.attshort.sa_layer.ffn.fc1.weight'].shape==(768,192)
assert last['model']['network.head.weight'].shape==(classes,192)
assert all(torch.isfinite(value).all() and value.dtype==torch.float32 for value in last['model'].values() if torch.is_tensor(value))
errors=[row['validation']['mgtrans_val_error'] for row in history]
selected=max(i for i,value in enumerate(errors) if value==min(errors))
assert best['epoch']==selected and extra['selection_history'][-1]['best_error']==min(errors)
assert len(last['optimizers'])==1 and all(group['lr']==.0001 and group['weight_decay']==1e-5 for group in last['optimizers'][0]['param_groups'])
protocol=read_json(run/'method_protocol.json')
assert protocol['actual_mib_updates']==epochs*(n-31) and protocol['actual_optimizer_updates']==epochs*n
assert 'fresh native thread' in protocol['graph']
del last,best,extra;gc.collect();torch.cuda.empty_cache()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
frame=pd.read_csv(run/'test_predictions.csv');assert frame.slide_id.is_unique and len(frame)==len(bundle.records('test'))
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
report={'state':'passed','case':case,'profile':'mgtrans_source_repaired','software':software_identity(),
 'dimension':dimension,'classes':classes,'patients':patients,'real_synthetic_coarse_patches':roots,'fine_patches':roots*4,
 'PAM_anchors':512,'hidden':192,'heads':3,'MLP':768,'GCN_layers':3,'native_num_workers':4,
 'max_epochs':200,'completed_epochs':epochs,'patience':20,'source_stop_epoch_zero_based':50,
 'batch_size':1,'train_slides':n,'actual_optimizer_updates':epochs*n,'actual_MI_updates':epochs*(n-31),
 'all_four_mechanism_markers_updated':True,'all_source_objectives_active':True,'selected_epoch':selected,
 'source_latest_tie_selection_verified':True,'selected_checkpoint_reloaded_twice':True,'prediction_sha256':prediction,
 'metrics':metrics,'device':torch.cuda.get_device_name(device),'deterministic_CUDA_algorithms':True,
 'wall_seconds':time.monotonic()-started,'synthetic':True,'formal_admission':False,
 'scope':'Full native released graph/MI training with explicit device/API/HNSW TLS ports; no naturalWSI or historical paper-score claim'}
write_json(out/'validation.json',report)
print(json.dumps({'state':'passed','case':case,'epochs':epochs,'actual_MI_updates':epochs*(n-31)}),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--devices', default='cuda:0', help='Unique native CUDA workers; all four cases remain mandatory')
    parser.add_argument('--workers-per-device', type=int, default=1,
                        help='Independent verification processes per GPU; concurrent wall time is not an efficiency benchmark')
    args = parser.parse_args()
    devices = args.devices.split(',')
    if (not devices or len(set(devices)) != len(devices) or any(not re.fullmatch(r'cuda:[0-9]+', device) for device in devices)
            or not 1 <= args.workers_per_device <= 4):
        raise ValueError('Provide distinct explicit CUDA devices')
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    target = out/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith('milplus/') and not name.endswith('/'):
                assert archive.read(name) == (Path(args.source_tree)/'src'/name).read_bytes(), name
        archive.extractall(target)
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join(path for path in environment.get('PYTHONPATH', '').split(os.pathsep)
        if path and Path(path).resolve() != (Path(args.source_tree)/'src').resolve())
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    cases = [{'case': index, 'state': 'pending'} for index in range(4)]
    active, handles, next_case = {}, [], 0
    slots = [(device, worker) for device in devices for worker in range(args.workers_per_device)]
    started = time.monotonic()
    def save():
        value = {'state': 'running', 'profiles': cases, 'wall_seconds': time.monotonic()-started,
                 'all_four_complete_source_budgets_required': True}
        temporary = out/'progress.tmp'; temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(out/'progress.json')
    try:
        while next_case < 4 or active:
            for slot in slots:
                device, worker = slot
                if next_case < 4 and slot not in active:
                    index = next_case; next_case += 1
                    log = (out/f'case_{index}.log').open('x'); handles.append(log)
                    command = [sys.executable, '-c', PROGRAM, str(target), str(out/f'case_{index}'), str(index), device]
                    child = subprocess.Popen(command, cwd=out, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    active[slot] = (index, child)
                    cases[index].update(state='running', pid=child.pid, device=device, independent_worker_slot=worker)
                    save()
            for slot, (index, child) in list(active.items()):
                code = child.poll()
                if code is not None:
                    cases[index].update(state='passed' if code == 0 else 'failed', returncode=code)
                    del active[slot]; save()
            time.sleep(1)
        if any(row['state'] != 'passed' for row in cases):
            raise RuntimeError('One or more complete native MG-Trans cases failed; preserve and inspect all case logs')
        reports = [json.loads((out/f'case_{index}/validation.json').read_text()) for index in range(4)]
        assert {(row['dimension'], row['classes']) for row in reports} == {(1024, 2), (1024, 3), (1536, 2), (1536, 3)}
        assert all(row['state'] == 'passed' and row['software'] == reports[0]['software'] for row in reports)
        report = {'state': 'passed', 'profiles': reports, 'software': reports[0]['software'],
                  'wall_seconds': time.monotonic()-started, 'formal_admission': False,
                  'independent_workers_per_device': args.workers_per_device,
                  'efficiency_benchmark': False,
                  'scope': 'All four independent-wheel complete200max native graph/MI recipes; broad CPU suite remains a separate gate'}
        (out/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'state': 'passed', 'profiles': 4}), flush=True)
    finally:
        for log in handles:
            log.close()


if __name__ == '__main__':
    main()
