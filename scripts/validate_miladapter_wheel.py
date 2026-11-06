"""All26 original MIL-Adapter conditions x both actual native FM widths.

Each independent installed-wheel run uses full257patch bags, default k8,
20epochs, full native512/768D author aggregators and actual author prototypes.
No source-only probe, short training or dummy text can satisfy this gate.
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
import gc,sys,time
from pathlib import Path
target,out=map(Path,sys.argv[1:3]);case=int(sys.argv[3]);device=sys.argv[4]
sys.path.insert(0,str(target))
import torch,milplus
from milplus.miladapter_demo import create_miladapter_demo
from milplus.miladapter_source import PROFILES
from milplus.models import build_bundle_model
from milplus.engine import train,evaluate,seed_everything
from milplus.data import Bundle
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.80.0' and torch.cuda.is_available()
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
torch.cuda.set_device(device);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
profile=sorted(PROFILES)[case//2];encoder=['CONCH','KEEP'][case%2]
project=['BRCA','NSCLC','RCC'][(case//2)%3];dim=512 if encoder=='CONCH' else 768
classes=3 if project=='RCC' else 2
out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
data=create_miladapter_demo(out/'data',patients=12*classes,project=project,encoder=encoder,patches=257)
bundle=Bundle(data);assert bundle.dim==dim and bundle.miladapter.prototypes.shape==(dim,classes)
assert all(r['count']==257 for r in bundle.miladapter.meta['records'].values())
seed_everything(2026);model=build_bundle_model(profile,bundle,{'sampling_seed':2026})
assert len(model.support)==8*classes
assert {r['global_patient_id'] for r in model.support}.isdisjoint({r['global_patient_id'] for r in bundle.records('val')+bundle.records('test')})
route,agg,init=PROFILES[profile]
markers=[]
if route=='ZSMIL':markers.append('classifier' if init=='ZS' else 'classifier.weight')
elif route=='TaskRes':markers.append('classifier')
elif route=='CLIPAdapter':
 markers.extend(['adapter.0.weight','adapter.2.weight'])
 assert model.network.adapter[0].weight.shape==(dim//4,dim)
elif route=='TIPAdapter':
 markers.append('cache_keys')
 assert model.network.cache_keys.shape==(8*classes,dim)
 assert model.network.cache_values.shape==(8*classes,classes) and not model.network.cache_values.requires_grad
if agg not in {'BGAP','BGMP'}:
 candidates=[name for name,p in model.network.named_parameters() if name.startswith(agg+'.') and p.requires_grad and name.endswith('weight')]
 assert candidates
 # The released WiKG gate_U/V/W tensors are defined but unused by its
 # forward_attention. Verify the genuinely executed head projection instead.
 markers.append('WIKGMIL.W_head.weight' if agg=='WIKGMIL' else candidates[0])
before={name:model.network.state_dict()[name].clone() for name in markers}
parameters=sum(p.numel() for p in model.parameters());del model;gc.collect()
run=train(data,profile,out/'run',seed=2026,device=device,config={'threads':2})
assert not (run/'test_predictions.csv').exists()
cfg=read_json(run/'request.json')['config']
assert cfg['epochs']==cfg['patience']==20 and cfg['lr']==.001 and cfg['weight_decay']==1e-5
assert cfg['grad_clip'] is None and cfg['max_instances'] is None and cfg['selection_metric']=='miladapter_completed_epoch'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
extra=last['model']['_extra_state'];state=last['model']
assert last['epoch']==best['epoch']==19 and len(last['history'])==20
assert extra['completed_epochs']==20 and extra['optimizer_updates']==20*8*classes
assert extra['scheduler']['last_epoch']==20 and extra['scheduler']['_step_count']==21 and extra['scheduler']['T_max']==20
assert len(extra['train_orders'])==20 and all(len(order)==len(set(order))==8*classes for order in extra['train_orders'])
assert all(h['validation']=={'validation_performed':False} for h in last['history'][:-1])
assert last['history'][-1]['validation']['miladapter_completed_epoch']==20
assert all(not torch.equal(value,state['network.'+name]) for name,value in before.items()),markers
assert torch.equal(state['network.text_embeddings'],torch.from_numpy(bundle.miladapter.prototypes))
if route in {'CLIPAdapter','TIPAdapter'}:assert torch.equal(state['network.classifier'],state['network.text_embeddings'])
if route=='TIPAdapter':
 assert state['network.cache_values'].argmax(1).tolist()==[next(r['target'] for r in bundle.records('train') if r['slide_id']==slide) for slide in read_json(run/'method_protocol.json')['support_slides']]
optimizer=last['optimizers'][0]
assert len(last['optimizers'])==len(optimizer['param_groups'])==1
assert optimizer['param_groups'][0]['lr']==0 and optimizer['param_groups'][0]['initial_lr']==.001
assert optimizer['param_groups'][0]['weight_decay']==1e-5
assert len(optimizer['state'])==len(extra['active_parameter_names'])>0
assert all(int(v['step'])==20*8*classes for v in optimizer['state'].values())
protocol=read_json(run/'method_protocol.json');assert protocol['native_dimension']==dim and protocol['k_shots']==8
assert protocol['completed_epochs']==20 and protocol['optimizer_updates']==20*8*classes
assert protocol['prototype_provenance']['encoder']==encoder and protocol['prototype_provenance']['project']==project
assert protocol['prototype_provenance']['prompt_provenance_status']=='author_import_generation_details_undisclosed'
del last,best,state,extra,optimizer;gc.collect();torch.cuda.empty_cache()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
import pandas as pd
assert pd.read_csv(run/'test_predictions.csv').miladapter_patches.eq(257).all()
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
write_json(out/'validation.json',{'state':'passed','case':case,'profile':profile,'encoder':encoder,'project':project,
 'classes':classes,'feature_dimension':dim,'native_full_aggregator':agg,'adapter':route,'all_patches':257,'epochs':20,
 'k_shots':8,'support_slides':8*classes,'actual_optimizer_updates':20*8*classes,'parameters':parameters,
 'actual_author_prototype_sha256':bundle.miladapter.text_meta['sha256'],'all_core_markers_updated':True,
 'source_final_weights_only_verified':True,'full_cosine_AdamW_and_support_state_verified':True,'selected_checkpoint_reloaded_twice':True,
 'prediction_sha256':prediction,'metrics':metrics,'software':software_identity(),'synthetic':True,
 'device':torch.cuda.get_device_name(device),'wall_seconds':time.monotonic()-started,'formal_admission':False,
 'scope':'Complete original26configurations x native512CONCH/768KEEP; author text matrices with undisclosed generation details, explicit source defects and KEEP CLIP input port; full20epoch/k8/full257patch GPU recipe; no realWSI/FM-extraction/paper-score claim'})
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--devices', default='cuda:0')
    parser.add_argument('--workers-per-device', type=int, choices=[1, 2], default=1)
    args = parser.parse_args(); devices = args.devices.split(',')
    if len(set(devices)) != len(devices) or any(not re.fullmatch(r'cuda:\d+', d) for d in devices):
        raise ValueError('Select actual unique allocated CUDA devices')
    wheel, source, out = map(lambda p: Path(p).resolve(), [args.wheel, args.source_tree, args.output])
    out.mkdir(parents=True, exist_ok=False); target = out/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        if any(not (target/name).resolve().is_relative_to(target) for name in archive.namelist()):
            raise ValueError('Unsafe wheel member')
        archive.extractall(target)
    cases = [{'case': i, 'state': 'pending'} for i in range(52)]
    slots = [d for d in devices for _ in range(args.workers_per_device)]
    active, handles, cursor = {}, [], 0

    def save():
        (out/'process.json').write_text(json.dumps({'state': 'running', 'cases': cases,
            'formal_admission': False}, indent=2)+'\n')

    try:
        while cursor < 52 or active:
            for slot, device in enumerate(slots):
                if slot in active or cursor >= 52:
                    continue
                i = cursor; cursor += 1
                log = (out/f'case_{i}.log').open('x'); handles.append(log)
                child = subprocess.Popen([sys.executable, '-c', PROGRAM, str(target), str(out/f'case_{i}'), str(i), device],
                    cwd=source, env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
                    stdout=log, stderr=subprocess.STDOUT)
                cases[i].update(state='running', pid=child.pid, device=device); active[slot] = (i, child); save()
            for slot, (i, child) in list(active.items()):
                result = child.poll()
                if result is not None:
                    cases[i].update(state='passed' if result == 0 else 'failed', returncode=result)
                    del active[slot]; save()
            time.sleep(1)
        if any(c['state'] != 'passed' for c in cases):
            raise RuntimeError('A full native MIL-Adapter case failed; preserve all attempts, no reduced fallback')
        reports = [json.loads((out/f'case_{i}/validation.json').read_text()) for i in range(52)]
        assert len({(r['profile'], r['encoder']) for r in reports}) == 52
        assert all(r['state'] == 'passed' and r['software'] == reports[0]['software'] for r in reports)
        (out/'validation.json').write_text(json.dumps({'state': 'passed', 'profiles': reports,
            'software': reports[0]['software'], 'formal_admission': False, 'synthetic': True,
            'scope': 'All26actual original conditions x both native FM dimensions, full source20epochs/k8/full257patches; whole and final package gates separate'}, indent=2)+'\n')
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
