"""Four complete source default raw256/K32/batch4/20epoch installed-wheel cases."""
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
from milplus.center_embeddings_demo import create_center_embeddings_demo
from milplus.engine import train,evaluate,seed_everything
from milplus.data import Bundle
from milplus.models import build_bundle_model
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.81.0' and torch.cuda.is_available()
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
torch.cuda.set_device(device);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
profile,patches,test_patches=[('center_embeddings_bag_classifier_repaired',33,33),
 ('center_embeddings_soft_assignment_repaired',33,33),
 ('center_embeddings_bag_classifier_repaired',65,1),('center_embeddings_soft_assignment_repaired',65,1)][case]
out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
data=create_center_embeddings_demo(out/'data',patients=60,patches=patches,test_patches=test_patches)
bundle=Bundle(data);assert len(bundle.records('train'))==len(bundle.records('val'))==len(bundle.records('test'))==20
assert bundle.dim==3 and all(r['shape'][1:]==[256,256,3] for r in bundle.center_embeddings.meta['slides'].values())
seed_everything(2026);model=build_bundle_model(profile,bundle)
assert model.train_batch_size==4 and sum(p.numel() for p in model.parameters())==21917897
assert model.network.features.layer4[2].conv2.weight.shape==(512,512,3,3)
assert model.network.bag_module[0].weight.shape==(128,32,3,3)
assert model.network.pooling.embedder[0].weight.shape==(512,512)
assert model.center_loss.centers.shape==(2,512)
markers=['network.features.conv1.weight','network.features.layer4.2.conv2.weight','network.p2.0.weight',
 'network.p3.0.weight','network.p4.0.weight','network.bag_module.0.weight','network.bag_module.6.weight',
 'network.bag_module.9.weight','network.pooling.embedder.0.weight','network.pooling.classifier.weight',
 'network.bag_classifier.0.weight','center_loss.centers']
initial=model.state_dict();before={name:initial[name].clone() for name in markers}
assert all(int(value)==0 for name,value in initial.items() if name.endswith('num_batches_tracked'))
parameter_counts=[len(list(model.network.parameters())),len(list(model.center_loss.parameters()))]
del model,initial;gc.collect()
run=train(data,profile,out/'run',seed=2026,device=device,config={'threads':2})
assert not (run/'test_predictions.csv').exists()
cfg=read_json(run/'request.json')['config']
assert cfg['epochs']==cfg['patience']==20 and cfg['lr']==1e-4 and cfg['weight_decay']==1e-4
assert cfg['grad_clip'] is None and cfg['max_instances'] is None and cfg['selection_metric']=='center_embeddings_val_balanced_accuracy'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
state=last['model'];extra=state['_extra_state'];history=last['history']
assert last['epoch']==19 and len(history)==20 and extra['completed_epochs']==20 and extra['optimizer_updates']==100
assert all(not torch.equal(before[name],state[name]) for name in markers)
assert all(int(value)==100 for name,value in state.items() if name.endswith('num_batches_tracked'))
assert len(extra['train_selections'])==20
for trace in extra['train_selections']:
 assert len(trace['bag_order'])==len(set(trace['bag_order']))==20 and set(trace['scanned'].values())=={patches}
 assert set(trace['selection'])==set(trace['bag_order'])==set(trace['scanned'])
 assert all(len(ids)==len(set(ids))==32 and min(ids)>=0 and max(ids)<patches for ids in trace['selection'].values())
assert len(last['optimizers'])==2
network_opt,center_opt=last['optimizers']
assert [g['lr'] for g in network_opt['param_groups']]==[.001*.1**3,.01*.1**3]
assert all(g['weight_decay']==1e-4 for g in network_opt['param_groups'])
assert center_opt['param_groups'][0]['lr']==1e-4 and center_opt['param_groups'][0]['weight_decay']==0
assert [len(network_opt['state']),len(center_opt['state'])]==parameter_counts
assert all(int(value['step'])==100 for o in last['optimizers'] for value in o['state'].values())
accuracy=[h['validation']['center_embeddings_val_balanced_accuracy'] for h in history]
assert best['epoch']==max(i for i,v in enumerate(accuracy) if v==max(accuracy))
protocol=read_json(run/'method_protocol.json')
assert protocol['input_pixels']==256 and protocol['selected_k']==32 and protocol['train_batch_size']==4
assert protocol['optimizer_updates']==100 and protocol['completed_epochs']==20
assert bundle.center_embeddings.meta['assets']['resnet34-333f7ec4.pth']=='333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef'
del last,best,state,extra,network_opt,center_opt;gc.collect();torch.cuda.empty_cache()
metrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction
import pandas as pd
frame=pd.read_csv(run/'test_predictions.csv')
assert len(frame)==20 and frame.center_candidate_count.eq(test_patches).all()
assert frame.center_selected_count.eq(min(32,test_patches)).all()
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
write_json(out/'validation.json',{'state':'passed','case':case,'profile':profile,'RGB_pixels':256,'native_K':32,
 'candidate_patches':patches,'TEST_patches':test_patches,'actual_ImageNet333f7ec4_bound':True,
 'epochs':20,'TRAIN_slides':20,'VAL_slides':20,'TEST_slides':20,'TRAIN_batch_size':4,'actual_optimizer_updates':100,
 'parameters':21917897,'full_source_R34_FPN_bagCNN_and_shared512embedding':True,'all_twelve_mechanism_markers_updated':True,
 'three_source_losses_and_both_optimizers_verified':True,'all_native_BN_counters_verified':True,
 'source_exhaustive_scans_top32_order_and_literal_lr_resets_verified':True,'source_last_tie_balanced_ACC_selection_verified':True,
 'selected_checkpoint_reloaded_twice':True,'prediction_sha256':prediction,'metrics':metrics,'software':software_identity(),
 'device':torch.cuda.get_device_name(device),'wall_seconds':time.monotonic()-started,'synthetic':True,'formal_admission':False,
 'scope':'Four full original20epoch/batch4/K32/raw256/R34/FPN/sharedembedding/centerjoint native references, original bag classifier and Student-t inference; no frozen FM shortcut, realWSI or paper-score claim'})
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--devices', default='cuda:0')
    parser.add_argument('--workers-per-device', type=int, choices=[1, 2], default=1)
    args = parser.parse_args(); devices = args.devices.split(',')
    if len(set(devices)) != len(devices) or any(not re.fullmatch(r'cuda:\d+', d) for d in devices):
        raise ValueError('Choose unique actual allocated CUDA devices')
    wheel, source, out = map(lambda x: Path(x).resolve(), [args.wheel, args.source_tree, args.output])
    out.mkdir(parents=True, exist_ok=False); target = out/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        if any(not (target/name).resolve().is_relative_to(target) for name in archive.namelist()):
            raise ValueError('Unsafe wheel member')
        archive.extractall(target)
    cases = [{'case': i, 'state': 'pending'} for i in range(4)]
    slots = [d for d in devices for _ in range(args.workers_per_device)]
    active, handles, cursor = {}, [], 0
    def save():
        (out/'process.json').write_text(json.dumps({'state': 'running', 'cases': cases, 'formal_admission': False}, indent=2)+'\n')
    try:
        while cursor < 4 or active:
            for slot, device in enumerate(slots):
                if slot in active or cursor >= 4:
                    continue
                i = cursor; cursor += 1; log = (out/f'case_{i}.log').open('x'); handles.append(log)
                child = subprocess.Popen([sys.executable, '-c', PROGRAM, str(target), str(out/f'case_{i}'), str(i), device],
                    cwd=source, env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
                    stdout=log, stderr=subprocess.STDOUT)
                cases[i].update(state='running', pid=child.pid, device=device); active[slot] = (i, child); save()
            for slot, (i, child) in list(active.items()):
                code = child.poll()
                if code is not None:
                    cases[i].update(state='passed' if code == 0 else 'failed', returncode=code); del active[slot]; save()
            time.sleep(1)
        if any(c['state'] != 'passed' for c in cases):
            raise RuntimeError('Full original Center Embeddings native case failed; preserve evidence and budget')
        reports = [json.loads((out/f'case_{i}/validation.json').read_text()) for i in range(4)]
        assert all(r['state'] == 'passed' and r['software'] == reports[0]['software'] for r in reports)
        assert len({(r['profile'], r['candidate_patches']) for r in reports}) == 4
        (out/'validation.json').write_text(json.dumps({'state': 'passed', 'profiles': reports,
            'software': reports[0]['software'], 'formal_admission': False, 'synthetic': True,
            'scope': 'All four full source native budgets; complete regression and final installed package gates remain separate'}, indent=2)+'\n')
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
