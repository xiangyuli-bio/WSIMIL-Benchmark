"""Four independent-wheel full512RGB/200epoch original compiled CPU cases.

All complete ResNet18 layers, sparse convolutions, five sampled tiles and Adam
updates are mandatory. CPU reference validation makes no CUDA/performance or
clinical/paper-score claim. Independent processes change placement only.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


PROGRAM = r'''
import gc,json,sys,time
from pathlib import Path
target,out=map(Path,sys.argv[1:3]);case=int(sys.argv[3])
sys.path.insert(0,str(target))
import torch,milplus
from milplus.sparseconvmil_demo import create_sparseconvmil_demo
from milplus.data import Bundle
from milplus.engine import train,evaluate,seed_everything
from milplus.models import build_bundle_model
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity
assert milplus.__version__=='0.78.0'
assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
torch.set_num_threads(4)
profile,classes=[('sparseconvmil_raw_source_c32',2),('sparseconvmil_raw_source_c32',3),
 ('sparseconvmil_raw_source_c128',2),('sparseconvmil_raw_source_c128',3)][case]
out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
data=create_sparseconvmil_demo(out/'data',patients=5*classes,classes=classes,patches=16,patch_pixels=512)
bundle=Bundle(data);n=len(bundle.records('train'));assert n==3*classes
assert all(bundle.sparseconvmil.inventory(row)['shape']==[16,512,512,3] for row in bundle.rows.to_dict('records'))
seed_everything(2026);model=build_bundle_model(profile,bundle)
channels=32 if profile.endswith('_c32') else 128
initial=model.network.state_dict()
markers=['tile_embedder.conv1.weight','tile_embedder.layer4.1.conv2.weight',
 'sparse_cnn.0.weight','sparse_cnn.2.weight','wsi_embedding_classifier.0.weight','wsi_embedding_classifier.2.weight']
before={name:initial[name].clone() for name in markers}
assert initial['tile_embedder.conv1.weight'].shape==(64,3,7,7)
assert initial['tile_embedder.layer4.1.conv2.weight'].shape==(512,512,3,3)
assert initial['sparse_cnn.0.weight'].shape==(9,1,512,channels)
assert initial['sparse_cnn.2.weight'].shape==(9,1,channels,channels)
assert initial['wsi_embedding_classifier.0.weight'].shape==(32,channels)
assert initial['wsi_embedding_classifier.2.weight'].shape==(classes,32)
parameters=sum(p.numel() for p in model.parameters())
backend=model.backend_identity
del model,initial;gc.collect()
run=train(data,profile,out/'run',seed=2026,config={'threads':4})
assert not (run/'test_predictions.csv').exists()
cfg=read_json(run/'request.json')['config']
assert cfg['epochs']==cfg['patience']==200 and cfg['lr']==.002 and cfg['weight_decay']==1e-6
assert cfg['model']=={} and cfg['max_instances'] is None and cfg['grad_clip'] is None
assert cfg['selection_metric']=='sparseconvmil_completed_epoch'
last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
state=last['model'];extra=state['_extra_state'];history=last['history']
updates=200*((n+1)//2)
assert len(history)==200 and last['epoch']==best['epoch']==199 and extra['completed_epochs']==200
assert extra['optimizer_updates']==updates and len(extra['samples_history'])==200
assert all(len(order)==n and set(order)=={r['slide_id'] for r in bundle.records('train')} for order in extra['train_orders'])
assert all(len(samples)==n and all(len(sample['indices'])==5 and min(sample['indices'])>=0 and max(sample['indices'])<16
 for sample in samples) for samples in extra['samples_history'])
assert all(not torch.equal(before[name],state['network.'+name]) for name in markers)
counts=[int(v) for key,v in state.items() if key.endswith('num_batches_tracked')]
assert counts and all(count==updates for count in counts)
for record in history:
 assert record['train']['sampled_tiles_per_slide']==5
 assert record['validation']['sparseconvmil_completed_epoch']==record['epoch']+1
 assert __import__('math').isfinite(record['train']['loss'])
assert len(last['optimizers'])==1
optimizer=last['optimizers'][0];assert len(optimizer['param_groups'])==1
assert optimizer['param_groups'][0]['lr']==.002 and optimizer['param_groups'][0]['weight_decay']==1e-6
assert all(int(v['step'])==updates for v in optimizer['state'].values())
protocol=read_json(run/'method_protocol.json');assert protocol['optimizer_updates']==updates
assert protocol['tiles_per_slide']==5 and protocol['batch_size']==2 and protocol['downsample']==10
assert protocol['backend']==backend and backend['backend']=='original_compiled_sparseconvnet'
assert all(torch.equal(state[k],best['model'][k]) for k in state if isinstance(state[k],torch.Tensor))
del last,best,state,extra,optimizer;gc.collect()
metrics=evaluate(run);prediction=digest_file(run/'test_predictions.csv')
assert evaluate(run)==metrics and digest_file(run/'test_predictions.csv')==prediction
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
write_json(out/'validation.json',{'state':'passed','case':case,'profile':profile,'classes':classes,
 'pixels':512,'all_patches':16,'sampled_tiles_per_slide':5,'sampling':'original with replacement',
 'epochs':200,'batch_size':2,'train_slides':n,'actual_optimizer_updates':updates,'channels':channels,
 'architecture':'full trained ResNet18, two actual compiled sparse convolutions and source classifier',
 'parameters':parameters,'backend':backend,'all_six_mechanism_markers_updated':True,
 'BN_complete_source_steps_verified':True,'final_weights_independent_of_VAL_verified':True,
 'selected_checkpoint_reloaded_twice':True,'prediction_sha256':prediction,'metrics':metrics,
 'software':software_identity(),'synthetic':True,'device':'cpu','wall_seconds':time.monotonic()-started,
 'formal_admission':False,'scope':'Full512RGB200epoch public-source recipe; no reduced/frozen/dense fallback and no realWSI/CUDA/historical paper-score claim'})
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['wheel', 'source-tree', 'output']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    wheel, source, output = [Path(value).resolve() for value in [args.wheel, args.source_tree, args.output]]
    if output.exists():
        raise ValueError('Preserve native attempts; select a new output root')
    output.mkdir(parents=True); target = output/'wheel_import'; target.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not (target/name).resolve().is_relative_to(target):
                raise ValueError('Unsafe wheel member')
        archive.extractall(target)
    cases = [{'case': index, 'state': 'pending'} for index in range(4)]
    active, handles, cursor = {}, [], 0

    def save():
        (output/'process.json').write_text(json.dumps({'state': 'running', 'cases': cases,
            'formal_admission': False, 'workers': args.workers}, indent=2)+'\n')

    try:
        while cursor < 4 or active:
            for slot in range(args.workers):
                if slot in active or cursor >= 4:
                    continue
                index = cursor; cursor += 1
                log = (output/f'case_{index}.log').open('x'); handles.append(log)
                command = [sys.executable, '-c', PROGRAM, str(target), str(output/f'case_{index}'), str(index)]
                child = subprocess.Popen(command, cwd=source,
                    env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                         'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}, stdout=log, stderr=subprocess.STDOUT)
                cases[index].update(state='running', pid=child.pid)
                active[slot] = (index, child); save()
            for slot, (index, child) in list(active.items()):
                code = child.poll()
                if code is not None:
                    cases[index].update(state='passed' if code == 0 else 'failed', returncode=code)
                    del active[slot]; save()
            time.sleep(1)
        if any(case['state'] != 'passed' for case in cases):
            raise RuntimeError('A complete native SparseConvMIL case failed; preserve evidence, no reduced fallback')
        reports = [json.loads((output/f'case_{index}/validation.json').read_text()) for index in range(4)]
        assert all(row['state'] == 'passed' and row['software'] == reports[0]['software'] for row in reports)
        assert [(r['profile'], r['classes']) for r in reports] == [
            ('sparseconvmil_raw_source_c32', 2), ('sparseconvmil_raw_source_c32', 3),
            ('sparseconvmil_raw_source_c128', 2), ('sparseconvmil_raw_source_c128', 3)]
        (output/'validation.json').write_text(json.dumps({'state': 'passed', 'profiles': reports,
            'software': reports[0]['software'], 'formal_admission': False, 'synthetic': True,
            'device': 'cpu', 'efficiency_benchmark': False,
            'scope': 'All four full512RGB200epoch independent installed-wheel source references; whole/final gates separate'}, indent=2)+'\n')
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
