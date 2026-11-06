"""Four independent-wheel full512D4097patch/50epoch ConMIL GPU references.

All original attention/KAN widths, fixed actual CONCH text, TRAIN-only cap,
source Adam/cosine and validation selection are mandatory. Parallelism places
independent cases without changing any model, input or training budget.
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


PROGRAM = "\nimport gc,json,sys,time\nfrom pathlib import Path\ntarget,out=map(Path,sys.argv[1:3]);case=int(sys.argv[3]);device=sys.argv[4]\nsys.path.insert(0,str(target))\nimport torch,milplus\nfrom milplus.conmil_demo import create_conmil_demo\nfrom milplus.data import Bundle\nfrom milplus.engine import train,evaluate,seed_everything\nfrom milplus.models import build_bundle_model\nfrom milplus.io import read_json,write_json,digest_file\nfrom milplus.provenance import software_identity\nassert milplus.__version__=='0.79.0' and torch.cuda.is_available()\nassert Path(milplus.__file__).resolve().is_relative_to(target.resolve())\ntorch.cuda.set_device(device);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)\nprofile,classes,grid=[('conmil_pos_source',2,5),('conmil_posneg_source',2,5),\n ('conmil_posneg_source',2,6),('conmil_posneg_multiclass_repaired',7,5)][case]\nout.mkdir(parents=True,exist_ok=False);started=time.monotonic()\ndata=create_conmil_demo(out/'data',patients=5*classes,classes=classes,patches=4097)\nbundle=Bundle(data);n=len(bundle.records('train'));assert n==3*classes\nassert all(bundle.conmil.meta['records'][row['slide_id']]['count']==4097 for row in bundle.rows.to_dict('records'))\nassert bundle.dim==512 and bundle.conmil.text.positive.shape==(4,512) and bundle.conmil.text.negative.shape==(2,512)\nseed_everything(2026);model=build_bundle_model(profile,bundle,{'kan_grid':grid})\nmarkers=['alpha','feat_proj.0.weight','attention_V.weight','attention_U.weight','attention_W.weight','sim_proj.weight',\n 'classifier.net.0.base_weight','classifier.net.0.spline_weight','classifier.net.3.base_weight','classifier.net.3.spline_weight']\ninitial=model.network.state_dict();before={name:initial[name].clone() for name in markers}\nassert initial['feat_proj.0.weight'].shape==(256,512)\nassert initial['attention_V.weight'].shape==initial['attention_U.weight'].shape==(128,256)\nassert initial['classifier.net.0.spline_weight'].shape==(128,256,grid+3)\nassert initial['classifier.net.3.spline_weight'].shape==(classes,128,grid+3)\nparameters=sum(p.numel() for p in model.parameters())\nassert parameters==(494723 if grid==5 else 527747)+(classes-2)*128*(grid+4)\ninitial_grids={name:value.clone() for name,value in initial.items() if name.endswith('.grid')}\ndel model,initial;gc.collect()\nconfig={'threads':2,'model':{'kan_grid':grid}}\nrun=train(data,profile,out/'run',seed=2026,device=device,config=config)\nassert not (run/'test_predictions.csv').exists()\ncfg=read_json(run/'request.json')['config']\nassert cfg['epochs']==cfg['patience']==50 and cfg['lr']==2e-4 and cfg['weight_decay']==1e-5\nassert cfg['max_instances'] is None and cfg['grad_clip'] is None and cfg['selection_metric']=='conmil_val_accuracy'\nlast=torch.load(run/'last.pt',map_location='cpu',weights_only=False)\nbest=torch.load(run/'best.pt',map_location='cpu',weights_only=False)\nstate=last['model'];extra=state['_extra_state'];history=last['history']\nassert len(history)==50 and last['epoch']==49 and extra['completed_epochs']==50 and extra['optimizer_updates']==50*n\nassert extra['scheduler']['last_epoch']==50 and extra['scheduler']['_step_count']==51 and extra['scheduler']['T_max']==50\nassert len(extra['train_orders'])==len(extra['samples_history'])==50\nassert all(len(order)==n and set(order)=={r['slide_id'] for r in bundle.records('train')} for order in extra['train_orders'])\nassert all(len(samples)==n and all(len(sample['indices'])==len(set(sample['indices']))==4096 and min(sample['indices'])>=0 and max(sample['indices'])<4097\n for sample in samples) for samples in extra['samples_history'])\nassert all(not torch.equal(before[name],state['network.'+name]) for name in markers)\nassert all(torch.equal(value,state['network.'+name]) for name,value in initial_grids.items())\nassert torch.equal(state['positive_queries'],bundle.conmil.text.positive) and torch.equal(state['negative_queries'],bundle.conmil.text.negative)\naccuracy=[h['validation']['conmil_val_accuracy'] for h in history]\nassert best['epoch']==accuracy.index(max(accuracy))\nassert len(last['optimizers'])==1\noptimizer=last['optimizers'][0];assert len(optimizer['param_groups'])==1\nassert optimizer['param_groups'][0]['lr']==0 and optimizer['param_groups'][0]['initial_lr']==2e-4 and optimizer['param_groups'][0]['weight_decay']==1e-5\nassert all(int(value['step'])==50*n for value in optimizer['state'].values())\nassert len(optimizer['state'])==16\nprotocol=read_json(run/'method_protocol.json');assert protocol['optimizer_updates']==50*n and protocol['native']['kan_grid']==grid\nassert protocol['actual_text']['weights_sha256']=='40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c'\nassert protocol['actual_text']['encoding']=='original_CONCH_causal_padding_CLS_project512_L2_normalized'\nselected=best['epoch'];del last,best,state,extra,optimizer;gc.collect();torch.cuda.empty_cache()\nmetrics=evaluate(run,device=device);prediction=digest_file(run/'test_predictions.csv')\nassert evaluate(run,device=device)==metrics and digest_file(run/'test_predictions.csv')==prediction\nimport pandas as pd\nassert pd.read_csv(run/'test_predictions.csv').conmil_patches.eq(4097).all()\nfor name,module in list(sys.modules.items()):\n if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name\nwrite_json(out/'validation.json',{'state':'passed','case':case,'profile':profile,'classes':classes,'kan_grid':grid,\n 'feature_dimension':512,'hidden':256,'attention_hidden':128,'spline_order':3,'all_patches':4097,'TRAIN_patches':4096,'VAL_TEST_patches':4097,\n 'epochs':50,'batch_size':1,'train_slides':n,'actual_optimizer_updates':50*n,'parameters':parameters,\n 'actual_pretrained_CONCH_text_bound':True,'all_ten_core_mechanism_markers_updated':True,\n 'immutable_KAN_grids_and_text_queries_unchanged':True,'actual_full_cosine_and_Adam_steps_verified':True,\n 'selected_epoch':selected,'source_strict_ACC_selection_verified':True,'selected_checkpoint_reloaded_twice':True,\n 'prediction_sha256':prediction,'metrics':metrics,'software':software_identity(),'synthetic':True,\n 'device':torch.cuda.get_device_name(device),'wall_seconds':time.monotonic()-started,'formal_admission':False,\n 'scope':'Full original native512/256/128/guided-attention/cubicKAN/4097patch50epoch GPU recipe; cached text actually encoded with full original CONCH onCPU; no realWSI/FM-extraction/historical paper-score claim'})\n"


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
            raise RuntimeError('One or more complete ConMIL native cases failed; preserve and inspect logs, no reduced fallback')
        reports = [json.loads((output/f'case_{index}/validation.json').read_text()) for index in range(4)]
        assert len(reports) == 4 and all(row['state'] == 'passed' and row['software'] == reports[0]['software'] for row in reports)
        assert [(r['profile'], r['classes'], r['kan_grid']) for r in reports] == [
            ('conmil_pos_source', 2, 5), ('conmil_posneg_source', 2, 5),
            ('conmil_posneg_source', 2, 6), ('conmil_posneg_multiclass_repaired', 7, 5)]
        (output/'validation.json').write_text(json.dumps({'state': 'passed', 'profiles': reports,
            'software': reports[0]['software'], 'formal_admission': False, 'synthetic': True,
            'independent_workers_per_device': args.workers_per_device, 'efficiency_benchmark': False,
            'scope': 'All four independent-wheel actual native512D4097patch/50epoch ConMIL references; whole regression and installed final package gates remain separate'}, indent=2)+'\n')
    finally:
        for handle in handles: handle.close()


if __name__ == '__main__':
    main()
