"""Full native Swin filtering/cache and100epoch source aggregation/publicCLI acceptance."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from milplus.data import Bundle
from milplus.engine import train,evaluate
from milplus.io import digest_file,read_json,write_json
from milplus.models import build_bundle_model
from milplus.nciemil_filter import FilterInputs,WEIGHTS_SHA256
from milplus.nciemil_prepare import prepare
from milplus.provenance import software_identity


def raw_data(root,k,classes):
    root.mkdir();(root/'images').mkdir();rng=np.random.default_rng(8911+k);slides=[];splits=[];index=[]
    for subset,repeats in [('train',2),('val',1),('test',1)]:
        for target in range(classes):
            for repeat in range(repeats):
                slide=f'{subset}_class{target}_patient{repeat}';p=root/'images'/(slide+'.h5');n=2*k+1
                with h5py.File(p,'w') as f:
                    pixels=f.create_dataset('pixels',shape=(n,256,256,3),dtype=np.uint8,chunks=(1,256,256,3))
                    for start in range(0,n,128):pixels[start:min(start+128,n)]=rng.integers(0,256,(min(128,n-start),256,256,3),dtype=np.uint8)
                    f.create_dataset('patch_ids',data=np.array([slide+f'_patch{i}' for i in range(n)],dtype=h5py.string_dtype()))
                slides.append({'slide_id':slide,'patient_id':slide,'global_patient_id':'native:'+slide,'label':f'class{target}'})
                splits.append({'patient_id':slide,'subset':subset});index.append({'slide_id':slide,'path':str(p.relative_to(root))})
    pd.DataFrame(slides).to_csv(root/'slides.csv',index=False);pd.DataFrame(splits).to_csv(root/'split.csv',index=False)
    pd.DataFrame(index).to_csv(root/'images.csv',index=False)
    config={'task_id':f'nciemil_native_k{k}_classes{classes}','schema_version':1,'task_type':'classification','synthetic':True,'classes':[f'class{i}' for i in range(classes)],
        'defaults':{'fm':'rgb','split':'fold0'},'splits':{'fold0':'split.csv'},'encoders':{'rgb':{'dimension':3,'index':'images.csv',
        'nciemil_raw':{'schema_version':1,'kind':'nciemil_rgb256_uint8_hdf5','image_provenance':'synthetic nativeRGB256 validation',
        'patch_provenance':'all original independently generated2K+1 patches retained','weak_targets':{f'class{i}':int(i>0) for i in range(classes)}}}}}
    (root/'task.yaml').write_text(yaml.safe_dump(config));return root


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert args.device.startswith('cuda') and torch.cuda.is_available(),'Actual native full pipeline requires allocated CUDA'
    identity=software_identity();results=[];state={'state':'running','pid':os.getpid(),'conditions':results,'formal_admission':False}
    write_json(out/'process.json',state)
    pretrain={'epochs':3,'lr':2e-4,'weight_decay':1e-5,'bag_batch_size':2,'seed':3407}
    cfg=out/'pretrain.yaml';cfg.write_text(yaml.safe_dump(pretrain));aggregation=out/'aggregation.yaml';aggregation.write_text('threads: 2\n')
    for k,classes in [(128,4),(512,2)]:
        data=raw_data(out/f'raw_k{k}',k,classes);pipeline=out/f'pipeline_k{k}'
        first=prepare(data,args.weights,pipeline,pretrain,k,args.device,stop_after=1)
        assert first['state']=='interrupted' and read_json(pipeline/'filter/status.json')['epochs']==1
        partial_sha=digest_file(pipeline/'filter/last.pt');gc.collect();torch.cuda.empty_cache()
        command=[sys.executable,'-m','milplus','prepare-nciemil','--data',str(data),'--weights',args.weights,
            '--output',str(pipeline),'--pretrain-config',str(cfg),'--k',str(k),'--device',args.device,'--resume']
        subprocess.run(command,check=True);bundle=Bundle(pipeline/'dataset')
        assert bundle.nciemil.k==k and bundle.dim==768 and len(bundle.rows)==4*classes
        filter_state=torch.load(pipeline/'filter/last.pt',map_location='cpu',weights_only=False)
        assert filter_state['completed_epochs']==3 and filter_state['updates']==3*classes
        assert len(filter_state['optimizer']['state'])>100
        assert filter_state['config']['initial_weights_sha256']==WEIGHTS_SHA256
        assert all({t['slide_id'] for t in e['traces']}=={r['slide_id'] for r in bundle.records('train')} for e in filter_state['history'])
        assert all(t['patches_scanned']==2*k+1 for e in filter_state['history'] for t in e['traces'])
        del filter_state;gc.collect()
        for profile in ['nciemil_full_reference_released_kv','nciemil_full_reference_independent_kv_repaired']:
            name=profile+f'_k{k}';run=out/name
            train(pipeline/'dataset',profile,run,seed=8912,device=args.device,config={'threads':2},stop_after=3)
            assert read_json(run/'status.json')['state']=='interrupted'
            subprocess.run([sys.executable,'-m','milplus','run','--data',str(pipeline/'dataset'),'--method',profile,
                '--output',str(run),'--seed','8912','--device',args.device,'--config',str(aggregation),'--resume'],check=True)
            last=torch.load(run/'last.pt',map_location='cpu',weights_only=False);best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
            extra=last['model']['_extra_state'];history=last['history'];cfg_run=read_json(run/'request.json')['config']
            assert cfg_run['epochs']==extra['completed_epochs']==len(history)==100 and cfg_run['lr']==2e-4 and cfg_run['weight_decay']==1e-5
            assert cfg_run['max_instances'] is None and cfg_run['grad_clip'] is None
            assert extra['optimizer_updates']==100*math.ceil(2*classes/4)
            assert set(extra['gradient_updates'].values())=={extra['optimizer_updates']}
            assert last['optimizers'][0]['state']=={} and last['optimizers'][0]['param_groups'][0]['momentum']==0
            curve=[e['validation']['nciemil_val_harmonic'] for e in history]
            assert best['epoch']==max(i for i,v in enumerate(curve) if v==max(curve))
            restore=build_bundle_model(profile,bundle);restore.configure_training(cfg_run);optimizer=restore.make_optimizer(2e-4,1e-5)
            restore.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizers'][0]);restore.validate_optimizer_state()
            assert restore.core.network._kv.out_features==(1024 if profile.endswith('released_kv') else 2048)
            del restore,optimizer,last,best;gc.collect();torch.cuda.empty_cache()
            metrics=evaluate(run,device=args.device);pred=digest_file(run/'test_predictions.csv')
            assert evaluate(run,device=args.device)==metrics and digest_file(run/'test_predictions.csv')==pred
            assert software_identity()==identity
            results.append({'profile':profile,'state':'passed','k':k,'classes':classes,'input_dimension':768,'hidden':1024,'heads':4,
                'raw_patches_per_slide':2*k+1,'selected_patches_per_slide':2*k,'pretraining':pretrain,
                'pretraining_interruption_checkpoint_sha256':partial_sha,'aggregation_epochs':100,
                'actual_aggregation_updates':100*math.ceil(2*classes/4),'selected_reload_twice':True,
                'prediction_sha256':pred,'metrics':metrics,'software':identity,'synthetic':True})
            write_json(out/'process.json',state)
    state.update(state='passed',software=identity,real_wsi_validation='not_run',paper_reproduction='not_run')
    write_json(out/'validation.json',state);write_json(out/'process.json',state)


if __name__=='__main__':main()
