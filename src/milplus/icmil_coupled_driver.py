"""Complete extract -> MIL -> encoder distillation -> re-extract -> MIL loop."""
import hashlib
import os
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import torch

from .data import Bundle
from .icmil_coupled_data import CoupledRGB
from .icmil_coupled_source import BagModel,backbone
from .icmil_coupled_stages import extract
from .icmil_coupled_training import Progress,Interrupted,bag_phase,distillation_phase,load,save,exists,checkpoint_path
from .io import digest_file,read_json,write_json
from .provenance import software_identity

PROFILES={'icmil_coupled_abmil_patient_reference':'abmil','icmil_coupled_maxpool_patient_reference':'maxpool',
    'icmil_coupled_dtfd_legacy14_patient_reference':'dtfd'}


def _lock(out,resume):
    path=out/'RUNNING.lock'
    if path.exists() and resume:
        prior=read_json(path)
        if prior['hostname']!=socket.gethostname():raise ValueError('Verify the previous host process is terminal before reclaiming its ICMIL run lock')
        proc=Path('/proc')/str(prior['pid'])
        if proc.exists() and (proc/'stat').read_text().split()[21]==prior['start_ticks']:
            raise ValueError('Original ICMIL run process is still live')
        path.unlink()
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:
        import json
        json.dump({'pid':os.getpid(),'hostname':socket.gethostname(),'start_ticks':Path('/proc/self/stat').read_text().split()[21]},f)
    return path


def _inputs(data,fm=None,split=None):
    bundle=Bundle(data,fm,split)
    if getattr(bundle,'icmil_coupled',None) is None:raise ValueError('Prepare real RGB256 inputs and original weights in encoder.icmil_coupled.manifest')
    inventory=bundle.check(deep=True)
    return bundle,bundle.icmil_coupled,inventory


def train(data,method,output,fm=None,split=None,seed=2026,device='cpu',config=None,resume=False,stop_after=None):
    from .engine import seed_everything
    from .registry import inspect_method
    profile=method if method in PROFILES else inspect_method(method)['id']
    if profile not in PROFILES:raise ValueError('Choose a declared complete coupled ICMIL branch')
    supplied=config or {};model=supplied.get('model',{})
    if set(supplied)-{'epochs','threads','model'} or not isinstance(model,dict) or set(model)-{'iterations','distill_epochs'}:
        raise ValueError('ICMIL preserves original1024/512/128 widths, full images, source Adam/constantLR and threeKL; only explicit stage budgets/iterations/threads are configurable')
    cfg={'bag_epochs':supplied.get('epochs',200),'distill_epochs':model.get('distill_epochs',200),
        'iterations':model.get('iterations',1),'threads':supplied.get('threads',2)}
    if any(type(v) is not int or v<1 for v in cfg.values()) or type(seed) is not int or (stop_after is not None and (type(stop_after) is not int or stop_after<1)):
        raise ValueError('Declare positive integer native phase budgets/iterations/threads and an integer seed')
    bundle,assets,inventory=_inputs(data,fm,split);out=Path(output).expanduser().resolve()
    request={'backend':'icmil_coupled_complete_reference','method':profile,'kind':PROFILES[profile],
        'data':str(bundle.root),'fm':bundle.fm,'split':bundle.split,'seed':seed,'device':str(device),'config':cfg,
        'software':software_identity(),'bundle_sha256':bundle.fingerprint,
        'feature_inventory_sha256':inventory['feature_inventory_sha256'],'raw_contract':assets.contract,
        'initial_weights_sha256':digest_file(assets.weights),'synthetic':bool(bundle.config.get('synthetic',False)),
        'protocol':'explicit patient-disjoint reference; original mechanisms and step ordering; no historical paper-score claim',
        'test_labels_used_before_explicit_evaluate':False,'formal_admission':False}
    if resume:
        if read_json(out/'request.json')!=request:raise ValueError('ICMIL resume changed software, real images, partitions, original initialization, method or stage budgets')
    else:out.mkdir(parents=True,exist_ok=False);write_json(out/'request.json',request)
    lock=_lock(out,resume)
    try:
        seed_everything(seed);torch.set_num_threads(cfg['threads']);dev=torch.device(device)
        if dev.type=='cuda' and not torch.cuda.is_available():raise ValueError('Actual requested coupled ICMIL CUDA runtime is unavailable; no CPU fallback')
        if dev.type not in {'cpu','cuda'}:raise ValueError('Use explicit CPU or CUDA')
        progress=Progress(out,stop_after);encoder_state=None;previous_bag_state=None
        for iteration in range(cfg['iterations']+1):
            encoder=backbone(assets.weights).to(dev)
            if encoder_state is not None:encoder.load_state_dict(encoder_state,strict=True)
            encoder_path=out/('encoder_'+str(iteration)+'.pt')
            if not exists(encoder_path):save({'encoder':{k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()}},encoder_path)
            else:
                recorded=load(encoder_path)['encoder']
                if any(not torch.equal(v.cpu(),recorded[k]) for k,v in encoder.state_dict().items()):raise ValueError('Saved ICMIL extraction encoder differs from the selected previous phase')
            index=extract(bundle,assets,encoder,out/('features_'+str(iteration)),dev)
            initial_bag=previous_bag_state if iteration>0 and PROFILES[profile] in {'maxpool','dtfd'} else None
            selected=bag_phase(PROFILES[profile],index,out/('bag_'+str(iteration)),cfg['bag_epochs'],dev,progress,initial_state=initial_bag)
            previous_bag_state=load(selected)['model']
            if iteration==cfg['iterations']:
                final_encoder=checkpoint_path(encoder_path)
                write_json(out/'selected.json',{'encoder':str(final_encoder),'encoder_sha256':digest_file(final_encoder),
                    'bag':str(selected),'bag_sha256':digest_file(selected),'iteration':iteration});break
            bag=BagModel(PROFILES[profile]);bag.load_state_dict(previous_bag_state,strict=True)
            refined=distillation_phase(bundle,assets,encoder.state_dict(),bag,out/('distill_'+str(iteration)),cfg['distill_epochs'],dev,progress)
            encoder_state=load(refined)['encoder']
        if bundle.check(deep=True)['feature_inventory_sha256']!=inventory['feature_inventory_sha256']:
            raise ValueError('Original coupled ICMIL input changed during full pipeline')
        write_json(out/'status.json',{'state':'completed','iterations':cfg['iterations'],'MIL_phases':cfg['iterations']+1,
            'encoder_distillation_phases':cfg['iterations'],'test_evaluated':False,'formal_admission':False})
        return out
    except Interrupted:
        previous=read_json(out/'status.json');write_json(out/'status.json',{**previous,'state':'interrupted'});return out
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','error':repr(error)});raise
    finally:lock.unlink(missing_ok=True)


def evaluate(run,subset='test',device='cpu'):
    from .engine import seed_everything,score
    out=Path(run).resolve();request=read_json(out/'request.json')
    if subset not in {'val','test'} or read_json(out/'status.json')['state']!='completed':raise ValueError('Complete every coupled ICMIL stage before evaluation')
    if request['software']!=software_identity():raise ValueError('Original coupled ICMIL runtime/source changed')
    bundle,assets,inventory=_inputs(request['data'],request['fm'],request['split'])
    if bundle.fingerprint!=request['bundle_sha256'] or inventory['feature_inventory_sha256']!=request['feature_inventory_sha256']:
        raise ValueError('Original coupled ICMIL inputs changed')
    seed_everything(request['seed']);torch.set_num_threads(request['config']['threads']);dev=torch.device(device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise ValueError('Requested CUDA unavailable; no CPU fallback')
    selected=read_json(out/'selected.json')
    for key in ['encoder','bag']:
        if digest_file(selected[key])!=selected[key+'_sha256']:raise ValueError('Selected full coupled ICMIL checkpoint changed')
    encoder=backbone(assets.weights).to(dev);encoder.load_state_dict(load(selected['encoder'])['encoder'],strict=True)
    index=extract(bundle,assets,encoder,out/('evaluation_features_'+subset),dev,subsets=(subset,))
    model=BagModel(request['kind']).to(dev);model.load_state_dict(load(selected['bag'])['model'],strict=True);model.eval()
    # A fixed fresh-evaluation permutation is explicit. Repeated TEST queries
    # never consume a training loader or change selected parameters.
    rows={r['slide_id']:r for r in bundle.records(subset)};predictions=[]
    with torch.no_grad():
        for record in index['rows']:
            x=np.load(record['path'],allow_pickle=False).copy();np.random.shuffle(x)
            p=torch.softmax(model(torch.from_numpy(x).to(dev)),dim=-1).squeeze(0).cpu().numpy()
            row=rows[record['slide_id']];predictions.append({**{k:row[k] for k in ['slide_id','patient_id','global_patient_id','target']},
                'subset':subset,'source_slides':row['slide_id'],'prediction':int(p.argmax()),'p_0':float(p[0]),'p_1':float(p[1])})
    if bundle.check(deep=True)['feature_inventory_sha256']!=inventory['feature_inventory_sha256']:raise ValueError('Original ICMIL images changed during evaluation')
    frame=pd.DataFrame(predictions);metrics=score(bundle,frame)
    frame.to_csv(out/(subset+'_predictions.csv'),index=False);write_json(out/(subset+'_metrics.json'),metrics)
    return metrics
