"""Raw max-MIL -> actual feature extraction -> two-level AP -> full PMIL fit."""
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import Bundle
from .icmil_coupled_driver import _lock
from .icmil_coupled_training import Progress,Interrupted,save,load,exists,checkpoint_path
from .io import digest_file,digest_object,read_json,write_json
from .pmil_prototypical_clustering import discover
from .pmil_prototypical_source import encoder,bag_model,bag_objective
from .pmil_prototypical_training import Patches,FeatureBags,encoder_phase,export_encoder,bag_phase
from .provenance import software_identity

PROFILE='pmil_prototypical_raw_patient_reference'


def inputs(data,fm=None,split=None):
    bundle=Bundle(data,fm,split)
    if getattr(bundle,'pmil_prototypical',None) is None:
        raise ValueError('Prepare PMIL actual RGB256 inputs and original ImageNet ResNet34 weights')
    return bundle,bundle.pmil_prototypical,bundle.check(deep=True)


@torch.no_grad()
def extract(bundle,assets,selected_encoder,output,device,subsets=('train','val')):
    from .engine import rng_state,restore_rng
    out=Path(output);out.mkdir(parents=True,exist_ok=True);index=out/'index.pt'
    rows=[r for subset in subsets for r in bundle.records(subset)]
    contract=digest_object({'raw':assets.contract,'bundle':bundle.fingerprint,'encoder':digest_file(selected_encoder),
        'subsets':list(subsets),'slides':[r['slide_id'] for r in rows],'source':software_identity()})
    if exists(index):
        state=load(index)
        if state['contract']!=contract:raise ValueError('PMIL extraction input/selected encoder/source changed')
        for row in state['rows']:
            if digest_file(row['path'])!=row['sha256']:raise ValueError('PMIL learned feature cache changed')
        restore_rng(state['rng']);return state['rows']
    model=encoder(assets.weights).to(device)
    model.load_state_dict(torch.load(selected_encoder,map_location='cpu',weights_only=True)['state_dict'],strict=True);model.eval()
    dataset=Patches(assets,rows);buffers=[];result=[]
    for row in rows:
        path=out/(digest_object(row['slide_id'])+'.'+uuid.uuid4().hex+'.npy')
        buffers.append(np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(assets.count(row),512)))
        result.append({'slide_id':row['slide_id'],'subset':row['subset'],'target':row['target'],
            'global_patient_id':row['global_patient_id'],'path':str(path),'patches':assets.count(row)})
    slide_index=0;offset=0;seen=0
    for x,_ in DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0):
        _,value=model(x.to(device));value=value.cpu().numpy()
        if value.dtype!=np.float32 or value.shape!=(len(x),512) or not np.isfinite(value).all():
            raise FloatingPointError('PMIL actual learned encoder output is not finite512D float32')
        start=0
        while start<len(value):
            n=min(len(value)-start,len(buffers[slide_index])-offset)
            buffers[slide_index][offset:offset+n]=value[start:start+n];offset+=n;start+=n;seen+=n
            if offset==len(buffers[slide_index]):
                buffers[slide_index].flush();slide_index+=1;offset=0
    if seen!=len(dataset) or slide_index!=len(rows) or offset:
        raise ValueError('PMIL full encoder extraction lost patch/slide ordering')
    for buffer in buffers:buffer.flush()
    del buffers
    for row in result:row['sha256']=digest_file(row['path'])
    save({'contract':contract,'rows':result,'rng':rng_state(),'state':'completed'},index)
    return result


def prototypes(feature_index,output):
    from .engine import rng_state,restore_rng
    out=Path(output);out.mkdir(parents=True,exist_ok=True);path=out/'prototypes.pt'
    train=[r for r in feature_index if r['subset']=='train']
    contract=digest_object([{k:r[k] for k in ['slide_id','sha256','subset','target','global_patient_id']} for r in train])
    if exists(path):
        value=load(path)
        if value['contract']!=contract:raise ValueError('PMIL TRAIN prototype discovery inputs changed')
        restore_rng(value['rng']);return value['result']

    def load_features(row):
        if digest_file(row['path'])!=row['sha256']:raise ValueError('Actual PMIL TRAIN features changed')
        return np.load(row['path'],allow_pickle=False)

    value=discover(train,load_features)
    save({'contract':contract,'result':value,'rng':rng_state()},path)
    write_json(out/'discovery.json',{k:v for k,v in value.items() if k!='features'})
    return value


def train(data,method,output,fm=None,split=None,seed=2026,device='cpu',config=None,resume=False,stop_after=None):
    from .engine import seed_everything
    if method not in {PROFILE,'PMIL Prototypical'}:raise ValueError('Choose the complete raw Prototypical PMIL reference')
    supplied=config or {};model=supplied.get('model',{})
    if set(supplied)-{'epochs','threads','model'} or not isinstance(model,dict) or set(model)-{'encoder_epochs'}:
        raise ValueError('PMIL preserves full ResNet34/512/128 widths, original losses/optimizers/AP; only explicit stage budgets/threads vary')
    cfg={'encoder_epochs':model.get('encoder_epochs',30),'bag_epochs':supplied.get('epochs',30),'threads':supplied.get('threads',2)}
    if any(type(v) is not int or v<1 for v in cfg.values()) or type(seed) is not int or (stop_after is not None and (type(stop_after) is not int or stop_after<1)):
        raise ValueError('Declare positive integer PMIL stage budgets/threads and integer seed')
    bundle,assets,inventory=inputs(data,fm,split);out=Path(output).expanduser().resolve()
    request={'backend':'pmil_prototypical_complete_reference','method':PROFILE,'data':str(bundle.root),
        'fm':bundle.fm,'split':bundle.split,'seed':seed,'device':str(device),'config':cfg,
        'software':software_identity(),'bundle_sha256':bundle.fingerprint,
        'feature_inventory_sha256':inventory['feature_inventory_sha256'],'raw_contract':assets.contract,
        'weights_sha256':digest_file(assets.weights),'synthetic':bool(bundle.config.get('synthetic',False)),
        'validation_repairs':'No-grad VAL every5epochs/final; selected encoder consistently re-extracted; no unconditional best overwrite',
        'test_labels_used_before_explicit_evaluate':False,'formal_admission':False}
    if resume:
        if read_json(out/'request.json')!=request:raise ValueError('PMIL resume changed method, inputs, source, initialization or phase budgets')
    else:out.mkdir(parents=True,exist_ok=False);write_json(out/'request.json',request)
    lock=_lock(out,resume)
    try:
        seed_everything(seed);torch.set_num_threads(cfg['threads']);dev=torch.device(device)
        if dev.type=='cuda' and not torch.cuda.is_available():raise ValueError('Requested actual PMIL CUDA runtime unavailable; no CPU fallback')
        if dev.type not in {'cpu','cuda'}:raise ValueError('Use explicit CPU or CUDA')
        progress=Progress(out,stop_after)
        selected=encoder_phase(bundle,assets,out/'max_mil',cfg['encoder_epochs'],dev,progress)
        exported=export_encoder(selected,out/'selected_patch_encoder.pt')
        features=extract(bundle,assets,exported,out/'features',dev)
        discovered=prototypes(features,out/'prototype_discovery')
        bag=bag_phase(assets,exported,features,discovered['features'],out/'pmil',cfg['bag_epochs'],dev,progress)
        if bundle.check(deep=True)['feature_inventory_sha256']!=inventory['feature_inventory_sha256']:
            raise ValueError('PMIL actual input bytes changed during full pipeline')
        write_json(out/'selected.json',{'encoder':str(exported),'encoder_sha256':digest_file(exported),
            'bag':str(bag),'bag_sha256':digest_file(bag),'prototype':str(checkpoint_path(out/'prototype_discovery/prototypes.pt')),
            'prototype_sha256':digest_file(checkpoint_path(out/'prototype_discovery/prototypes.pt'))})
        write_json(out/'status.json',{'state':'completed','test_evaluated':False,'formal_admission':False,
            'stages':['raw_max_MIL','actual_512D_feature_extraction','local_AP','global_AP','dual_CE_PMIL']})
        return out
    except Interrupted:
        write_json(out/'status.json',{**read_json(out/'status.json'),'state':'interrupted'});return out
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','error':repr(error)});raise
    finally:lock.unlink(missing_ok=True)


def evaluate(run,subset='test',device='cpu'):
    from .engine import seed_everything,score
    out=Path(run).resolve();request=read_json(out/'request.json')
    if subset not in {'val','test'} or read_json(out/'status.json')['state']!='completed':
        raise ValueError('Complete all original PMIL stages before evaluating held-out bags')
    if request['software']!=software_identity():raise ValueError('PMIL actual software changed')
    bundle,assets,inventory=inputs(request['data'],request['fm'],request['split'])
    if bundle.fingerprint!=request['bundle_sha256'] or inventory['feature_inventory_sha256']!=request['feature_inventory_sha256']:
        raise ValueError('PMIL actual raw data/labels/partitions changed')
    selected=read_json(out/'selected.json')
    for key in ['encoder','bag','prototype']:
        if digest_file(selected[key])!=selected[key+'_sha256']:raise ValueError('PMIL selected learned checkpoint changed')
    seed_everything(request['seed']);torch.set_num_threads(request['config']['threads']);dev=torch.device(device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise ValueError('Requested actual PMIL CUDA unavailable; no CPU fallback')
    features=extract(bundle,assets,selected['encoder'],out/('evaluation_'+subset),dev,subsets=(subset,))
    p=load(selected['prototype'])['result']['features'];prototype=torch.from_numpy(p.copy()).to(dev)
    model=bag_model(assets.weights,selected['encoder'],prototype).to(dev);model.load_state_dict(load(selected['bag'])['model'],strict=True);model.eval()
    rows={r['slide_id']:r for r in bundle.records(subset)};result=[]
    with torch.no_grad():
        for record in features:
            x,_=FeatureBags([record])[0]
            # Selected K5 membership follows the actual learned patch head.
            value=bag_objective(model,x.to(dev),prototype,torch.tensor([record['target']],device=dev))
            probability=value['bag_logits'].softmax(1)[0].cpu().numpy();row=rows[record['slide_id']]
            result.append({**{k:row[k] for k in ['slide_id','patient_id','global_patient_id','target']},
                'subset':subset,'source_slides':row['slide_id'],'prediction':int(probability[1]>=.5),
                'p_0':float(probability[0]),'p_1':float(probability[1])})
    if bundle.check(deep=True)['feature_inventory_sha256']!=inventory['feature_inventory_sha256']:
        raise ValueError('PMIL original inputs changed during evaluation')
    frame=pd.DataFrame(result);metrics=score(bundle,frame)
    frame.to_csv(out/(subset+'_predictions.csv'),index=False);write_json(out/(subset+'_metrics.json'),metrics)
    return metrics
