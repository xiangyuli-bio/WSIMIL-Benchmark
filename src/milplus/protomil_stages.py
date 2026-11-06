"""Lossless content caches, actual annotated concept probes and bound review."""
from pathlib import Path
import fcntl,hashlib,html,os,uuid
import h5py
import numpy as np
import torch
from . import protomil_source as source
from .io import read_json,write_json,digest_file,digest_object
from .protomil_data import read_image,validate_probe,actual_features


def tensor_sha(value):
    x=value.detach().cpu().contiguous()
    return hashlib.sha256(str(x.dtype).encode()+repr(tuple(x.shape)).encode()+x.numpy().tobytes()).hexdigest()


def model_sha(model):return digest_object({k:tensor_sha(v) for k,v in model.state_dict().items()})


def normalized_inputs(assets,cache):
    """Original normalized .pt pieces for the literal source SAE loader."""
    out=Path(cache)/'normalized_inputs';out.mkdir(parents=True,exist_ok=True);paths=[]
    for row in assets.population:
        key=digest_object({'feature':row['sha256'],'patch_order':row['ordered_ids_sha256'],'normalization':'original_float32_CPU_L2_prepare_sae_data','torch':torch.__version__,'implementation':digest_file(Path(__file__).with_name('protomil_data.py'))})
        target=out/(key+'.pt');meta=target.with_suffix('.json')
        with (out/(key+'.lock')).open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not target.exists():
                x=assets.pretraining_features(row);tmp=target.with_name(target.name+'.'+uuid.uuid4().hex)
                torch.save(x,tmp);sha=digest_file(tmp);os.replace(tmp,target);write_json(meta,{'sha256':sha,'tensor_sha256':tensor_sha(x),'shape':list(x.shape)})
            value=read_json(meta)
            if digest_file(target)!=value['sha256'] or value['shape']!=[row['patches'],512]:raise ValueError('Actual normalized SAE cache changed')
        paths.append(str(target))
    return paths


def conceptualize(assets,sae,row,cache,device,verify_existing=False):
    root=Path(cache)/'concept_features';root.mkdir(parents=True,exist_ok=True)
    key=digest_object({'SAE':model_sha(sae),'input':row['sha256'],'patch_order':row['ordered_ids_sha256'],
        'implementation':digest_file(__file__),'source':source.COMMIT,'SAE_dependency':source.DTN_COMMIT,'torch':torch.__version__,'device':str(device),'dtype':'float32_full_original_WSI_forward'})
    path=root/(key+'.h5');metadata=path.with_suffix('.json');actual=None
    with (root/(key+'.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists() or verify_existing:
            # Exact source extract_concept_activation: complete WSI forward,
            # no chunked-GEMM change, FP16 conversion or concept truncation.
            x=assets.features(row).to(device)
            with torch.no_grad():h,_=sae(x)
            actual=h.squeeze().detach().cpu().contiguous()
            if actual.shape!=(row['patches'],2048) or not torch.isfinite(actual).all() or (actual<0).any():raise ValueError('Original full SAE concept output changed shape or became invalid')
        if not path.exists():
            tmp=path.with_name(path.name+'.'+uuid.uuid4().hex)
            with h5py.File(tmp,'w') as f:f.create_dataset('features',data=actual.numpy(),compression='gzip',shuffle=True)
            payload={'sha256':digest_file(tmp),'tensor_sha256':tensor_sha(actual),'shape':list(actual.shape),'sae_state_sha256':model_sha(sae),
                'input_sha256':row['sha256'],'original_patch_order':row['ordered_ids_sha256'],'format_port':'lossless HDF5 gzip float32; source tensor values preserved'}
            os.replace(tmp,path);write_json(metadata,payload)
        record=read_json(metadata)
        if digest_file(path)!=record['sha256'] or record['shape']!=[row['patches'],2048]:raise ValueError('Actual concept cache bytes or complete bag shape changed')
        if actual is not None:
            with h5py.File(path) as f:existing=torch.from_numpy(f['features'][:])
            if not torch.equal(actual,existing):raise ValueError('Fresh complete source SAE projection differs from immutable cache')
    return {'path':str(path),'metadata':str(metadata),**record}


def load_concepts(record,removed=()):
    if digest_file(record['path'])!=record['sha256']:raise ValueError('Actual frozen concept feature cache changed')
    with h5py.File(record['path']) as f:x=torch.from_numpy(f['features'][:])
    if list(x.shape)!=record['shape'] or tensor_sha(x)!=record['tensor_sha256']:raise ValueError('Actual complete concept rows/dtype/order changed')
    if removed:x[:,list(removed)]=0
    return x


def probe_bank(assets,sae,cache,output,device,per_class=10000):
    if type(per_class) is not int or per_class<5:raise ValueError('At least five actual normal/tumor probes are needed for original top10 concepts')
    positive=[p for p in assets.probes if p['region_label']==1];negative=[p for p in assets.probes if p['region_label']==0]
    if len(positive)!=per_class or len(negative)!=per_class:raise ValueError('Actual normal/tumor probing populations differ from the explicitly requested budget')
    out=Path(output);out.mkdir(parents=True,exist_ok=True);all_probes=positive+negative;features=[];indices={};counts={};totals={}
    # Source analysis concatenates positive rows before negative rows; use the
    # same NumPy argsort(...)[::-1][:10] tie ordering under the pinned NumPy.
    all_ids={p['slide_id'] for p in all_probes};raw={sid:actual_features(assets.rows[sid]['path']) for sid in all_ids}
    outputs=[]
    for group in [positive,negative]:
        normalized=[]
        for p in group:
            if p['global_patient_id']!=assets.rows[p['slide_id']]['global_patient_id'] or assets.rows[p['slide_id']]['subset']!='train':raise ValueError('Concept probing accessed a held-out patient')
            validate_probe(p,assets.value['probe_policy'])
            patch=raw[p['slide_id']][p['patch_index']]
            # Literal gather_img_features uses NumPy linalg.norm per patch,
            # separately from the WSI Torch row-norm preprocessing.
            normalized.append((patch/np.linalg.norm(patch)).numpy())
        x=torch.tensor(np.asarray(normalized)).to(device)
        with torch.no_grad():h,_=sae(x)
        outputs.append(h.squeeze().detach().cpu().numpy())
    values=np.concatenate(outputs,axis=0)
    if values.shape!=(2*per_class,2048) or not np.isfinite(values).all():raise ValueError('Actual grouped probe SAE output changed')
    active=np.unique(np.concatenate([np.where(row!=0)[0] for row in values])).tolist()
    for concept in active:
        column=values[:,concept];indices[str(concept)]=np.argsort(column)[::-1][:10].tolist();counts[str(concept)]=int(np.count_nonzero(column));totals[str(concept)]=float(np.sum(column))
    result={'schema_version':1,'sae_state_sha256':model_sha(sae),'contract':assets.contract,'positive_before_negative':True,
        'per_class':per_class,'active_concepts':active,'top10_indices':indices,'activation_counts':counts,'activation_totals':totals,
        'probes':all_probes,'selection':'literal per-patch NumPy norm and full positive/negative group forward; NumPy argsort descending first10','pathologist_review':'not_performed_by_software'}
    result['bank_sha256']=digest_object(result);write_json(out/'concept_bank.json',result)
    # The report references actual user-supplied RGB/masks. HDF5-backed inputs
    # are displayed through embedded PNGs; no generated pathology explanation.
    lines=['<!doctype html><meta charset="utf-8"><title>ProtoMIL concept inspection</title>',
        '<h1>ProtoMIL concept inspection</h1><p>Review source TRAIN patches. Software has not supplied pathologist names or decisions.</p>']
    from PIL import Image
    import io,base64
    encoded={}
    for concept in active:
        lines.append('<h2>Concept '+str(concept)+'</h2>')
        for idx in indices[str(concept)]:
            if idx not in encoded:
                data=io.BytesIO();Image.fromarray(read_image(all_probes[idx]['image'])).save(data,format='PNG');encoded[idx]=base64.b64encode(data.getvalue()).decode()
            p=all_probes[idx];label=html.escape(p['slide_id']+' / '+p['patch_id']+' / annotation='+str(p['region_label']))
            lines.append('<figure style="display:inline-block"><img width="128" src="data:image/png;base64,'+encoded[idx]+'"><figcaption>'+label+'</figcaption></figure>')
    (out/'concept_inspection.html').write_text('\n'.join(lines));return result


def review_path(assets,bank):
    if assets.review_file is None:return None
    return Path(assets.review_file.replace('{sae_state_sha256}',bank['sae_state_sha256']).replace('{concept_bank_sha256}',bank['bank_sha256']))


def read_review(assets,bank,required):
    if not required:return {'removed_concepts':[],'review_sha256':None,'review_scope':'non-intervened original concept condition'}
    path=review_path(assets,bank)
    if path is None or not path.is_file():return None
    value=read_json(path)
    expected={'schema_version','sae_state_sha256','concept_bank_sha256','reviewer','provenance','removed_concepts','concept_names'}
    if not isinstance(value,dict) or set(value)!=expected or value['schema_version']!=1 or value['sae_state_sha256']!=bank['sae_state_sha256'] or value['concept_bank_sha256']!=bank['bank_sha256']:
        raise ValueError('Concept review does not match this actual SAE and probe bank; never reuse historical neuron IDs blindly')
    if not isinstance(value['reviewer'],str) or not value['reviewer'].strip() or not isinstance(value['provenance'],str) or not value['provenance'].strip():raise ValueError('Actual concept review provenance/reviewer is required')
    if value['reviewer']=='synthetic_validation_only' and not assets.bundle.config.get('synthetic',False):raise ValueError('Synthetic concept decisions cannot be used as actual pathologist review')
    removed=value['removed_concepts'];names=value['concept_names']
    if not isinstance(removed,list) or len(set(removed))!=len(removed) or any(type(i) is not int or i not in bank['active_concepts'] for i in removed):raise ValueError('Removed concept IDs must belong to the actual trained active concept bank')
    if not isinstance(names,dict) or any(not str(k).isdigit() or int(k) not in bank['active_concepts'] or not isinstance(v,str) or not v.strip() for k,v in names.items()) or any(str(i) not in names for i in removed):raise ValueError('Bind each actual removed concept to its supplied reviewed description')
    return {**value,'review_sha256':digest_file(path),'review_path':str(path)}
