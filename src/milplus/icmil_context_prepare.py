"""Explicit fixed-bag/PCA reference preparation, fitted on all TRAIN patches only.

The public author TCGA feature file is already 25D. Its historical PCA fit is
not provided. This named preparation is our auditable reference, not a claim
to regenerate that unpublished transformation.
"""
import copy
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .io import digest_file,write_json


def prepare(data,output,bag_size,projection,seed=2026,fm=None,split=None):
    if type(bag_size) is not int or bag_size<1 or type(seed) is not int or seed<0:
        raise ValueError('Declare a positive fixed bag budget and nonnegative integer selection seed')
    if projection not in {'native','train_pca25_reference'}:
        raise ValueError('Choose native <=25D or explicit TRAIN-only PCA25 reference preparation')
    b=Bundle(data,fm,split)
    if (b.task_type!='classification' or len(b.classes)!=2
            or b.config.get('training_unit','slide')!='slide' or b.config.get('evaluation_unit','slide')!='slide'):
        raise ValueError('Original context ICMIL needs binary, slide-unit bags with patient-safe partitions')
    if projection=='native' and not 1<=b.dim<=25:
        raise ValueError('Native ICMIL cannot silently truncate an FM wider than25')
    if projection=='train_pca25_reference' and b.dim<=25:
        raise ValueError('Use native for <=25D; do not manufacture additional PCA dimensions')
    inventory=b.check(deep=True);rows=b.rows.to_dict('records')
    if any(r['instances']<bag_size for r in inventory['features']):
        raise ValueError('Every original bag must meet the explicit budget; no replacement sampling or zero-padded instances')
    center=np.zeros(b.dim,dtype=np.float64);components=None;count=0
    if projection=='train_pca25_reference':
        # Stable merge of full per-slide centered sufficient statistics. No
        # held-out feature or label participates in fitting; no patch cap.
        scatter=np.zeros((b.dim,b.dim),dtype=np.float64)
        for row in b.records('train'):
            x=b.load(row)[0].numpy().astype(np.float64);n=len(x);mu=x.mean(0);delta=mu-center
            centered=x-mu;scatter+=centered.T@centered
            if count:scatter+=np.outer(delta,delta)*(count*n/(count+n))
            center+=delta*(n/(count+n));count+=n
        if count<26:raise ValueError('TRAIN-only PCA25 needs at least26 actual TRAIN instances')
        values,vectors=np.linalg.eigh(scatter/(count-1));order=np.argsort(values)[::-1][:25]
        components=vectors[:,order].T.copy()
        # Resolve arbitrary eigenvector signs; degenerate eigenspaces remain
        # bound to exact serialized components/environment, not guessed later.
        for v in components:
            pivot=np.argmax(np.abs(v))
            if v[pivot]<0:v*=-1
    out=Path(output).expanduser().resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        (out/'features').mkdir();(out/'selection').mkdir();artifacts={};index=[];selection=[]
        if components is not None:
            np.savez(out/'projection.npz',mean=center,components=components,
                eigenvalues=values[order],train_instances=np.asarray(count))
            artifacts[str(out/'projection.npz')]=digest_file(out/'projection.npz')
        for row in rows:
            x,coords=b.load(row);x=x.numpy();key=row['slide_id']
            # Stable across labels, dataset order and partition assignment.
            slide_seed=int.from_bytes(hashlib.sha256((str(seed)+'\0'+key).encode()).digest()[:8],'big')
            chosen=np.sort(np.random.default_rng(slide_seed).choice(len(x),bag_size,replace=False))
            transformed=x[chosen] if components is None else (x[chosen].astype(np.float64)-center)@components.T
            # IDs are metadata, never filesystem paths.
            filename=hashlib.sha256(key.encode()).hexdigest();path=out/'features'/(filename+'.h5')
            with h5py.File(path,'w') as f:
                f['features']=transformed.astype(np.float32)
                f['original_row_indices']=chosen
                if coords is not None:f['coords']=coords.numpy()[chosen]
            indices=out/'selection'/(filename+'.npy');np.save(indices,chosen,allow_pickle=False)
            artifacts[str(path)]=digest_file(path);artifacts[str(indices)]=digest_file(indices)
            index.append({'slide_id':key,'path':str(path),'sha256':digest_file(path)})
            selection.append({'slide_id':key,'patient_id':row['patient_id'],'global_patient_id':row['global_patient_id'],
                'subset':row['subset'],'original_instances':len(x),'indices':str(indices),
                'indices_sha256':digest_file(indices),'source_feature':next(r for r in inventory['features'] if r['slide_id']==key)})
        proof={'schema_version':1,'projection':projection,'source_bundle_sha256':b.fingerprint,
            'source_feature_inventory_sha256':inventory['feature_inventory_sha256'],'fit_subsets':['train'],
            'heldout_labels_used':False,'output_bag_size':bag_size,'output_dimension':25 if components is not None else b.dim,
            'input_dimension':b.dim,'pca_train_instances':count,'pca_uses_all_train_patches':components is not None,
            'selection_seed':seed,'selection':'uniform_without_replacement_sorted_original_row_order',
            'historical_author_PCA_reconstruction_claimed':False,'artifacts':artifacts,'rows':selection}
        if b.check(deep=True)['feature_inventory_sha256']!=inventory['feature_inventory_sha256']:
            raise ValueError('Input features changed during complete ICMIL preparation')
        write_json(out/'preparation.json',proof)
        pd.DataFrame(index).to_csv(out/'features.csv',index=False)
        # Persist selected identity/label/split tables, not paths into a second
        # experiment and not metadata from feature indexes.
        slides=b.rows[['slide_id','patient_id','global_patient_id','label']]
        slides.to_csv(out/'slides.csv',index=False)
        b.rows[['patient_id','subset']].drop_duplicates().to_csv(out/'split.csv',index=False)
        c={'schema_version':1,'task_id':b.config['task_id'],'task_type':'classification',
            'classes':copy.deepcopy(b.classes),'synthetic':bool(b.config.get('synthetic',False)),
            'training_unit':'slide','evaluation_unit':'slide','slides':'slides.csv',
            'label_provenance':b.config.get('label_provenance','See source bundle '+str(b.root)),
            'defaults':{'fm':b.fm,'split':b.split},'splits':{b.split:'split.csv'},
            'encoders':{b.fm:{'dimension':proof['output_dimension'],'index':'features.csv',
                'source_encoder':copy.deepcopy(b.encoder),'icmil_context':{'schema_version':1,
                'manifest':'preparation.json','bag_size':bag_size}}}}
        (out/'task.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
        from .icmil_context_data import ContextInputs
        ContextInputs(out)
        write_json(out/'preparation_status.json',{'state':'completed','projection':projection,'formal_admission':False})
        return out
    except BaseException as error:
        write_json(out/'preparation_status.json',{'state':'failed','error':repr(error)});raise
