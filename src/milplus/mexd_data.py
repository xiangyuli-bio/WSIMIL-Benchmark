"""Actual ordered feature/patch identities; no private pretrained MoE is invented."""
import hashlib

import numpy as np

from .data import load_array,resolve
from .io import digest_file,digest_object


def names(bundle,row,count):
    if row.get('patch_ids_path'):
        values=load_array(resolve(bundle.root,row['patch_ids_path']),'patch_ids')
    else:
        path=resolve(bundle.root,row['path'])
        if path.suffix in {'.h5','.hdf5'}:
            import h5py
            with h5py.File(path,'r')as handle:has_names='patch_ids'in handle
            values=load_array(path,'patch_ids')if has_names else None
        else:values=None
    if values is None:
        return [f'feature_row_{i}'for i in range(count)],'content_bound_original_feature_row'
    values=np.asarray(values)
    if values.shape!=(count,):raise ValueError('MExD patch identities must align with every real feature row')
    result=[v.decode('utf-8')if isinstance(v,bytes)else v for v in values.tolist()]
    if len(set(result))!=len(result)or any(not isinstance(v,str)or not v for v in result):
        raise ValueError('MExD original patch identities must be unique nonempty strings')
    return result,'provided_original_patch_ids'


class Inputs:
    def __init__(self,bundle):
        if bundle.task_type!='classification'or bundle.config.get('training_unit','slide')!='slide':
            raise ValueError('MExD uses one original slide classification bag per update')
        self.bundle=bundle;self.rows={r['slide_id']:r for subset in ['train','val','test']for r in bundle.records(subset)}
        self.inventory=[]
        for subset in ['train','val','test']:
            for row in sorted(bundle.records(subset),key=lambda r:r['slide_id']):
                features,_=bundle.load(row)
                if len(features)<2:
                    raise ValueError('Full MExD negative/positive routing needs at least two actual instances')
                ids,kind=names(bundle,row,len(features))
                self.inventory.append({'slide_id':row['slide_id'],'global_patient_id':row['global_patient_id'],
                    'target':int(row['target']),'subset':subset,'instances':len(features),
                    'feature_file_sha256':digest_file(resolve(bundle.root,row['path'])),
                    'computation_float32_sha256':hashlib.sha256(features.numpy().tobytes()).hexdigest(),
                    'ordered_ids':ids,'identity_kind':kind})
        self.identity=digest_object({'features':self.inventory,'classes':bundle.classes,'fm':bundle.fm,
                                     'encoder':bundle.encoder,'split':bundle.split,
                                     'class_roles':bundle.config.get('class_roles',{})})
