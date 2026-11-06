"""Real RGB224 bags, original patch identities and fixed actual author teacher assets."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from .data import resolve
from .io import digest_file,digest_object,read_json,write_json
from .promptmil_source import WEIGHTS

KIND='promptmil_original_RGB224_and_cohort_DINO'


def identity(bundle):
    return digest_object({'task':bundle.config['task_id'],'type':bundle.task_type,'classes':bundle.classes,
        'fm':bundle.fm,'encoder':{k:v for k,v in bundle.encoder.items() if k!='promptmil'},
        'training_unit':bundle.config.get('training_unit','slide'),
        'evaluation_unit':bundle.config.get('evaluation_unit','slide'),'rows':bundle.rows.to_dict('records')})


def describe(bundle,row,schema):
    path=resolve(bundle.root,row['path'])
    if schema.get('storage') == 'source_merged_jpeg_rgb224':
        from .promptmil_jpeg import describe as describe_jpeg
        return describe_jpeg(path,schema)
    if path.suffix.lower() not in {'.h5','.hdf5'}:raise ValueError('Pack actual Prompt-MIL RGB224 and IDs in HDF5')
    with h5py.File(path,'r')as f:
        for key in [schema['pixel_key'],schema['patch_ids_key']]:
            if key not in f or isinstance(f.get(key,getlink=True),(h5py.ExternalLink,h5py.SoftLink)):
                raise ValueError('Prompt-MIL pixels and original IDs must be embedded, not externally linked')
            if not isinstance(f[key],h5py.Dataset) or f[key].is_virtual or f[key].external:
                raise ValueError('Prompt-MIL RGB cannot use hidden external storage')
        rgb=f[schema['pixel_key']]
        if rgb.ndim!=4 or rgb.shape[0]<1 or rgb.shape[1:]!=(224,224,3) or rgb.dtype!=np.uint8:
            raise ValueError('Prompt-MIL needs full actual RGB224 uint8 patches, no frozen vectors or silent resize')
        ids=f[schema['patch_ids_key']].asstr()[:].tolist()
        if len(ids)!=len(rgb) or len(ids)!=len(set(ids)) or any(not isinstance(v,str)or not v.strip()for v in ids):
            raise ValueError('Prompt-MIL original patch IDs must be unique, complete and aligned')
        return {'shape':list(rgb.shape),'ordered_patch_ids_sha256':digest_object(ids)}


def check_schema(bundle,schema):
    if (not isinstance(schema,dict) or schema.get('schema_version')!=1 or schema.get('kind')!=KIND
        or bundle.task_type!='classification' or bundle.dim!=3 or len(bundle.classes)<2
        or bundle.config.get('training_unit','slide')!='slide'
        or schema.get('condition')not in WEIGHTS or schema.get('normalization')!='source_uint8_div255_only'
        or any(not isinstance(schema.get(k),str)or not schema[k].strip()
               for k in ['pixel_key','patch_ids_key','initial_weights','pixel_provenance','patch_provenance'])
        or any(k in bundle.encoder for k in ['raw_images','raw_regions','macnet','icmil_coupled','pmil_prototypical','multiscale','vision_language'])):
        raise ValueError('Prompt-MIL needs slide classification, actual RGB224/IDs, original cohort DINO and declared source preprocessing')
    if schema.get('storage','hdf5_rgb224') not in {'hdf5_rgb224','source_merged_jpeg_rgb224'}:
        raise ValueError('Prompt-MIL storage must be explicit original RGB224 or merged JPEG')
    if schema.get('storage') == 'source_merged_jpeg_rgb224':
        from .promptmil_jpeg import check_schema as check_jpeg
        check_jpeg(schema)
    return schema


def prepare(bundle,schema,output):
    schema=check_schema(bundle,schema);out=resolve(bundle.root,output)
    if out.exists():raise FileExistsError(out)
    weights=resolve(bundle.root,schema['initial_weights']).resolve()
    if digest_file(weights)!=WEIGHTS[schema['condition']]['sha256']:
        raise ValueError('Original Prompt-MIL cohort teacher weights differ')
    rows={}
    for row in bundle.rows.to_dict('records'):
        path=resolve(bundle.root,row['path']).resolve()
        rows[row['slide_id']]={**describe(bundle,row,schema),'path':str(path),'sha256':digest_file(path),
                              **{k:row[k]for k in ['target','subset','global_patient_id']}}
    value={'schema_version':1,'kind':KIND,'identity':identity(bundle),'schema':schema,'rows':rows,
        'original_weights':str(weights),'original_weights_sha256':WEIGHTS[schema['condition']]['sha256'],
        'synthetic':bool(bundle.config.get('synthetic',False)),'preparer_sha256':digest_file(Path(__file__))}
    value['contract']=digest_object(value);write_json(out,value);return out


class PromptInputs:
    def __init__(self,bundle,manifest):
        self.bundle=bundle;self.path=resolve(bundle.root,manifest);v=read_json(self.path);contract=v.pop('contract',None)
        if (contract!=digest_object(v)or v.get('schema_version')!=1 or v.get('kind')!=KIND
            or v.get('identity')!=identity(bundle)or v.get('preparer_sha256')!=digest_file(Path(__file__))
            or v.get('synthetic')!=bool(bundle.config.get('synthetic',False))):
            raise ValueError('Prompt-MIL actual input/label/split/preparer binding changed')
        self.schema=check_schema(bundle,v['schema']);self.condition=self.schema['condition']
        self.weights=Path(v['original_weights']);self.rows=v['rows'];self.contract=contract
        if set(self.rows)!=set(bundle.rows.slide_id):raise ValueError('Prompt-MIL slide population changed')
        for row in bundle.rows.to_dict('records'):
            item=self.rows[row['slide_id']]
            if (item['path']!=str(resolve(bundle.root,row['path']).resolve())
                or any(item[k]!=row[k]for k in ['target','subset','global_patient_id'])):
                raise ValueError('Prompt-MIL original joined identity changed')
        if v['original_weights_sha256']!=WEIGHTS[self.condition]['sha256']:
            raise ValueError('Prompt-MIL original teacher identity changed')
        self.verify_bytes()

    def verify_bytes(self):
        if digest_file(self.weights)!=WEIGHTS[self.condition]['sha256']:
            raise ValueError('Prompt-MIL original checkpoint bytes changed')
        for row in self.bundle.rows.to_dict('records'):self.inventory(row,True)

    def inventory(self,row,deep=False):
        item=self.rows[row['slide_id']]
        if deep:
            actual=describe(self.bundle,row,self.schema)
            if digest_file(item['path'])!=item['sha256']or actual!={k:item[k]for k in actual}:
                raise ValueError('Prompt-MIL original RGB or aligned patch identity bytes changed')
        return dict(item)

    def pixels(self,row):
        item=self.rows[row['slide_id']]
        if self.schema.get('storage') == 'source_merged_jpeg_rgb224':
            from .promptmil_jpeg import pixels as jpeg_pixels
            if digest_file(item['path']) != item['sha256']:
                raise ValueError('Prompt-MIL original JPEG bag index bytes changed')
            rgb=jpeg_pixels(item['path'],self.schema)
        else:
            with h5py.File(item['path'],'r')as f:rgb=f[self.schema['pixel_key']][:]
        # Preserve original image_to_tensor NHWC->NCHW strides and /255 dtype
        # promotion; source examples do not enable ImageNet normalization.
        return torch.from_numpy(rgb).permute(0,3,1,2)/255.


def pack(table,output):
    from PIL import Image
    frame=pd.read_csv(table,dtype=str,keep_default_na=False)
    if not {'slide_id','patch_id','path'}<=set(frame)or frame[['slide_id','patch_id','path']].eq('').any().any():
        raise ValueError('Pack table needs original slide_id, patch_id and RGB224 image path')
    if frame[['slide_id','patch_id']].duplicated().any():raise ValueError('Duplicate source patch identity')
    dest=Path(output).resolve();dest.mkdir(parents=True,exist_ok=False);records=[]
    for number,(sid,rows)in enumerate(frame.groupby('slide_id',sort=False)):
        p=dest/f'{number:06d}.h5'
        with h5py.File(p,'x')as f:
            data=f.create_dataset('rgb',shape=(len(rows),224,224,3),dtype=np.uint8,chunks=(1,224,224,3),compression='gzip',compression_opts=1)
            for i,row in enumerate(rows.to_dict('records')):
                with Image.open(row['path'])as image:
                    if image.mode!='RGB'or image.size!=(224,224):raise ValueError('Pack requires actual RGB224; no implicit conversion/crop/resize')
                    data[i]=np.asarray(image)
            f['patch_ids']=np.asarray(rows.patch_id.tolist(),dtype=h5py.string_dtype('utf-8'))
            f.attrs['decoder']='Pillow RGB; packed decoded pixel bytes define the actual controlled input'
        records.append({'slide_id':sid,'path':str(p)})
    pd.DataFrame(records).to_csv(dest/'features.csv',index=False);return dest/'features.csv'
