"""Native complete physical HIPT pyramid inputs; no inferred or padded missing cells."""
import copy
import math
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from .data import load_array, resolve
from .hipt_source import COMMIT, PATCH_SHA, TENSOR_SHA, load_backbone
from .io import digest_file, digest_object, read_json, write_json


def encoder_identity(bundle):
    return digest_object({'fm':bundle.fm,'encoder':{k:v for k,v in bundle.encoder.items() if k!='hipt'}})


def patch_stride_for(schema,row=None):
    """Resolve declared physical spacing; never infer it from feature order.

    Version1 retains its single cohort-wide stride. Version2 explicitly binds
    one measured stride to each slide, for mixed native scanning magnifications.
    The low-level alignment helper also accepts the original unversioned scalar
    geometry dictionaries used by independent source-reader diagnostics.
    """
    version=schema.get('schema_version',1)
    if type(version) is not int or version not in {1,2}:
        raise ValueError('HIPT physical geometry schema version must be1 or2')
    if version==1:
        if 'patch_stride_by_slide' in schema:
            raise ValueError('HIPT version1 requires one unambiguous physical patch stride')
        stride=schema.get('patch_stride')
    else:
        values=schema.get('patch_stride_by_slide')
        if 'patch_stride' in schema or not isinstance(values,dict) or not values:
            raise ValueError('HIPT version2 requires only explicit per-slide physical patch strides')
        slide_id=None if row is None else row.get('slide_id')
        if not isinstance(slide_id,str) or slide_id not in values:
            raise ValueError('HIPT physical patch stride is missing for this slide')
        stride=values[slide_id]
    if (isinstance(stride,bool) or not isinstance(stride,(float,int)) or
            stride<=0 or stride>2**53-1 or not math.isfinite(stride)):
        raise ValueError('HIPT physical patch stride must be positive, finite and exactly representable')
    if schema.get('coordinate_frame')=='level0_pixels' and stride!=int(stride):
        raise ValueError('HIPT level0 pixel grids require actual integer-valued coordinates and stride')
    return stride


def schema_contract(schema,bundle):
    if (not isinstance(schema,dict) or type(schema.get('schema_version')) is not int or
            schema.get('schema_version') not in {1,2} or schema.get('kind')!='hipt_complete_16x16_pyramid' or
            schema.get('source_commit')!=COMMIT or schema.get('grid_shape')!=[16,16] or
            type(schema.get('synthetic')) is not bool or schema['synthetic']!=bool(bundle.config.get('synthetic',False))):
        raise ValueError('HIPT requires its actual complete physical 16x16 hierarchy and matching source/synthetic declaration')
    if (bundle.task_type!='classification' or bundle.dim!=384 or
            bundle.config.get('training_unit','slide')!='slide'):
        raise ValueError('Native HIPT requires individual classification WSI bags with 384-dimensional patch tokens')
    if schema.get('coordinate_frame') not in {'level0_pixels','micrometers'}:
        raise ValueError('HIPT requires explicit physical coordinate units')
    if schema['schema_version']==1:
        patch_stride_for(schema)
    else:
        values=schema.get('patch_stride_by_slide')
        if not isinstance(values,dict) or set(values)!=set(bundle.rows.slide_id):
            raise ValueError('HIPT per-slide physical strides must cover exact task slides without extras')
        for slide_id in bundle.rows.slide_id:
            patch_stride_for(schema,{'slide_id':slide_id})
    for key in ['patching_provenance','patch_encoder_provenance','preprocessing_provenance']:
        if not isinstance(schema.get(key),str) or not schema[key].strip():
            raise ValueError('HIPT requires '+key)
    if schema.get('patch_encoder')!='HIPT_ViT256_DINO' or schema.get('patch_encoder_weights_sha256')!=PATCH_SHA:
        raise ValueError('Native HIPT local weights require the actual paired ViT256 DINO embedding space, not merely equal feature width')
    if bundle.encoder.get('embedding_space')!='HIPT_ViT256_DINO' or bundle.encoder.get('weights_parent_sha256')!=PATCH_SHA:
        raise ValueError('HIPT requires matching native patch embedding space and parent weight provenance in the encoder')
    spatial=bundle.encoder.get('spatial',{})
    if spatial.get('unit')!=schema['coordinate_frame'] or spatial.get('origin')!='top_left':
        raise ValueError('HIPT primary coordinate frame and pyramid geometry declaration differ')


def aligned_inputs(bundle,row,path,schema):
    stride=patch_stride_for(schema,row)
    x,_=bundle.load(row,require_coords=True)
    coords=np.asarray(load_array(resolve(bundle.root,row.get('coords_path') or row['path']),'coords'))
    with h5py.File(path,'r') as h:
        ids=np.asarray(h['region_ids'][:]);origins=np.asarray(h['region_origins'][:])
    if (ids.shape!=(len(x),) or ids.dtype.kind not in 'iu' or not len(ids) or
            origins.ndim!=2 or origins.shape[1]!=2 or not len(origins) or origins.dtype.kind not in 'iuf' or
            not np.isfinite(origins).all() or (origins<0).any() or
            len(np.unique(origins,axis=0))!=len(origins) or
            coords.shape!=(len(x),2) or coords.dtype.kind not in 'iuf' or not np.isfinite(coords).all() or
            (coords<0).any() or (coords>2**53-1).any() or (origins>2**53-1).any() or
            len(np.unique(coords,axis=0))!=len(coords)):
        raise ValueError('HIPT requires unique physical origins/coordinates and one aligned integer region assignment per patch')
    if set(ids.tolist())!=set(range(len(origins))):
        raise ValueError('HIPT region indices must cover the explicitly ordered origins exactly')
    if schema['coordinate_frame']=='level0_pixels' and (
            not np.equal(coords,np.floor(coords)).all() or not np.equal(origins,np.floor(origins)).all() or
            stride!=int(stride)):
        raise ValueError('HIPT level0 pixel grids require actual integer-valued coordinates and stride')
    permutations=[]
    for index,origin in enumerate(origins):
        rows=np.flatnonzero(ids==index)
        if len(rows)!=256:
            raise ValueError('HIPT cannot fabricate missing physical cells or square arbitrary bag sequences')
        offsets=(coords[rows].astype(np.float64)-origin.astype(np.float64))/stride
        rounded=np.rint(offsets)
        if not np.allclose(offsets,rounded,rtol=0,atol=1e-6) or (rounded<0).any() or (rounded>=16).any():
            raise ValueError('HIPT region assignment conflicts with actual physical 16x16 patch grid')
        integer=rounded.astype(np.int64)
        cell=integer[:,1]*16+integer[:,0]
        if len(np.unique(cell))!=256:
            raise ValueError('HIPT region cells collide')
        permutations.append(rows[np.argsort(cell,kind='stable')])
    order=np.stack(permutations)
    return torch.stack([x[idx] for idx in order]),coords,origins,order


def prepare(bundle,index,schema,weights,output):
    schema_contract(schema,bundle);output=Path(output).resolve()
    if output.exists():raise ValueError('HIPT pyramid manifest output must be new')
    frame=pd.read_csv(index,dtype=str,keep_default_na=False)
    if set(frame)!={'slide_id','path'} or frame.slide_id.duplicated().any() or set(frame.slide_id)!=set(bundle.rows.slide_id):
        raise ValueError('HIPT pyramid index must cover exact task slides without target columns')
    paths={r.slide_id:resolve(Path(index).resolve().parent,r.path).resolve() for r in frame.itertuples()}
    if len(set(paths.values()))!=len(paths):raise ValueError('HIPT slides cannot reuse a physical pyramid file')
    weights=Path(weights).resolve()
    load_backbone(weights)
    slides={}
    for row in bundle.rows.to_dict('records'):
        path=paths[row['slide_id']];sequence,coords,origins,order=aligned_inputs(bundle,row,path,schema)
        slides[row['slide_id']]={'path':os.path.relpath(path,output.parent),'sha256':digest_file(path),
            'primary_feature_sha256':digest_file(resolve(bundle.root,row['path'])),'regions':len(sequence),
            'physical_geometry_sha256':digest_object({'coords':coords.tolist(),'origins':origins.tolist(),'row_order':order.tolist()})}
    write_json(output,{'schema_version':schema['schema_version'],'schema':copy.deepcopy(schema),
        'weights':{'path':os.path.relpath(weights,output.parent),'sha256':digest_file(weights),'tensor_map_sha256':TENSOR_SHA},
        'encoder_sha256':encoder_identity(bundle),'slides':slides})
    return output


class HIPTPyramid:
    def __init__(self,bundle,manifest=None):
        if manifest is None:
            registration=bundle.encoder.get('hipt')
            if not isinstance(registration,dict) or set(registration)!={'manifest'} or not isinstance(registration['manifest'],str):
                raise ValueError('HIPT requires encoder.hipt.manifest')
            manifest=resolve(bundle.root,registration['manifest'])
        self.bundle=bundle;self.path=Path(manifest).resolve();self.meta=read_json(self.path)
        version=self.meta.get('schema_version')
        if (type(version) is not int or version not in {1,2} or
                self.meta.get('encoder_sha256')!=encoder_identity(bundle)):
            raise ValueError('HIPT pyramid encoder binding differs')
        schema_contract(self.meta['schema'],bundle)
        if version!=self.meta['schema']['schema_version']:
            raise ValueError('HIPT manifest and physical geometry schema versions differ')
        if set(self.meta['slides'])!=set(bundle.rows.slide_id):
            raise ValueError('HIPT pyramid manifest slide identities differ')
        entry=self.meta['weights']
        self.weights=resolve(self.path.parent,entry['path']).resolve()
        if entry.get('tensor_map_sha256')!=TENSOR_SHA or digest_file(self.weights)!=entry['sha256']:
            raise ValueError('HIPT registered pretrained weight file changed')
        self.backbone=load_backbone(self.weights)
        self.contract=digest_file(self.path)
        self.files=[self.path,self.weights,*[resolve(self.path.parent,e['path']) for e in self.meta['slides'].values()]]
        if len({p.resolve() for p in self.files})!=len(self.files):
            raise ValueError('HIPT prepared input files must be distinct')

    def inventory(self,row,deep=False):
        entry=self.meta['slides'][row['slide_id']]
        record={'manifest_sha256':self.contract,'pyramid_sha256':entry['sha256'],
                'teacher_tensor_sha256':TENSOR_SHA,'regions':entry['regions']}
        if deep:self.load(row)
        return record

    def load(self,row):
        if digest_file(self.path)!=self.contract:raise ValueError('HIPT pyramid manifest changed')
        entry=self.meta['slides'][row['slide_id']];path=resolve(self.path.parent,entry['path'])
        if (digest_file(path)!=entry['sha256'] or
                digest_file(resolve(self.bundle.root,row['path']))!=entry['primary_feature_sha256']):
            raise ValueError('HIPT prepared input files changed')
        sequence,coords,origins,order=aligned_inputs(self.bundle,row,path,self.meta['schema'])
        geometry=digest_object({'coords':coords.tolist(),'origins':origins.tolist(),'row_order':order.tolist()})
        if geometry!=entry['physical_geometry_sha256'] or len(sequence)!=entry['regions']:
            raise ValueError('HIPT physical pyramid mapping changed')
        return sequence,coords,origins,order

