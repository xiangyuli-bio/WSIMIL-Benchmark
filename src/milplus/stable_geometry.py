"""StableMIL's physical fusion, oriented regions and frozen nearest-neighbour masks."""
import errno
import importlib.metadata
import json
import math
import os
import tempfile
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

from .io import digest_file, digest_object, write_json
from .smmile_data import array_identity, positive_int
from .spatial_attention import checked_points
from .upstream import load_definitions, verified_source


def author_geometry():
    return geometry_functions(verified_source('stablemil')[1])


@lru_cache(maxsize=1)
def geometry_functions(commit):
    scope={'torch':torch,'np':np,'ConvexHull':ConvexHull,'NearestNeighbors':NearestNeighbors}
    def add(path,names):
        module=load_definitions('stablemil',path,names,scope)
        scope.update({name:getattr(module,name) for name in names})
    add('models/token_fuse.py',['create_bounding_box','assign_labels_to_regions','fuse_labels'])
    add('models/find_Region.py',['minimum_bounding_rectangle','distance_to_line_vectorized','assign_coords_to_regions_vectorized','find_region_ag'])
    add('classification/preprocess/preprocess.py',['fuse_token','region_sort','create_tokens'])
    return scope


NATIVE_RADIUS=6*math.sqrt(2)


def build_geometry(coords,reference_size,ratio=2,aggregate_num=256,k_neighbors=8,radius=NATIVE_RADIUS,repaired=False,max_bytes=1024**3):
    for name,value in [('ratio',ratio),('aggregate_num',aggregate_num),('k_neighbors',k_neighbors),('max_dense_bytes',max_bytes)]:positive_int(value,name)
    if not isinstance(coords,torch.Tensor) or coords.is_complex() or coords.dtype==torch.bool:
        raise ValueError('StableMIL requires real coordinate tensors')
    points=checked_points(coords).astype(np.float64)
    if (points<0).any() or (abs(points)>2**52).any() or not math.isfinite(reference_size) or reference_size<=0 or not math.isfinite(radius) or radius<=0:
        raise ValueError('StableMIL coordinates/reference_size/radius must be finite positive in the registered physical frame')
    coarse=np.floor(points/reference_size).astype(np.int64)
    span=np.ptp(coarse,axis=0);grid=tuple(max(1,math.ceil(int(s)/ratio)) for s in span)
    if grid[0]*grid[1]*16+len(points)*160+ratio**2*len(points)*16>max_bytes:
        raise ValueError('StableMIL physical fusion labels exceed max_dense_bytes before allocation')
    native=author_geometry();repairs=[]
    if (span==0).any():
        if not repaired:raise ValueError('StableMIL source fusion requires nonzero extent on both axes; choose the named repaired route')
        normalized=[]
        for axis,count in enumerate(grid):
            normalized.append(np.zeros(len(coarse),dtype=np.int64) if span[axis]==0 else
                np.floor((coarse[:,axis]-coarse[:,axis].min())/(span[axis]/count)).clip(0,count-1).astype(np.int64))
        labels=torch.from_numpy(normalized[1]*grid[0]+normalized[0]);order=torch.argsort(labels)
        groups=torch.split(torch.from_numpy(coarse)[order].float(),torch.bincount(labels).tolist())
        groups=[g for g in groups if len(g)]
        padded=[g[:ratio**2] if len(g)>=ratio**2 else torch.cat([g,g.mean(0).expand(ratio**2-len(g),-1)]) for g in groups]
        fused=torch.stack(padded).mean(1)/ratio
        labels,order,fused=labels.numpy(),order.numpy(),fused.numpy();repairs.append('zero-extent physical axis uses one bin')
    else:labels,order,fused=native['fuse_token'](coarse,ratio)
    if len(fused)<5 or np.linalg.matrix_rank(fused-fused.mean(0))<2:
        if not repaired:raise ValueError('StableMIL source oriented regions require >=5 noncollinear fused points')
        regions=np.zeros(len(fused),dtype=np.int64);region_order=np.argsort(regions)
        repairs.append('degenerate physical geometry uses one semantic region')
    else:regions,region_order=native['region_sort'](fused,aggregate_num)
    regions=np.asarray(regions,dtype=np.int64);region_order=np.asarray(region_order,dtype=np.int64)
    indices,outside=native['create_tokens'](fused[region_order],k_neighbors,radius)
    retained=[]
    for label in np.unique(labels):retained.extend(order[labels[order]==label][:ratio**2].tolist())
    arrays={'fuse_labels':np.asarray(labels,dtype=np.int64),'fuse_sorted_idx':np.asarray(order,dtype=np.int64),
        'fuse_coords':np.asarray(fused,dtype=np.float32),'region_indices':regions,'region_sorted_index':region_order,
        'attention_mask_1':np.asarray(indices,dtype=np.int64),'attention_mask_2':np.asarray(outside,dtype=np.int64),
        'retained_rows':np.asarray(retained,dtype=np.int64)}
    metadata={'coordinate_sha256':array_identity(points)['sha256'],'reference_size':float(reference_size),'ratio':ratio,'aggregate_num':aggregate_num,
        'k_neighbors':k_neighbors,'radius':radius,'raw_instances':len(points),'fused_instances':len(fused),
        'semantic_regions':len(np.unique(regions)),'retained_instances':len(retained),'repairs':repairs,
        'source_commit':verified_source('stablemil')[1],'arrays':{k:array_identity(v) for k,v in arrays.items()}}
    metadata['geometry_sha256']=digest_object(metadata)
    return arrays,metadata


class GeometryCache:
    """Bounded, input-hashed geometry only; neither labels nor cohort fitting enter it."""
    def __init__(self,max_bytes=128*1024**2):
        self.max_bytes=positive_int(max_bytes,'geometry_cache_bytes');self.records=OrderedDict();self.bytes=0;self.directory=None
        self.identity={'source_commit':verified_source('stablemil')[1], 'versions':{name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','scikit-learn']}}

    def get(self,coords,**options):
        key=digest_object({'coordinates':array_identity(coords.detach().cpu().numpy()),'options':options,'implementation':self.identity})
        if key not in self.records:
            location=None if self.directory is None else self.directory/key
            if location is not None and location.exists():
                record=json.loads((location/'record.json').read_text());path=location/'arrays.h5'
                if record['input_key']!=key or digest_file(path)!=record['sha256']:
                    raise ValueError('StableMIL persisted geometry identity/hash mismatch')
                meta=record['metadata']
                if digest_object({k:v for k,v in meta.items() if k!='geometry_sha256'})!=meta['geometry_sha256']:
                    raise ValueError('StableMIL geometry metadata changed')
                with h5py.File(path,'r') as h:arrays={k:h[k][:] for k in h}
                if set(arrays)!=set(meta['arrays']) or any(array_identity(arrays[k])!=v for k,v in meta['arrays'].items()):
                    raise ValueError('StableMIL persisted geometry arrays changed')
            else:
                arrays,meta=build_geometry(coords,**options)
                if location is not None:
                    self.directory.mkdir(parents=True,exist_ok=True)
                    temporary=Path(tempfile.mkdtemp(prefix='.'+key+'-',dir=self.directory))
                    with h5py.File(temporary/'arrays.h5','w') as h:
                        for name,value in arrays.items():h[name]=value
                    write_json(temporary/'record.json',{'input_key':key,'metadata':meta,'sha256':digest_file(temporary/'arrays.h5')})
                    try:os.rename(temporary,location)  # Atomic completed directory; concurrent identical writers never expose a partial pair.
                    except OSError as error:
                        if error.errno not in {errno.EEXIST,errno.ENOTEMPTY}:raise
                        (temporary/'arrays.h5').unlink();(temporary/'record.json').unlink();temporary.rmdir()
                        return self.get(coords,**options)
            size=sum(v.nbytes for v in arrays.values())
            while self.records and self.bytes+size>self.max_bytes:
                _,(_,_,removed)=self.records.popitem(last=False);self.bytes-=removed
            if size<=self.max_bytes:self.records[key]=(arrays,meta,size);self.bytes+=size
            return arrays,meta
        self.records.move_to_end(key);arrays,meta,_=self.records[key]
        if any(array_identity(arrays[k])!=v for k,v in meta['arrays'].items()):raise ValueError('StableMIL geometry cache changed')
        return arrays,meta
