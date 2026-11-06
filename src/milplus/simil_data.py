"""Patch-aligned, source-bound SI-MIL PathExpert assets; no inferred descriptors."""
import copy
import math
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .data import load_array, resolve
from .io import digest_file, digest_object, read_json, write_json
from .simil_normalization import PathExpertNormalization

COMMIT='3ffb218672ffe0e4a183924af50e5920b498483d'


def encoder_identity(bundle):
    return digest_object({'fm':bundle.fm,'encoder':{k:v for k,v in bundle.encoder.items() if k!='simil'}})


def validate_schema(schema,bundle):
    if schema.get('schema_version')!=1 or schema.get('kind')!='raw_filtered_pathexpert':
        raise ValueError('SI-MIL requires raw filtered PathExpert descriptors before training-only normalization')
    if type(schema.get('synthetic')) is not bool or schema['synthetic']!=bool(bundle.config.get('synthetic',False)):
        raise ValueError('SI-MIL synthetic input markers differ')
    names=schema.get('feature_names')
    if (not isinstance(names,list) or len(names)<2 or any(not isinstance(n,str) or not n.strip() for n in names)
            or len(set(names))!=len(names)):
        raise ValueError('SI-MIL requires exact ordered unique PathExpert feature names')
    for key in ['provenance','segmentation_model','feature_extraction_revision','patch_filtering_provenance','aligned_field_of_view_provenance']:
        if not isinstance(schema.get(key),str) or not schema[key].strip():
            raise ValueError('SI-MIL PathExpert schema requires '+key)
    weights=schema.get('segmentation_weights_sha256')
    if not isinstance(weights,str) or len(weights)!=64 or any(c not in '0123456789abcdef' for c in weights):
        raise ValueError('SI-MIL requires actual segmentation weight SHA256')
    if not schema['synthetic'] and weights=='0'*64:
        raise ValueError('A placeholder segmentation weight hash is not natural-data provenance')
    fov=schema.get('field_of_view_um')
    deep_fov=bundle.encoder.get('field_of_view_um')
    if (isinstance(fov,bool) or not isinstance(fov,(int,float)) or not math.isfinite(fov)
            or fov<=0 or isinstance(deep_fov,bool) or fov!=deep_fov):
        raise ValueError('SI-MIL deep and PathExpert patches must declare the same physical field of view')
    return names


def paired(bundle,row,path,dimension):
    x,_=bundle.load(row,require_coords=True)
    xy=np.asarray(load_array(resolve(bundle.root,row.get('coords_path') or row['path']),'coords'))
    aux=np.asarray(load_array(path))
    axy=np.asarray(load_array(path,'coords'))
    if (aux.shape!=(len(x),dimension) or aux.dtype.kind not in 'iuf' or np.isinf(aux).any()
            or xy.shape!=(len(x),2) or xy.dtype.kind not in 'iuf' or not np.isfinite(xy).all()
            or (xy<0).any() or len(np.unique(xy,axis=0))!=len(xy)
            or axy.dtype.kind not in 'iuf' or not np.array_equal(axy,xy)):
        raise ValueError('SI-MIL requires row-aligned real patch coordinates and finite-or-missing PathExpert descriptors')
    return x,aux,xy


def prepare(bundle,index,schema,output):
    validate_schema(schema,bundle)
    if bundle.task_type!='classification' or len(bundle.classes)!=2 or bundle.config.get('training_unit','slide')!='slide':
        raise ValueError('Native SI-MIL requires binary individual slide bags')
    output=Path(output).resolve()
    if output.exists():
        raise ValueError('SI-MIL manifest output must be new')
    table=pd.read_csv(index,dtype=str,keep_default_na=False)
    if set(table)!={'slide_id','path'} or table.slide_id.duplicated().any() or set(table.slide_id)!=set(bundle.rows.slide_id):
        raise ValueError('SI-MIL auxiliary index must cover exact slides without target columns or duplicates')
    paths={r.slide_id:resolve(Path(index).resolve().parent,r.path).resolve() for r in table.itertuples()}
    if len(set(paths.values()))!=len(paths):
        raise ValueError('SI-MIL auxiliary slides cannot share one physical descriptor file')
    slides={}
    for row in bundle.rows.to_dict('records'):
        sid=row['slide_id'];path=paths[sid]
        x,_,_=paired(bundle,row,path,len(schema['feature_names']))
        slides[sid]={'path':os.path.relpath(path,output.parent),'sha256':digest_file(path),
                     'primary_feature_sha256':digest_file(resolve(bundle.root,row['path'])),
                     'instances':len(x)}
    manifest={'schema_version':1,'source_commit':COMMIT,'encoder_sha256':encoder_identity(bundle),
              'schema':copy.deepcopy(schema),'slides':slides}
    write_json(output,manifest)
    return output


class PathExpertAssets:
    def __init__(self,bundle,manifest=None):
        if manifest is None:
            registration=bundle.encoder.get('simil')
            if not isinstance(registration,dict) or set(registration)!={'manifest'} or not isinstance(registration['manifest'],str):
                raise ValueError('SI-MIL requires encoder.simil.manifest')
            manifest=resolve(bundle.root,registration['manifest'])
        self.bundle=bundle;self.path=Path(manifest).resolve();self.meta=read_json(self.path)
        m=self.meta
        if m.get('schema_version')!=1 or m.get('source_commit')!=COMMIT or m.get('encoder_sha256')!=encoder_identity(bundle):
            raise ValueError('SI-MIL descriptor manifest source or encoder identity differs')
        self.names=validate_schema(m['schema'],bundle)
        self.dimension=len(self.names)
        if set(m['slides'])!=set(bundle.rows.slide_id):
            raise ValueError('SI-MIL descriptor manifest must cover exact task slides')
        if len({resolve(self.path.parent,v['path']).resolve() for v in m['slides'].values()})!=len(m['slides']):
            raise ValueError('SI-MIL descriptor files must be distinct per slide')
        self.files=[self.path,*[resolve(self.path.parent,v['path']) for v in m['slides'].values()]]
        self.contract=digest_file(self.path)

    def load(self,row):
        if digest_file(self.path)!=self.contract:
            raise ValueError('SI-MIL descriptor manifest changed')
        entry=self.meta['slides'][row['slide_id']]
        path=resolve(self.path.parent,entry['path'])
        if digest_file(path)!=entry['sha256'] or digest_file(resolve(self.bundle.root,row['path']))!=entry['primary_feature_sha256']:
            raise ValueError('SI-MIL prepared descriptors or deep features changed')
        x,aux,xy=paired(self.bundle,row,path,self.dimension)
        if len(x)!=entry['instances']:
            raise ValueError('SI-MIL original row count changed')
        return x,aux,xy

    def inventory(self,row,deep=False):
        entry=self.meta['slides'][row['slide_id']]
        result={'manifest_sha256':self.contract,'descriptor_sha256':entry['sha256'],
                'dimension':self.dimension,'instances':entry['instances']}
        if deep:self.load(row)
        return result

    def fit_training_normalization(self,workspace_budget_bytes=2*1024**3):
        """Exact TRAIN quantiles, bounded by column blocks, without sampling patients/patches.

        HDF5 descriptors are sliced on disk. Other supported formats require a whole
        slide read and therefore include its size in the explicit workspace budget.
        The budget covers estimated numeric arrays, not Python/Pandas process RSS.
        """
        if type(workspace_budget_bytes) is not int or workspace_budget_bytes<1:
            raise ValueError('SI-MIL normalization workspace budget must be a positive integer')
        rows=self.bundle.records('train')
        if not rows:raise ValueError('SI-MIL normalization requires TRAIN slides')
        entries=[self.meta['slides'][r['slide_id']] for r in rows]
        total=sum(e['instances'] for e in entries)
        # Concatenation, float64 conversion, bins and quantile scratch space.
        one_column=total*8*6
        fallback=max((e['instances']*self.dimension*8 for e in entries
                      if resolve(self.path.parent,e['path']).suffix.lower() not in {'.h5','.hdf5'}),default=0)
        width=min(self.dimension,(workspace_budget_bytes-fallback)//max(1,one_column))
        if width<1:
            raise ValueError('SI-MIL exact TRAIN quantiles exceed the normalization workspace budget; increase it explicitly')
        # Input alignment/value checks are streamed by slide, before and after fitting.
        for row in rows:self.load(row)
        norm=PathExpertNormalization();norm.edges=[];norm.means=[];norm.stds=[]
        for start in range(0,self.dimension,width):
            stop=min(start+width,self.dimension);columns=[]
            for entry in entries:
                path=resolve(self.path.parent,entry['path'])
                if path.suffix.lower() in {'.h5','.hdf5'}:
                    with h5py.File(path,'r') as h:
                        columns.append(np.asarray(h['features'][:,start:stop],dtype=np.float64))
                else:
                    columns.append(np.asarray(load_array(path)[:,start:stop],dtype=np.float64).copy())
            block=PathExpertNormalization().fit(np.concatenate(columns,axis=0))
            norm.edges.extend(block.edges);norm.means.extend(block.means);norm.stds.extend(block.stds)
        for row in rows:self.load(row)
        self.fit_plan={'scope':'TRAIN patches only, exact quantiles and population standard deviations',
            'training_slides':[r['slide_id'] for r in rows],'training_instances':total,
            'column_block_size':width,'workspace_budget_bytes':workspace_budget_bytes,
            'estimated_numeric_workspace_bytes':width*one_column+fallback}
        return norm

