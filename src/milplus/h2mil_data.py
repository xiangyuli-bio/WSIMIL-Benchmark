"""H2-MIL data bridge: actual thumbnail/coarse/fine features and physical grids."""
import copy
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .data import load_array, resolve
from .io import digest_file, digest_object, read_json, write_json

COMMIT='ea472fefd5eab553c58041a5acc203ac7d467d06'

def encoder_identity(bundle):
    return digest_object({'fm':bundle.fm,'encoder':{k:v for k,v in bundle.encoder.items() if k!='h2mil'}})

def schema_contract(bundle,schema):
    if (schema.get('schema_version')!=1 or schema.get('kind')!='h2mil_thumbnail_coarse_fine' or
        schema.get('source_commit')!=COMMIT or type(schema.get('synthetic')) is not bool or
        schema['synthetic']!=bool(bundle.config.get('synthetic',False)) or bundle.task_type!='classification' or
        bundle.config.get('training_unit','slide')!='slide'):
        raise ValueError('H2-MIL requires its explicit physical three-level classification schema')
    spatial=bundle.encoder.get('spatial',{})
    if schema.get('coordinate_frame')!='level0_pixels' or spatial.get('unit')!='level0_pixels' or spatial.get('origin')!='top_left':
        raise ValueError('H2-MIL requires a common physical level0 top-left coordinate frame')
    stride=schema.get('fine_stride_level0')
    if type(stride) is not int or stride<=0 or spatial.get('patch_size')!=stride:
        raise ValueError('H2-MIL fine physical stride must match the primary nonoverlapping patch size')
    for key in ['patching_provenance','thumbnail_provenance','coarse_feature_provenance','fine_feature_provenance','encoder_provenance']:
        if not isinstance(schema.get(key),str) or not schema[key].strip():raise ValueError('Missing '+key)
    if schema.get('encoder_identity_sha256')!=encoder_identity(bundle):
        raise ValueError('H2-MIL all three real image levels must share the declared feature space')

def integer_grid(value,size,name):
    if value.shape!=(size,2) or value.dtype.kind not in 'iu' or (value<0).any() or (value>2**31-1).any() or len(np.unique(value,axis=0))!=size:
        raise ValueError('H2-MIL invalid unique '+name+' grid coordinates')
    return value.astype(np.int64)

def level_edges(grid,offset):
    table={tuple(x):i+offset for i,x in enumerate(grid.tolist())};edges=[]
    for (x,y),i in table.items():
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if (dx or dy) and (x+dx,y+dy) in table:edges.append((i,table[x+dx,y+dy]))
    return edges

def graph_inputs(bundle,row,path,schema):
    fine,_=bundle.load(row,require_coords=True)
    primary=resolve(bundle.root,row.get('coords_path') or row['path'])
    coordinates=np.asarray(load_array(primary,'coords'))
    expected={'thumbnail_features','coarse_features','coarse_grid','fine_grid','fine_parent'}
    with h5py.File(path,'r') as h:
        if set(h)!=expected:raise ValueError('H2-MIL hierarchy file must contain only actual level features and geometry, no outcomes')
        thumbnail=np.asarray(h['thumbnail_features'][:]);coarse=np.asarray(h['coarse_features'][:])
        grid_c=np.asarray(h['coarse_grid'][:]);grid_f=np.asarray(h['fine_grid'][:]);parents=np.asarray(h['fine_parent'][:])
    if (coarse.ndim!=2 or coarse.shape[1]!=bundle.dim or not len(coarse) or
        thumbnail.shape!=(1,bundle.dim) or any(x.dtype.kind!='f' or not np.isfinite(x).all() for x in [thumbnail,coarse])):
        raise ValueError('H2-MIL needs actual finite thumbnail and coarse features in the primary encoder space')
    grid_c=integer_grid(grid_c,len(coarse),'coarse');grid_f=integer_grid(grid_f,len(fine),'fine')
    if (parents.shape!=(len(fine),) or parents.dtype.kind not in 'iu' or
        (parents<0).any() or (parents>=len(coarse)).any() or
        set(parents.tolist())!=set(range(len(coarse)))):
        raise ValueError('H2-MIL every fine node needs a real coarse parent; no empty coarse substructures')
    if not np.array_equal(grid_f//2,grid_c[parents]):
        raise ValueError('H2-MIL parent edges disagree with actual twofold image pyramid geometry')
    if coordinates.shape!=grid_f.shape or not np.array_equal(coordinates,grid_f*schema['fine_stride_level0']):
        raise ValueError('H2-MIL supplied grid must agree exactly with the real primary physical coordinates')
    if (grid_c.max(0)==0).any() or (grid_f.max(0)==0).any():
        raise ValueError('Original H2-MIL per-level axis normalization needs a nonzero physical extent on both axes')
    # Exactly the author's independent per-level grid-axis normalization.
    normalized=np.concatenate([np.zeros((1,2)),grid_c/grid_c.max(0),grid_f/grid_f.max(0)]).astype(np.float32)
    tree=np.concatenate([[-1],np.zeros(len(coarse),dtype=np.int64),parents+1])
    node_type=np.concatenate([[0],np.ones(len(coarse),dtype=np.int64),np.full(len(fine),2,dtype=np.int64)])
    edges=[]
    for child,parent in enumerate(tree):
        if parent>=0:edges.extend([(child,int(parent)),(int(parent),child)])
    edges+=level_edges(grid_c,1)+level_edges(grid_f,1+len(coarse))
    edges=np.array(sorted(set(edges)),dtype=np.int64).T
    values=torch.cat([torch.from_numpy(thumbnail).float(),torch.from_numpy(coarse).float(),fine])
    graph=Data(x=values,batch=torch.zeros(len(values),dtype=torch.long),edge_index_tree_8nb=torch.from_numpy(edges),
        node_type=torch.from_numpy(node_type),node_tree=torch.from_numpy(tree),x_y_index=torch.from_numpy(normalized),data_id=row['slide_id'],
        physical_coordinates=torch.from_numpy(np.concatenate([np.zeros((1,2),dtype=np.int64),
            grid_c*schema['fine_stride_level0']*2,grid_f*schema['fine_stride_level0']])))
    geometry=digest_object({'coarse_grid':grid_c.tolist(),'fine_grid':grid_f.tolist(),'parent':parents.tolist(),
        'coordinates':coordinates.tolist(),'tree':tree.tolist(),'edges':edges.tolist()})
    return graph,geometry

def prepare(bundle,index,schema,output):
    schema_contract(bundle,schema);output=Path(output).resolve()
    if output.exists():raise ValueError('H2-MIL manifest output must be new')
    frame=pd.read_csv(index,dtype=str,keep_default_na=False)
    if set(frame)!={'slide_id','path'} or frame.slide_id.duplicated().any() or set(frame.slide_id)!=set(bundle.rows.slide_id):
        raise ValueError('H2-MIL exact slide inventory and no labels required')
    paths={r.slide_id:resolve(Path(index).resolve().parent,r.path).resolve() for r in frame.itertuples()}
    if len(set(paths.values()))!=len(paths):raise ValueError('H2-MIL slides cannot reuse one physical hierarchy file')
    entries={}
    for row in bundle.rows.to_dict('records'):
        path=paths[row['slide_id']];graph,geometry=graph_inputs(bundle,row,path,schema)
        entries[row['slide_id']]={'path':os.path.relpath(path,output.parent),'sha256':digest_file(path),
            'primary_sha256':digest_file(resolve(bundle.root,row['path'])),
            'coordinate_sha256':digest_file(resolve(bundle.root,row.get('coords_path') or row['path'])),
            'geometry_sha256':geometry,'nodes':len(graph.x)}
    write_json(output,{'schema_version':1,'schema':copy.deepcopy(schema),'encoder_sha256':encoder_identity(bundle),'slides':entries})
    return output

class H2Hierarchy:
    def __init__(self,bundle,manifest=None):
        if manifest is None:
            registration=bundle.encoder.get('h2mil')
            if not isinstance(registration,dict) or set(registration)!={'manifest'} or not isinstance(registration['manifest'],str):
                raise ValueError('H2-MIL requires encoder.h2mil.manifest')
            manifest=resolve(bundle.root,registration['manifest'])
        self.bundle=bundle;self.path=Path(manifest).resolve();self.meta=read_json(self.path)
        if self.meta.get('schema_version')!=1 or self.meta.get('encoder_sha256')!=encoder_identity(bundle):
            raise ValueError('H2-MIL encoder binding differs')
        schema_contract(bundle,self.meta['schema'])
        if set(self.meta['slides'])!=set(bundle.rows.slide_id):raise ValueError('H2-MIL slide inventory changed')
        self.contract=digest_file(self.path)
        self.files=[self.path,*[resolve(self.path.parent,e['path']).resolve() for e in self.meta['slides'].values()]]
        if len(set(self.files))!=len(self.files):raise ValueError('H2-MIL slides require distinct physical hierarchy paths')
        primary=[resolve(bundle.root,r['path']).resolve() for r in bundle.rows.to_dict('records')]
        primary.extend(resolve(bundle.root,r['coords_path']).resolve() for r in bundle.rows.to_dict('records') if r.get('coords_path'))
        self.files=list(dict.fromkeys([*self.files,*primary]))
    def inventory(self,row,deep=False):
        entry=self.meta['slides'][row['slide_id']]
        result={'manifest_sha256':self.contract,'hierarchy_sha256':entry['sha256'],
            'geometry_sha256':entry['geometry_sha256'],'nodes':entry['nodes']}
        if deep:self.load(row)
        return result
    def load(self,row):
        if digest_file(self.path)!=self.contract:raise ValueError('H2-MIL manifest changed')
        entry=self.meta['slides'][row['slide_id']];path=resolve(self.path.parent,entry['path'])
        if digest_file(path)!=entry['sha256'] or digest_file(resolve(self.bundle.root,row['path']))!=entry['primary_sha256'] or digest_file(
            resolve(self.bundle.root,row.get('coords_path') or row['path']))!=entry['coordinate_sha256']:
            raise ValueError('H2-MIL physical hierarchy or primary features changed')
        graph,geometry=graph_inputs(self.bundle,row,path,self.meta['schema'])
        if geometry!=entry['geometry_sha256'] or len(graph.x)!=entry['nodes']:raise ValueError('H2-MIL prepared geometry changed')
        return graph
