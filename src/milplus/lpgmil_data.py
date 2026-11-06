"""TRAIN-only lymphocyte candidates, explicit hierarchical prototypes and hashes.

The author's proto_vector.pt and its preparer are not released. Thresholds and
Ward linkage below are declared reference choices, never recovered defaults.
Actual nucleus counts must be computed beforehand using a documented detector.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import cut_tree, linkage

from .data import load_array, resolve
from .io import digest_file, digest_object, read_json, write_json


def patch_ids(bundle,row,count):
    path = resolve(bundle.root,row['path'])
    if row.get('patch_ids_path'):
        names = load_array(resolve(bundle.root,row['patch_ids_path']),'patch_ids')
    elif path.suffix in {'.h5','.hdf5'}:
        names = load_array(path,'patch_ids')
    else:
        raise ValueError('Provide actual patch_ids_path for LPGMIL non-H5 feature bags')
    names = [v.decode('utf-8') if isinstance(v,bytes) else v for v in np.asarray(names).tolist()]
    if len(names)!=count or len(set(names))!=count or any(not isinstance(v,str) or not v for v in names):
        raise ValueError('LPGMIL patch IDs must be nonempty original unique strings matching feature rows')
    return names


def inventory(bundle):
    result=[]
    for subset in ['train','val','test']:
        for row in sorted(bundle.records(subset),key=lambda r:r['slide_id']):
            values,_ = bundle.load(row)
            names=patch_ids(bundle,row,len(values))
            result.append({'slide_id':row['slide_id'],'global_patient_id':row['global_patient_id'],
                'subset':subset,'target':int(row['target']),'count':len(values),'patch_ids':names,
                'path':str(resolve(bundle.root,row['path'])),'sha256':digest_file(resolve(bundle.root,row['path']))})
    return result


def prepare(bundle,counts,detector_provenance,output,min_fraction,min_nuclei):
    if bundle.task_type!='classification' or bundle.config.get('training_unit','slide')!='slide':
        raise ValueError('LPGMIL requires classification and individual WSI bags')
    if type(min_fraction) not in (float,int) or not np.isfinite(min_fraction) or not 0<min_fraction<=1:
        raise ValueError('Declare an explicit lymphocyte fraction in (0,1]')
    if type(min_nuclei) is not int or min_nuclei<1:
        raise ValueError('Declare an explicit positive minimum nucleus count')
    provenance=read_json(detector_provenance)
    if provenance.get('synthetic'):
        if not bundle.config.get('synthetic',False):
            raise ValueError('Synthetic nucleus counts cannot initialize a clinical LPGMIL task')
    else:
        for key in ['detector','version','weights_sha256','patch_geometry','nucleus_type_mapping']:
            if not provenance.get(key):raise ValueError('LPGMIL nucleus provenance missing '+key)
        sha=provenance['weights_sha256']
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Declare the real nucleus detector checkpoint SHA256')
    table=pd.read_csv(counts,dtype={'slide_id':str,'patch_id':str})
    required={'slide_id','patch_id','n_lymphocytes','n_nuclei'}
    if not required<=set(table.columns) or table.empty or table.isna().any().any():
        raise ValueError('Need complete TRAIN slide/patch nucleus counts')
    if table.duplicated(['slide_id','patch_id']).any():raise ValueError('Duplicate original nucleus-count patch identity')
    for key in ['n_lymphocytes','n_nuclei']:
        x=pd.to_numeric(table[key],errors='raise').to_numpy()
        if not np.isfinite(x).all() or (x<0).any() or (x!=np.floor(x)).any():
            raise ValueError('Nucleus counts must be finite nonnegative integers')
    if (table.n_lymphocytes>table.n_nuclei).any():raise ValueError('Lymphocyte count exceeds all nuclei')
    train=bundle.records('train');expected={r['slide_id'] for r in train}
    if set(table.slide_id)!=expected:
        raise ValueError('Prototype counts must contain every TRAIN slide and no VAL/TEST or unknown slide')
    all_inventory=inventory(bundle)
    index=table.set_index(['slide_id','patch_id']);values=[];candidate_ids=[]
    for row in sorted(train,key=lambda r:r['slide_id']):
        features,_=bundle.load(row);names=patch_ids(bundle,row,len(features))
        actual=table[table.slide_id==row['slide_id']]
        if set(actual.patch_id)!=set(names):raise ValueError('Nucleus patch IDs and actual TRAIN feature rows differ')
        for i,name in enumerate(names):
            counts_row=index.loc[(row['slide_id'],name)]
            if counts_row.n_nuclei>=min_nuclei and counts_row.n_lymphocytes/counts_row.n_nuclei>=min_fraction:
                values.append(features[i].numpy());candidate_ids.append([row['slide_id'],name,i])
    if len(values)<6:raise ValueError('Need at least six real lymphocyte-dense TRAIN patches for six prototypes')
    # No hidden random subsampling: hierarchical clustering retains all actual candidates.
    features=np.asarray(values,dtype=np.float32)
    distances=linkage(features.astype(np.float64),method='ward',metric='euclidean',optimal_ordering=False)
    assignment=cut_tree(distances,n_clusters=[6]).reshape(-1)
    centers=np.stack([features[assignment==i].mean(0) for i in range(6)]).astype(np.float32)
    if not np.isfinite(centers).all():raise ValueError('LPGMIL prototype preparation produced nonfinite centers')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(table).to_csv(root/'train_nucleus_counts.csv',index=False)
    write_json(root/'detector_provenance.json',provenance)
    np.save(root/'prototypes.npy',centers);np.save(root/'cluster_assignment.npy',assignment)
    manifest={'schema_version':1,'kind':'lpgmil_train_hierarchical_prototypes_reference',
        'feature_dimension':bundle.dim,'classes':bundle.classes,'split':bundle.split,
        'feature_inventory':all_inventory,'feature_inventory_sha256':digest_object(all_inventory),
        'prototypes':'prototypes.npy','prototypes_sha256':digest_file(root/'prototypes.npy'),
        'counts':'train_nucleus_counts.csv','counts_sha256':digest_file(root/'train_nucleus_counts.csv'),
        'detector':'detector_provenance.json','detector_sha256':digest_file(root/'detector_provenance.json'),
        'assignment':'cluster_assignment.npy','assignment_sha256':digest_file(root/'cluster_assignment.npy'),
        'candidates':candidate_ids,'train_patients':sorted({r['global_patient_id']for r in train}),
        'reference_choices':{'minimum_fraction':float(min_fraction),'minimum_nuclei':min_nuclei,
            'hierarchical_linkage':'Ward/euclidean_float64','cluster_cut':'exact_six_cut_tree',
            'centroid':'float32_mean_of_actual_candidates','candidate_sampling':'none',
            'source_original_prototype_file_available':False},
        'preparer_sha256':digest_file(__file__),'synthetic':bool(bundle.config.get('synthetic',False))}
    write_json(root/'manifest.json',manifest)
    return root/'manifest.json'


class PrototypeInputs:
    def __init__(self,bundle,manifest):
        self.bundle=bundle;self.path=resolve(bundle.root,manifest);self.manifest=read_json(self.path)
        self.root=self.path.parent;self.files=[self.path]
        m=self.manifest
        if m.get('schema_version')!=1 or m.get('kind')!='lpgmil_train_hierarchical_prototypes_reference':
            raise ValueError('LPGMIL requires its complete TRAIN-only prototype manifest')
        if m['feature_dimension']!=bundle.dim or m['classes']!=bundle.classes or m['split']!=bundle.split:
            raise ValueError('LPGMIL prototype FM width, classes or actual split changed')
        if m['synthetic'] and not bundle.config.get('synthetic',False):
            raise ValueError('Synthetic LPGMIL assets cannot initialize a clinical task')
        if m['train_patients']!=sorted({r['global_patient_id']for r in bundle.records('train')}):
            raise ValueError('LPGMIL prototypes belong to different actual TRAIN patients')
        for key in ['prototypes','counts','detector','assignment']:
            path=resolve(self.root,m[key]).resolve()
            if not path.is_relative_to(self.root) or digest_file(path)!=m[key+'_sha256']:
                raise ValueError('LPGMIL prototype asset changed or escaped manifest root: '+key)
            self.files.append(path)
        values=np.load(self.root/m['prototypes'],allow_pickle=False)
        if values.shape!=(6,bundle.dim) or values.dtype!=np.float32 or not np.isfinite(values).all():
            raise ValueError('LPGMIL needs actual finite float32 six-by-FM-width prototypes')
        self.prototypes=torch.from_numpy(values.copy())
        self.identity=digest_object({'manifest':digest_file(self.path),
            'files':{p.name:digest_file(p)for p in self.files},'preparer':m['preparer_sha256']})

    def verify(self):
        m=self.manifest
        for key in ['prototypes','counts','detector','assignment']:
            if digest_file(self.root/m[key])!=m[key+'_sha256']:
                raise ValueError('LPGMIL actual prepared asset changed: '+key)
        actual=inventory(self.bundle)
        if actual!=m['feature_inventory'] or digest_object(actual)!=m['feature_inventory_sha256']:
            raise ValueError('LPGMIL original feature bytes, patch IDs, patient labels or split changed')

