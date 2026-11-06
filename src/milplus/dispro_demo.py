"""Explicitly synthetic incomplete patient WSI/RNA fixtures, no test dependencies."""
import io

import numpy as np
import pandas as pd
import yaml

from .demo import create_demo
from .io import write_json


def public_pathway_sizes():
    """Retain all actual public column sizes, never guess a missing 330th group."""
    from .dispro_source import _fixed
    from .upstream import pinned_text
    root,sha=_fixed()
    table=pd.read_csv(io.StringIO(pinned_text(root,sha,'datasets/pathway_signatures.csv')))
    sizes=[int(table[c].notna().sum()) for c in table]
    if len(sizes)!=331 or min(sizes)<1:
        raise ValueError('DisPro public pathway table changed from the reviewed complete331 columns')
    return sizes


def create_dispro_demo(output,patients=30,dimension=1024,pathway_sizes=None,instances=None):
    if instances is not None and (type(instances) is not int or instances<1):
        raise ValueError('DisPro synthetic instances must be a positive integer')
    sizes=public_pathway_sizes() if pathway_sizes is None else list(pathway_sizes)
    if not sizes or any(type(n) is not int or n<1 for n in sizes):
        raise ValueError('DisPro synthetic pathway sizes must all be positive integers')
    root=create_demo(output,patients=patients,dimension=dimension,survival=True)
    cfg=yaml.safe_load((root/'task.yaml').read_text())
    split=pd.read_csv(root/'split.csv')
    presence=[{'global_patient_id':p,'wsi_observed':int(i%4!=2),'omics_observed':int(i%4!=1)}
              for i,p in enumerate(split.patient_id)]
    pd.DataFrame(presence).to_csv(root/'presence.csv',index=False)
    groups=[{'name':f'synthetic_pathway_{i:03d}',
             'features':[f'synthetic_pathway_{i:03d}_gene_{j:03d}' for j in range(n)]}
            for i,n in enumerate(sizes)]
    genes=[g for path in groups for g in path['features']]
    rng=np.random.default_rng(2065)
    expression=pd.DataFrame(rng.normal(size=(patients,len(genes))),columns=genes)
    expression.insert(0,'global_patient_id',split.patient_id.tolist())
    available={p['global_patient_id'] for p in presence if p['omics_observed']}
    expression.loc[expression.global_patient_id.isin(available)].to_csv(root/'expression.csv',index=False)
    write_json(root/'pathways.json',groups)
    write_json(root/'dispro_inputs.json',{'schema_version':1,'kind':'dispro_incomplete_patient_inputs',
        'synthetic':True,'presence':'presence.csv','expression':'expression.csv','pathways':'pathways.json',
        'gene_columns':genes,'normalization':'train_standard',
        'missingness_provenance':'Explicit synthetic i%4 presence; absent assays have no RNA row, no actual clinical missingness claim',
        'assay_preprocessing':'Independent synthetic Gaussian values, not real gene measurements',
        'pathway_provenance':'Synthetic disjoint gene identifiers; group sizes match all331 reviewed author public columns' if pathway_sizes is None else
            'Explicit synthetic group sizes for input/plumbing checks, not the complete public331-pathway network'})
    cfg['encoders']['toy']['dispro']={'manifest':'dispro_inputs.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    index=pd.read_csv(root/'features.csv');slides=pd.read_csv(root/'slides.csv').set_index('slide_id')
    absent={p['global_patient_id'] for p in presence if not p['wsi_observed']}
    for i,row in index.iterrows():
        gid=slides.loc[row.slide_id,'patient_id']
        if gid in absent:index.loc[i,'path']='missing://wsi/'+gid
        elif instances is not None:
            import h5py
            with h5py.File(root/row['path'],'w') as f:
                f['features']=rng.normal(size=(instances,dimension)).astype(np.float32)
                f['coords']=np.column_stack([np.arange(instances)*256,np.zeros(instances)]).astype(np.int64)
            slides.loc[row['slide_id'],'width_level0']=instances*256
    if instances is not None:slides.to_csv(root/'slides.csv')
    index.to_csv(root/'features.csv',index=False)
    return root
