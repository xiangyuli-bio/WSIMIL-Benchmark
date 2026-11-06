"""Synthetic complete 16x16 physical grids paired with actual native HIPT teacher weights."""

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .hipt_data import prepare
from .hipt_source import COMMIT, PATCH_SHA


def create_hipt_demo(root,weights,classes=2,patients=None,regions=2):
    if type(regions) is not int or regions<1:
        raise ValueError('Synthetic HIPT requires a positive explicit region count')
    root=create_demo(root,patients=10*classes if patients is None else patients,dimension=384,classes=classes)
    config=yaml.safe_load((root/'task.yaml').read_text())
    config['encoders']['toy']['spatial']['patch_size']=512
    config['encoders']['toy']['embedding_space']='HIPT_ViT256_DINO'
    config['encoders']['toy']['weights_parent_sha256']=PATCH_SHA
    (root/'task.yaml').write_text(yaml.safe_dump(config))
    bundle=Bundle(root);(root/'pyramids').mkdir();rng=np.random.default_rng(2026);index=[]
    for row in bundle.rows.to_dict('records'):
        origins=np.array([[8192*r,0] for r in range(regions)],dtype=np.int64)
        offsets=np.array([[j*512,i*512] for i in range(16) for j in range(16)])
        coords=np.concatenate([offsets+origin for origin in origins])
        x=rng.normal(size=(256*regions,384)).astype('float32');x[:,row['target']]+=1.25
        ids=np.repeat(np.arange(regions),256)
        permutation=rng.permutation(256*regions)
        with h5py.File(root/row['path'],'w') as h:h['features']=x[permutation];h['coords']=coords[permutation]
        path=root/'pyramids'/f"{row['slide_id']}.h5"
        with h5py.File(path,'w') as h:h['region_ids']=ids[permutation];h['region_origins']=origins
        index.append({'slide_id':row['slide_id'],'path':str(path.relative_to(root))})
    slides=pd.read_csv(root/'slides.csv');slides['width_level0']=8192*regions;slides['height_level0']=8192
    slides.to_csv(root/'slides.csv',index=False)
    pd.DataFrame(index).to_csv(root/'pyramids.csv',index=False)
    schema={'schema_version':1,'kind':'hipt_complete_16x16_pyramid','source_commit':COMMIT,
        'synthetic':True,'grid_shape':[16,16],'coordinate_frame':'level0_pixels','patch_stride':512,
        'patch_encoder':'HIPT_ViT256_DINO','patch_encoder_weights_sha256':PATCH_SHA,
        **{k:'Synthetic Gaussian features and explicit physical grid for software checks; no actual DINO/WSI extraction' for k in
           ['patching_provenance','patch_encoder_provenance','preprocessing_provenance']}}
    prepare(bundle,root/'pyramids.csv',schema,weights,root/'hipt.json')
    config['encoders']['toy']['hipt']={'manifest':'hipt.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(config))
    return root

