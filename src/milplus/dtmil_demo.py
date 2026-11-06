"""Synthetic, explicitly physical-grid fixtures for complete DT-MIL software checks."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .demo import create_demo


def create_dtmil_demo(output,patients=20,dimension=1280,classes=2,grid_shape=(6,7),cached_views=False):
    if (not isinstance(grid_shape,(tuple,list)) or len(grid_shape)!=2
            or any(type(n) is not int or n<1 for n in grid_shape)):
        raise ValueError('DT-MIL synthetic grid_shape must contain two positive integers')
    root=Path(create_demo(output,patients=patients,dimension=dimension,classes=classes))
    cfg_path=root/'task.yaml';cfg=yaml.safe_load(cfg_path.read_text())
    cfg['evaluation_unit']='slide'
    cfg['encoders']['toy']['spatial']={'unit':'level0_pixels','origin':'top_left','patch_size':256,
        'patch_stride':256,'provenance':'Deterministic synthetic lattice only; no natural tissue/clinical claim'}
    if cached_views:
        cfg['encoders']['toy']['dtmil_cached_views']={'dataset':'train_views','provenance':'Fixed two synthetic perturbations aligned with registered val rows'}
    labels=pd.read_csv(root/'slides.csv').set_index('slide_id').label.to_dict()
    h,w=grid_shape;yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    coords=np.column_stack([xx.ravel()*256,yy.ravel()*256]).astype(np.int64)
    rng=np.random.default_rng(2063)
    for row in pd.read_csv(root/'features.csv').to_dict('records'):
        values=rng.normal(0,.2,(len(coords),dimension)).astype(np.float32)
        values[:,cfg['classes'].index(labels[row['slide_id']])]+=1
        with h5py.File(root/row['path'],'w') as f:
            f['features']=values;f['coords']=coords
            if cached_views:
                f['train_views']=np.stack([values+.01,values-.01],1)
    slides=pd.read_csv(root/'slides.csv');slides['width_level0']=w*256;slides['height_level0']=h*256
    slides.to_csv(root/'slides.csv',index=False)
    cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    return root
