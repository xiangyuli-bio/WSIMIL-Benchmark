"""Explicitly synthetic H2 thumbnail/coarse/fine fixture; no fabricated real FM provenance."""
import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .demo import create_demo
from .h2mil_data import COMMIT, encoder_identity, prepare


def create_h2mil_demo(root,patients=20,classes=2,dimension=1024,coarse_side=2):
    if type(coarse_side) is not int or coarse_side<2:raise ValueError('Synthetic H2 hierarchy requires at least two coarse grid cells along each axis')
    root=create_demo(root,patients=patients,classes=classes,dimension=dimension)
    cfg=yaml.safe_load((root/'task.yaml').read_text())
    cfg['encoders']['toy']['spatial']['provenance']='Explicit synthetic twofold grid and independent Gaussian level embeddings'
    (root/'task.yaml').write_text(yaml.safe_dump(cfg));bundle=Bundle(root)
    coarse=np.array([[x,y] for y in range(coarse_side) for x in range(coarse_side)],dtype=np.int64)
    fine=np.array([[x,y] for y in range(2*coarse_side) for x in range(2*coarse_side)],dtype=np.int64)
    parents=fine[:,1]//2*coarse_side+fine[:,0]//2
    (root/'hierarchies').mkdir();index=[];rng=np.random.default_rng(2026)
    for row in bundle.rows.to_dict('records'):
        permutation=rng.permutation(len(fine))
        features=rng.normal(size=(len(fine),dimension)).astype('float32');features[:,row['target']]+=1.25
        thumbnail=rng.normal(size=(1,dimension)).astype('float32');thumbnail[:,row['target']]+=1.25
        middle=rng.normal(size=(len(coarse),dimension)).astype('float32');middle[:,row['target']]+=1.25
        with h5py.File(root/row['path'],'w') as h:h['features']=features[permutation];h['coords']=fine[permutation]*256
        path=root/'hierarchies'/f"{row['slide_id']}.h5"
        with h5py.File(path,'w') as h:
            h['thumbnail_features']=thumbnail;h['coarse_features']=middle;h['coarse_grid']=coarse
            h['fine_grid']=fine[permutation];h['fine_parent']=parents[permutation]
        index.append({'slide_id':row['slide_id'],'path':str(path.relative_to(root))})
    slides=pd.read_csv(root/'slides.csv');slides['width_level0']=coarse_side*512;slides['height_level0']=coarse_side*512
    slides.to_csv(root/'slides.csv',index=False);pd.DataFrame(index).to_csv(root/'hierarchies.csv',index=False)
    schema={'schema_version':1,'kind':'h2mil_thumbnail_coarse_fine','source_commit':COMMIT,'synthetic':True,
        'coordinate_frame':'level0_pixels','fine_stride_level0':256,'encoder_identity_sha256':encoder_identity(bundle),
        **{k:'Synthetic Gaussian level embeddings and explicit twofold physical grid; no real KimiaNet or WSI extraction' for k in
           ['patching_provenance','thumbnail_provenance','coarse_feature_provenance','fine_feature_provenance','encoder_provenance']}}
    prepare(bundle,root/'hierarchies.csv',schema,root/'hierarchy.json')
    cfg['encoders']['toy']['h2mil']={'manifest':'hierarchy.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(cfg));return root
