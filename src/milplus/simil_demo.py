"""Explicitly synthetic patch-aligned SI-MIL task fixtures, with no biological meaning."""

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle, load_array, resolve
from .demo import create_demo
from .simil_data import prepare


def create_simil_demo(root, dimension=384, descriptors=246):
    data=create_demo(root,patients=20,dimension=dimension)
    config=yaml.safe_load((data/'task.yaml').read_text())
    config['encoders']['toy']['field_of_view_um']=448.
    (data/'task.yaml').write_text(yaml.safe_dump(config))
    bundle=Bundle(data);(data/'pathexpert').mkdir()
    schema={'schema_version':1,'kind':'raw_filtered_pathexpert','synthetic':True,
            'feature_names':[f'synthetic_descriptor_{i:03d}' for i in range(descriptors)],
            'field_of_view_um':448.,'segmentation_weights_sha256':'0'*64,
            **{key:'synthetic Gaussian descriptors; no nuclei or clinical meaning' for key in
               ['provenance','segmentation_model','feature_extraction_revision','patch_filtering_provenance','aligned_field_of_view_provenance']}}
    rng=np.random.default_rng(2026);index=[]
    for row in bundle.rows.to_dict('records'):
        x,_=bundle.load(row,require_coords=True)
        xy=load_array(resolve(data,row['path']),'coords')
        aux=rng.normal(size=(len(x),descriptors)).astype('float32')
        aux[rng.random(aux.shape)<.05]=np.nan
        path=data/'pathexpert'/f"{row['slide_id']}.h5"
        with h5py.File(path,'w') as h:
            h['features']=aux;h['coords']=xy
        index.append({'slide_id':row['slide_id'],'path':str(path.relative_to(data))})
    pd.DataFrame(index).to_csv(data/'pathexpert.csv',index=False)
    manifest=prepare(bundle,data/'pathexpert.csv',schema,data/'pathexpert.json')
    config['encoders']['toy']['simil']={'manifest':str(manifest.relative_to(data))}
    (data/'task.yaml').write_text(yaml.safe_dump(config))
    return data
