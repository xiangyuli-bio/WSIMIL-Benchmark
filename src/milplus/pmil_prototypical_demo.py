"""Explicitly synthetic RGB256 end-to-end software fixtures, never WSI results."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml


def create_demo(output,weights,patches=5):
    if type(patches) is not int or patches<5:raise ValueError('Use at least five real synthetic RGB patches for all original pseudo-groups')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);(root/'pixels').mkdir()
    rng=np.random.default_rng(97);slides=[];features=[];splits=[]
    for i in range(6):
        sid='synthetic_'+str(i);patient='synthetic_patient_'+str(i)
        path=root/'pixels'/(sid+'.h5')
        with h5py.File(path,'w') as f:
            f['rgb']=rng.integers(0,256,(patches,256,256,3),dtype=np.uint8)
            f.create_dataset('patch_ids',data=[sid+'/patch_'+str(j)+'.png' for j in range(patches)],dtype=h5py.string_dtype('utf-8'))
        slides.append({'slide_id':sid,'patient_id':patient,'global_patient_id':patient,'label':['negative','positive'][i%2]})
        features.append({'slide_id':sid,'path':str(path)});splits.append({'patient_id':patient,'subset':['train','val','test'][i//2]})
    pd.DataFrame(slides).to_csv(root/'slides.csv',index=False);pd.DataFrame(features).to_csv(root/'pixels.csv',index=False)
    pd.DataFrame(splits).to_csv(root/'split.csv',index=False)
    task={'schema_version':1,'task_id':'synthetic_prototypical_PMIL','synthetic':True,'task_type':'classification',
        'classes':['negative','positive'],'training_unit':'slide','evaluation_unit':'slide','slides':'slides.csv',
        'label_provenance':'synthetic software fixture only','defaults':{'fm':'raw_rgb','split':'fold0'},
        'splits':{'fold0':'split.csv'},'encoders':{'raw_rgb':{'dimension':3,'index':'pixels.csv',
            'image_identity':'synthetic original RGB256; not cached FM features'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False))
    from .data import Bundle
    from .pmil_prototypical_data import prepare,KIND,NORMALIZATION
    schema={'schema_version':1,'kind':KIND,'layout':'NHWC','dtype':'uint8','color':'RGB','patch_pixels':256,
        'normalization':NORMALIZATION,'pixel_key':'rgb','patch_ids_key':'patch_ids',
        'image_provenance':'synthetic256RGB software fixture, not clinical pathology',
        'patch_provenance':'synthetic unique ordered patch IDs','coordinate_frame':'path_identity_only',
        'initial_weights':str(Path(weights).resolve())}
    prepare(Bundle(root),schema,root/'pmil_prototypical_manifest.json')
    task['encoders']['raw_rgb']['pmil_prototypical']={'manifest':'pmil_prototypical_manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False));Bundle(root).check(deep=True)
    return root
