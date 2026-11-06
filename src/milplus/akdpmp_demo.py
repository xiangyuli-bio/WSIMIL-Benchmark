"""Explicit synthetic engineering tasks; never substitute for real WSI evidence."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .akdpmp_data import prepare


def create_demo(output,preset='tcga',patches=41,task_sizes=None,dimension=None):
    if preset not in {'tcga','skin'} or type(patches) is not int or patches<2:raise ValueError('Use an actual source preset and at least two patches')
    sizes=task_sizes or ([2,2,3] if preset=='tcga' else [2,2,2]);dim=dimension or (1024 if preset=='tcga' else 512)
    if len(sizes)<2 or any(type(v) is not int or v<1 for v in sizes) or type(dim) is not int or dim<1:raise ValueError('Declare nonempty actual task class lists and positive actual feature dimension')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir();rng=np.random.default_rng(2026)
    classes=[];tasks=[];mapping={};slides=[];splits=[];features=[]
    for t,count in enumerate(sizes):
        names=[f't{t}_class{k}' for k in range(count)];classes+=names;tasks.append({'id':f'task{t}','classes':names})
        for subset in ['train','val','test']:
            for cls,name in enumerate(names):
                sid=f't{t}_{subset}_{cls}';p=root/'features'/(sid+'.h5')
                with h5py.File(p,'w') as f:
                    x=rng.normal(size=(patches,dim)).astype('float32');x[:,:4]+=cls*.1+t*.05
                    f['features']=x;f['patch_ids']=np.asarray([f'{sid}_original_patch_{i}'.encode() for i in range(patches)])
                slides.append({'slide_id':sid,'patient_id':sid,'label':name});splits.append({'patient_id':sid,'subset':subset});features.append({'slide_id':sid,'path':str(p)});mapping[sid]=f'task{t}'
    for name,rows in [('slides',slides),('split',splits),('features',features)]:pd.DataFrame(rows).to_csv(root/(name+'.csv'),index=False)
    config={'schema_version':1,'task_id':'synthetic_akdpmp_'+preset,'synthetic':True,'task_type':'classification','classes':classes,
        'training_unit':'slide','evaluation_unit':'slide','defaults':{'fm':'synthetic_fm','split':'fold0'},
        'splits':{'fold0':'split.csv'},'encoders':{'synthetic_fm':{'dimension':dim,'index':'features.csv'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    manifest=prepare(Bundle(root),{'schema_version':1,'tasks':tasks,'slide_tasks':mapping,
        'provenance':'Explicitly synthetic full-width software tasks; not WSI data or paper-score reproduction'},root/'akdpmp_manifest.json')
    config['encoders']['synthetic_fm']['akdpmp']={'manifest':str(manifest)};(root/'task.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    return root
