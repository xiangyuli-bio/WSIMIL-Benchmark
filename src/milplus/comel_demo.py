"""Explicitly synthetic, full-dimensional continual task data for software checks."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .comel_data import prepare


def create_demo(output,tasks=3,patches=17,dimension=1024):
    if any(type(v) is not int or v<2 for v in [tasks,patches,dimension]):raise ValueError('Use at least two tasks/instances/dimensions')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir()
    rng=np.random.default_rng(2026);slides=[];splits=[];index=[];mapping={};tasklist=[];classes=[]
    for t in range(tasks):
        pair=[f'task{t}_negative',f'task{t}_positive'];classes+=pair;tasklist.append({'id':f'task{t}','classes':pair})
        for subset in ['train','val','test']:
            for target in [0,1]:
                sid=f't{t}_{subset}_{target}';path=root/'features'/(sid+'.h5')
                with h5py.File(path,'w') as f:
                    x=rng.normal(size=(patches,dimension)).astype('float32');x[:,:4]+=target*.2+t*.05
                    f['features']=x;f['patch_ids']=np.asarray([f'{sid}_original_patch_{i}'.encode() for i in range(patches)])
                slides.append({'slide_id':sid,'patient_id':sid,'label':pair[target]});splits.append({'patient_id':sid,'subset':subset})
                row={'slide_id':sid,'path':str(path)}
                if subset!='train':
                    p=root/'features'/(sid+'_annotations.npy');np.save(p,(np.arange(patches)%2 if target else np.zeros(patches)).astype('int64'))
                    row['instance_labels_path']=str(p)
                index.append(row);mapping[sid]=f'task{t}'
    pd.DataFrame(slides).to_csv(root/'slides.csv',index=False);pd.DataFrame(splits).to_csv(root/'split.csv',index=False)
    pd.DataFrame(index).fillna('').to_csv(root/'features.csv',index=False)
    config={'schema_version':1,'task_id':'synthetic_comel_continual','synthetic':True,'task_type':'classification',
        'classes':classes,'training_unit':'slide','evaluation_unit':'slide','defaults':{'fm':'synthetic_fm','split':'fold0'},
        'splits':{'fold0':'split.csv'},'encoders':{'synthetic_fm':{'dimension':dimension,'index':'features.csv'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    manifest=prepare(Bundle(root),{'schema_version':1,'tasks':tasklist,'slide_tasks':mapping,
        'provenance':'Explicitly synthetic random-feature engineering checks; not real WSI or paper-score reproduction'},root/'comel_manifest.json')
    config['encoders']['synthetic_fm']['comel']={'manifest':str(manifest)}
    (root/'task.yaml').write_text(yaml.safe_dump(config,sort_keys=False));return root
