"""Actual synthetic complete feature bags; no clinical data or pretrained FM claim."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .mexd_graph import COHORTS


def create_demo(output,cohort='tcga_nsclc',dimension=768,patches=1025,classes=None,background=False):
    if cohort not in set(COHORTS)|{'class_port'} or type(dimension)is not int or dimension<1 or type(patches)is not int or patches<2:
        raise ValueError('Declare a MExD cohort, positive FM width and at least2real patches')
    if cohort=='class_port':
        classes=5 if classes is None else classes
        if type(classes)is not int or classes<2:raise ValueError('MExD class-port demo needs at least2classes')
    else:
        if classes is not None and classes!=COHORTS[cohort]['classes']:raise ValueError('Native MExD cohort class count changed; use class_port')
        classes=COHORTS[cohort]['classes']
        if background and cohort!='camelyon16':raise ValueError('Observed background differs from native cohort; use class_port')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir()
    rng=np.random.default_rng(108)
    slides=[];features=[];splits=[]
    for subset in ['train','val','test']:
        for i in range(2*classes):
            sid=f'synthetic_{subset}_{i:03d}';label=i%classes
            value=rng.normal(0,.5,(patches,dimension)).astype(np.float32);value[:,label%dimension]+=.3
            relative='features/'+sid+'.h5'
            with h5py.File(root/relative,'w')as f:
                f['features']=value;f['patch_ids']=np.asarray([f'{sid}_original_{j}'.encode()for j in range(patches)])
            slides.append({'slide_id':sid,'patient_id':sid,'global_patient_id':sid,'label':f'class_{label}'})
            features.append({'slide_id':sid,'path':relative});splits.append({'patient_id':sid,'subset':subset})
    for name,records in [('slides',slides),('features',features),('split',splits)]:
        pd.DataFrame(records).to_csv(root/(name+'.csv'),index=False)
    task={'schema_version':1,'task_id':'synthetic_mexd_'+cohort,'task_type':'classification','synthetic':True,
        'classes':[f'class_{i}'for i in range(classes)],'training_unit':'slide','evaluation_unit':'patient',
        'slides':'slides.csv','defaults':{'fm':'synthetic_fm','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'synthetic_fm':{'dimension':dimension,'index':'features.csv','weights':'synthetic_only',
            'preprocessing':'actual ordered independent Gaussian rows with original synthetic patch IDs'}}}
    if cohort=='camelyon16'or background:task['class_roles']={'background':'class_0'}
    (root/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False))
    return root
