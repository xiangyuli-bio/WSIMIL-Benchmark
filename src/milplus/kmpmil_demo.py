"""Explicit synthetic four-task original-clinical-name fixture, not actual TCGA."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
import yaml
from .data import Bundle
from .kmpmil_data import prepare
from .kmpmil_source import source_text_assets
from .upstream import pinned_text


def create_demo(output,weights,sha256,patches=17,train_per_class=9):
    if type(patches) is not int or patches<2 or type(train_per_class) is not int or train_per_class<1:
        raise ValueError('KMP needs positive slide counts per class and two actual patches')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir()
    text,descriptors=source_text_assets()
    tasks=[];classes=[];descriptions={};mapping={};slides=[];splits=[];indices=[];rng=np.random.default_rng(2026)
    for task,names in text['classnames'].items():
        labels=list(names);tasks.append({'id':task,'classes':labels});classes+=labels;descriptions.update(names)
        for subset in ['train','val','test']:
            for cls,label in enumerate(labels):
                for index in range(train_per_class if subset=='train' else 2):
                    sid=f'{task}_{subset}_{label}_{index}';path=root/'features'/(sid+'.h5')
                    with h5py.File(path,'w') as f:
                        x=rng.normal(size=(patches,512)).astype('float32');x[:,:4]+=cls*.1
                        x/=np.linalg.norm(x,axis=1,keepdims=True)
                        f['features']=x;f['patch_ids']=np.array([f'{sid}_patch_{i}'.encode() for i in range(patches)])
                    slides.append({'slide_id':sid,'patient_id':sid,'label':label});splits.append({'patient_id':sid,'subset':subset});indices.append({'slide_id':sid,'path':str(path)});mapping[sid]=task
    for name,rows in [('slides',slides),('split',splits),('features',indices)]:pd.DataFrame(rows).to_csv(root/(name+'.csv'),index=False)
    cfg={'schema_version':1,'task_id':'synthetic_qpmil_four_tasks','task_type':'classification','synthetic':True,'classes':classes,
        'training_unit':'slide','evaluation_unit':'slide','defaults':{'fm':'synthetic_CONCH_aligned','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'synthetic_CONCH_aligned':{'dimension':512,'index':'features.csv'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    schema={'schema_version':1,'tasks':tasks,'slide_tasks':mapping,'provenance':'Explicitly synthetic aligned feature bags, not TCGA data or WSI extraction',
        'class_descriptions':descriptions,'templates':text['templates'],'semantic_descriptors':descriptors,
        'conch':{'path':str(Path(weights).resolve()),'sha256':sha256,'feature_encoding':{'model':'CONCH','proj_contrast':True,'normalize':True}}}
    manifest=prepare(Bundle(root),schema,root/'kmpmil_manifest.json')
    cfg['encoders']['synthetic_CONCH_aligned']['kmpmil']={'manifest':str(manifest)}
    (root/'task.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));return root
