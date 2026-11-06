"""Explicitly synthetic two-dimensional StableMIL fixtures; never infer real WSI grids."""
import h5py
import numpy as np
import pandas as pd
import yaml

from .demo import create_demo


def create_stable_demo(output,patients=30,dimension=16,classes=3,survival=False,seed=39):
    root=create_demo(output,patients=patients,dimension=dimension,classes=classes,survival=survival,seed=seed)
    slides=pd.read_csv(root/'slides.csv');features=pd.read_csv(root/'features.csv')
    if survival:
        slides=slides.drop_duplicates('patient_id')
        features=features[features.slide_id.isin(slides.slide_id)]
        slides.to_csv(root/'slides.csv',index=False);features.to_csv(root/'features.csv',index=False)
    rng=np.random.default_rng(seed+100)
    coords=np.asarray([(x*512,y*512) for x in range(10) for y in range(10)],dtype=np.int64)
    labels=dict(zip(slides.slide_id,slides.label))
    for row in features.to_dict('records'):
        values=rng.normal(size=(100,dimension)).astype(np.float32)
        values[:,int(labels[row['slide_id']].split('_')[-1])]+=1.25
        with h5py.File(root/row['path'],'w') as h:h['features']=values;h['coords']=coords
    path=root/'task.yaml';config=yaml.safe_load(path.read_text())
    config['encoders']['toy']['spatial']={'unit':'level0_pixels','origin':'top_left','patch_size':512,
        'provenance':'SYNTHETIC regular 10x10 registered grid for software tests, not inferred real WSI coordinates'}
    if survival:config['representative_slide_rule']='synthetic fixture: first generated slide per synthetic patient, frozen before training'
    path.write_text(yaml.safe_dump(config,sort_keys=False));return root
