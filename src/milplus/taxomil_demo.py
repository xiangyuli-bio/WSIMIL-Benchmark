"""Clearly synthetic full-width TaxoMIL bags with the real public taxonomy/GPT2."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .upstream import pinned_text
from .taxomil_source import checked_source


def create_taxomil_demo(output,assets,dataset='BRACS',patients=42,dimension=1024,patches=257,train_per_class=None):
    root,sha=checked_source()
    labels=json.loads(pinned_text(root,sha,'config/dataset_labels.json'))
    if dataset not in labels or dimension!=1024 or type(patches) is not int or patches<1:
        raise ValueError('TaxoMIL original domain/native1024D complete bags required')
    taxonomy=labels[dataset];classes=taxonomy['fine'];count=len(classes)
    if (type(patients) is not int or patients<6*count or
            train_per_class is None and patients%(3*count) or
            train_per_class is not None and (type(train_per_class) is not int or train_per_class<2 or patients!=(train_per_class+4)*count)):
        raise ValueError('TaxoMIL controls need at least two TRAIN/VAL/TEST patients per original fine class')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'features').mkdir()
    rng=np.random.default_rng(8707);slides=[];features=[];splits=[]
    for index in range(patients):
        patient=f'patient_{index:04d}';slide=patient+'_slide';target=index%count
        subset=(['train','val','test'][(index//count)%3] if train_per_class is None else
            'train' if index//count<train_per_class else 'val' if index//count<train_per_class+2 else 'test');path='features/'+slide+'.pt'
        # Vary actual lengths: source padding is part of its released recipe.
        values=rng.normal(size=(patches-index%3,dimension)).astype(np.float32);values[:,target]+=2.
        torch.save(torch.from_numpy(values),out/path)
        slides.append({'slide_id':slide,'patient_id':patient,'label':classes[target],
            'coarse_label':taxonomy['hierarchy'][classes[target]]})
        features.append({'slide_id':slide,'path':path});splits.append({'patient_id':patient,'subset':subset})
    for name,rows in [('slides',slides),('features',features),('split',splits)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    task={'schema_version':1,'task_id':'synthetic_taxomil_'+dataset.lower(),'synthetic':True,
        'task_type':'classification','classes':classes,'training_unit':'slide','evaluation_unit':'slide',
        'hierarchy':{'coarse_classes':taxonomy['coarse'],'fine_to_coarse':taxonomy['hierarchy'],
            'label_provenance':'Original public TaxoMIL '+dataset+' diagnostic taxonomy; synthetic patients/no clinical results'},
        'label_provenance':'Synthetic full-width Gaussian bags; no real WSI or clinical experiment',
        'slides':'slides.csv','defaults':{'fm':'toy','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'toy':{'dimension':dimension,'index':'features.csv','weights':'synthetic',
            'preprocessing':'Synthetic full-width complete variable-length bags; no real pathology FM',
            'taxomil':{'dataset':dataset,'assets':str(Path(assets).resolve())}}}}
    (out/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False));return out
