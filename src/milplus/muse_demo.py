"""Full native-dimensional synthetic controls with actual600-description banks."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .io import write_json
from .muse_data import KIND, prepare
from .muse_source import bank_assets


def create_muse_demo(output, weights, patients=24, cohort='tcga_brca', dimension=512, patches=257):
    weights=Path(weights).resolve();bank_assets(cohort)
    if (not weights.is_file() or type(patients) is not int or patients<6 or patients%6
            or type(patches) is not int or patches<1 or dimension not in {512,768,1024,1536}):
        raise ValueError('MUSE synthetic demo needs actual weights, a positive source-supported dimension/patch count and6kpatients')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir()
    rng=np.random.default_rng(8500);slides=[];features=[];splits=[]
    for index in range(patients):
        label=index%2;patient=f'patient_{index:04d}';slide=patient+'_slide'
        subset=['train','val','test'][(index//2)%3];path='features/'+slide+'.h5'
        value=rng.normal(size=(patches,dimension)).astype(np.float32);value[:,label]+=1.
        with h5py.File(root/path,'w') as archive:archive['features']=value
        slides.append({'slide_id':slide,'patient_id':patient,'label':f'class_{label}'})
        features.append({'slide_id':slide,'path':path});splits.append({'patient_id':patient,'subset':subset})
    for name,rows in [('slides',slides),('features',features),('split',splits)]:pd.DataFrame(rows).to_csv(root/(name+'.csv'),index=False)
    task={'schema_version':1,'task_id':'synthetic_muse_'+cohort,'synthetic':True,'task_type':'classification',
        'classes':['class_0','class_1'],'training_unit':'slide','evaluation_unit':'slide',
        'label_provenance':'Synthetic controls in original binary class-prompt order; not clinical diagnoses',
        'slides':'slides.csv','defaults':{'fm':'toy','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'toy':{'dimension':dimension,'index':'features.csv','weights':'synthetic',
                          'preprocessing':'Synthetic Gaussian complete frozen bags; no real WSI extraction'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False))
    metadata={'schema_version':1,'kind':KIND,'synthetic':True,'checkpoint':str(weights),'cohort':cohort,
        'class_order':task['classes'],'feature_condition':'synthetic_native_dimension_fixture',
        'feature_preprocessing':'Full synthetic NbyD vectors, no learned or random stand-in text model',
        'feature_provenance':'Synthetic deterministic software controls; not real patch extraction',
        'label_provenance':'Explicit synthetic controls corresponding to source class-prompt positions',
        'train_subset_provenance':f'{patients//6}synthetic subjects per class in each disjoint TRAIN/VAL/TEST subset'}
    write_json(root/'muse_metadata.json',metadata);prepare(Bundle(root),'muse_metadata.json','muse/manifest.json')
    task['encoders']['toy']['muse']={'manifest':'muse/manifest.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False));return root
