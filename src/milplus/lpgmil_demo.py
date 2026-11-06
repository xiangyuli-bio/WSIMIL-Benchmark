"""Explicit synthetic actual bags and nucleus counts; no clinical evidence."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .io import write_json


def create_demo(output,patches=1025,dimension=1024,classes=2):
    if any(type(v)is not int or v<1 for v in [patches,dimension,classes])or patches<8 or classes<2:
        raise ValueError('LPGMIL demo needs at least8actual patches, positive FM width and at least2classes')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir()
    rng=np.random.default_rng(107);slides=[];features=[];splits=[];counts=[]
    for subset in ['train','val','test']:
        for i in range(2*classes):
            sid=f'synthetic_{subset}_{i:03d}';label=i%classes
            value=rng.normal(0,.4,(patches,dimension)).astype(np.float32)
            value[:,label%dimension]+=.3
            names=[f'{sid}_actual_patch_{j}'for j in range(patches)]
            path='features/'+sid+'.h5'
            with h5py.File(root/path,'w')as f:
                f['features']=value;f['patch_ids']=np.asarray([v.encode()for v in names])
            slides.append({'slide_id':sid,'patient_id':sid,'global_patient_id':sid,'label':f'class_{label}'})
            splits.append({'patient_id':sid,'subset':subset});features.append({'slide_id':sid,'path':path})
            if subset=='train':
                counts.extend({'slide_id':sid,'patch_id':name,'n_lymphocytes':18 if j%4==0 else 2,
                               'n_nuclei':20}for j,name in enumerate(names))
    for name,values in [('slides',slides),('features',features),('split',splits),('train_counts',counts)]:
        pd.DataFrame(values).to_csv(root/(name+'.csv'),index=False)
    write_json(root/'detector.json',{'synthetic':True,'detector':'synthetic_fixture_actual_count_rows',
                                  'not_clinical_validation':True})
    task={'schema_version':1,'task_id':'synthetic_lpgmil_reference','task_type':'classification',
        'synthetic':True,'training_unit':'slide','evaluation_unit':'patient',
        'classes':[f'class_{i}'for i in range(classes)],'slides':'slides.csv',
        'defaults':{'fm':'synthetic_fm','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'synthetic_fm':{'dimension':dimension,'index':'features.csv','weights':'synthetic_only',
            'preprocessing':'actual ordered finite feature rows and original patch identities'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False))
    from .data import Bundle
    from .lpgmil_data import prepare
    manifest=prepare(Bundle(root),root/'train_counts.csv',root/'detector.json',root/'lpgmil_assets',.75,10)
    task['encoders']['synthetic_fm']['lpgmil']={'manifest':str(manifest.relative_to(root))}
    (root/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False))
    return root
