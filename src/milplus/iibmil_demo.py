"""Real synthetic bag arrays with native500-row overflow/padding and tail minibatches."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml


def create_demo(output, classes=2, dimension=1280):
    if type(classes) is not int or classes < 2 or type(dimension) is not int or dimension < 1:
        raise ValueError('Declare positive feature width and at least two actual classes')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir()
    rng=np.random.default_rng(105)
    slides,features,splits=[],[],[]
    train_count=max(classes,17 if classes==2 else 5)
    for subset,count in [('train',train_count),('val',2*classes),('test',2*classes)]:
        for i in range(count):
            sid=f'synthetic_{subset}_{i:03d}';label=i%classes;n=[641,437,501][i%3]
            value=rng.normal(0,1,(n,dimension)).astype(np.float32);value[:,label%dimension]+=.4
            path='features/'+sid+'.h5'
            with h5py.File(root/path,'w') as f:
                f['features']=value
                f['patch_ids']=np.asarray([f'{sid}_actual_row_{j}'.encode() for j in range(n)])
            slides.append({'slide_id':sid,'patient_id':sid,'global_patient_id':sid,'label':f'class_{label}'})
            splits.append({'patient_id':sid,'subset':subset});features.append({'slide_id':sid,'path':path})
    pd.DataFrame(slides).to_csv(root/'slides.csv',index=False)
    pd.DataFrame(splits).to_csv(root/'split.csv',index=False)
    pd.DataFrame(features).to_csv(root/'features.csv',index=False)
    task={'schema_version':1,'task_id':'synthetic_iibmil_'+str(classes)+'classes',
        'synthetic':True,'task_type':'classification','classes':[f'class_{i}' for i in range(classes)],
        'training_unit':'slide','evaluation_unit':'patient',
        'label_provenance':'synthetic fixture; not clinical WSI or paper-score reproduction',
        'slides':'slides.csv','defaults':{'fm':'synthetic_fm','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'synthetic_fm':{'dimension':dimension,'index':'features.csv','weights':'synthetic_only',
                                 'preprocessing':'actual ordered Gaussian feature rows; original synthetic patch IDs'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False))
    return root
