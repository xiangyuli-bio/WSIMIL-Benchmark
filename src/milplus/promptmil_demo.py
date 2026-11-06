"""Explicitly synthetic full-resolution RGB, with distinct actual pixel arrays and patch IDs."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from .data import Bundle
from .promptmil_data import KIND,prepare


def create_demo(output,weights,condition='brca',patches=1025,classes=None):
    if condition not in {'brca','bright','crc'}or type(patches)is not int or patches<1:
        raise ValueError('Choose a native Prompt-MIL cohort and positive actual RGB patch count')
    classes=(3 if condition=='bright'else 2)if classes is None else classes
    if type(classes)is not int or classes<2:raise ValueError('Declare at least two real synthetic classes')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);(root/'rgb').mkdir()
    slides,features,splits=[],[],[];y,x=np.indices((224,224),dtype=np.int32)
    for phase,(subset,count)in enumerate([('train',max(9,classes)),('val',2*classes),('test',2*classes)]):
        for i in range(count):
            sid=f'synthetic_{subset}_{i:03d}';label=i%classes;path='rgb/'+sid+'.h5'
            with h5py.File(root/path,'x')as f:
                rgb=f.create_dataset('rgb',shape=(patches,224,224,3),dtype=np.uint8,chunks=(1,224,224,3),compression='gzip',compression_opts=1)
                # All full224RGB pixels are materialized. Deterministic varied
                # spatial/color patterns keep these nonclinical test assets
                # losslessly compressible; no encoder computation is skipped.
                base=(x*(3+2*phase)+y*(5+2*['brca','bright','crc'].index(condition))
                      +(x//7)*(y//11)+i*13+label*23)%256
                for j in range(patches):
                    rgb[j]=np.stack([(base+j*17+c*37+(x%(j%17+1))*c)%256 for c in range(3)],axis=-1).astype(np.uint8)
                f['patch_ids']=np.asarray([f'{sid}_original_synthetic_pixel_patch_{j}'for j in range(patches)],dtype=h5py.string_dtype('utf-8'))
            slides.append({'slide_id':sid,'patient_id':sid,'global_patient_id':sid,'label':f'class_{label}'})
            features.append({'slide_id':sid,'path':path});splits.append({'patient_id':sid,'subset':subset})
    pd.DataFrame(slides).to_csv(root/'slides.csv',index=False);pd.DataFrame(features).to_csv(root/'features.csv',index=False)
    pd.DataFrame(splits).to_csv(root/'split.csv',index=False)
    task={'schema_version':1,'task_id':'synthetic_promptmil_'+condition,'synthetic':True,'task_type':'classification',
          'classes':[f'class_{i}'for i in range(classes)],'training_unit':'slide','evaluation_unit':'patient',
          'slides':'slides.csv','defaults':{'fm':'native_raw_RGB','split':'fold0'},'splits':{'fold0':'split.csv'},
          'label_provenance':'synthetic engineering fixture; no clinical WSI or paper-result claim',
          'encoders':{'native_raw_RGB':{'dimension':3,'index':'features.csv','preprocessing':'original224RGB uint8; no resize or feature-vector substitute'}}}
    p=root/'task.yaml';p.write_text(yaml.safe_dump(task,sort_keys=False))
    schema={'schema_version':1,'kind':KIND,'condition':condition,'pixel_key':'rgb','patch_ids_key':'patch_ids',
            'normalization':'source_uint8_div255_only','initial_weights':str(Path(weights).resolve()),
            'pixel_provenance':'fully materialized distinct deterministic full224RGB synthetic spatial/color patterns',
            'patch_provenance':'distinct original synthetic pixel patch IDs in actual file row order'}
    prepare(Bundle(root),schema,root/'promptmil.json')
    task['encoders']['native_raw_RGB']['promptmil']={'manifest':'promptmil.json'}
    p.write_text(yaml.safe_dump(task,sort_keys=False));return root
