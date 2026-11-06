"""Explicitly synthetic native RGB task bundles for software validation only."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import yaml
from .io import write_json


def create_attention2majority_demo(root,weights,classes=2,patches=17):
    if classes not in {2,3} or type(patches) is not int or patches<1:
        raise ValueError('Native Attention2majority software fixture requires two/three classes and positive original patch counts')
    root=Path(root)
    root.mkdir();rng=np.random.default_rng(9011+classes);rows=[];splits=[];indices=[];pool=[]
    pixels=224 if classes==2 else 128
    for subset,copies in [('train',2),('val',1),('test',1)]:
        for target in range(classes):
            for index in range(copies):
                slide=f'{subset}_class{target}_patient{index}';p=root/(slide+'.h5')
                with h5py.File(p,'w') as f:
                    f.create_dataset('pixels',data=rng.integers(0,256,(patches,pixels,pixels,3),dtype=np.uint8))
                    f.create_dataset('patch_ids',data=np.asarray([slide+f'_patch{i}' for i in range(patches)],dtype=h5py.string_dtype()))
                rows.append({'slide_id':slide,'patient_id':slide,'global_patient_id':'synthetic:'+slide,'label':f'class{target}'})
                splits.append({'patient_id':slide,'subset':subset});indices.append({'slide_id':slide,'path':p.name})
                if subset=='train':pool.append(slide)
    pd.DataFrame(rows).to_csv(root/'slides.csv',index=False);pd.DataFrame(splits).to_csv(root/'split.csv',index=False)
    pd.DataFrame(indices).to_csv(root/'images.csv',index=False)
    schema={'schema_version':1,'kind':'attention2majority_uint8_rgb_hdf5','mode':'camelyon_binary' if classes==2 else 'kidney_three_grade',
        'image_provenance':'synthetic complete native RGB224/128 software validation','patch_provenance':'all original randomly generated pixels and unique patch IDs',
        'weights':str(Path(weights).resolve())}
    if classes==2:
        schema.update(discriminator_train_pool=pool,discriminator_pool_evidence='pool_review.json')
        write_json(root/'pool_review.json',{'kind':'reviewed_large_tumor_TRAIN_pool','selected_train_slide_ids':pool,
            'heldout_labels_used':False,'review_provenance':'Synthetic TRAIN-only pool for software validation, no clinical tumor annotation claim','synthetic':True})
    config={'schema_version':1,'task_id':f'attention2majority_synthetic_{classes}class','task_type':'classification','synthetic':True,
        'classes':[f'class{i}' for i in range(classes)],'defaults':{'fm':'rgb','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'rgb':{'dimension':3,'index':'images.csv','attention2majority_raw':schema}}}
    (root/'task.yaml').write_text(yaml.safe_dump(config));return root
