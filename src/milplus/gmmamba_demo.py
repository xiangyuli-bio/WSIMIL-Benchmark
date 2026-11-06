"""Explicit synthetic complete coordinate bags for full512D GMMamba."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml


def create_gmmamba_demo(output,patients=12,dimension=512,classes=2,patches=257):
    if (type(patients) is not int or patients<3*classes or patients%(3*classes)
            or type(classes) is not int or classes not in {2,3}
            or type(dimension) is not int or dimension<1 or type(patches) is not int or patches<40):
        raise ValueError('GMMamba synthetic fixture needs3k patients/class,positive FM width and >=40 aligned coordinate patches')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir()
    rng=np.random.default_rng(8606);slides=[];features=[];splits=[]
    for index in range(patients):
        patient=f'patient_{index:04d}';slide=patient+'_slide';target=index%classes
        subset=['train','val','test'][(index//classes)%3];path='features/'+slide+'.h5'
        values=rng.normal(size=(patches,dimension)).astype(np.float32);values[:,target]+=2.
        # Ten separated measured-position controls, preserving all original
        # feature row order. These are explicitly synthetic, never real WSI.
        coords=np.array([[int(i%10)*100000+int(i//10)*256,int(i//10%7)*256] for i in range(patches)],dtype=np.int64)
        with h5py.File(root/path,'w') as archive:archive['features']=values;archive['coords']=coords
        slides.append({'slide_id':slide,'patient_id':patient,'label':f'class_{target}'})
        features.append({'slide_id':slide,'path':path});splits.append({'patient_id':patient,'subset':subset})
    for name,rows in [('slides',slides),('features',features),('split',splits)]:pd.DataFrame(rows).to_csv(root/(name+'.csv'),index=False)
    task={'schema_version':1,'task_id':'synthetic_gmmamba','synthetic':True,'task_type':'classification',
        'classes':[f'class_{i}' for i in range(classes)],'training_unit':'slide','evaluation_unit':'slide',
        'label_provenance':'Synthetic complete coordinate controls; no clinical labels','slides':'slides.csv',
        'defaults':{'fm':'toy','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'toy':{'dimension':dimension,'index':'features.csv','weights':'synthetic',
            'preprocessing':'Synthetic complete Gaussian frozen bags with deterministic class signal',
            'spatial':{'unit':'level0_pixels','origin':'top_left','patch_size':256,
                'provenance':'Synthetic independent integer-position controls in10 separated clusters; not real WSI'}}}}
    (root/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False));return root
