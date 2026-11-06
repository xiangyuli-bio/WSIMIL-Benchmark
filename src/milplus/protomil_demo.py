"""Explicit synthetic numeric/RGB-mask fixture, not real CONCH or WSI data."""
from pathlib import Path
import json
import h5py,numpy as np,pandas as pd,yaml
from .data import Bundle
from .protomil_data import prepare


def _task(root,prefix,patches,train_per_class,rng):
    root.mkdir();(root/'features').mkdir();slides=[];split=[];features=[]
    for subset in ['train','val','test']:
        for target,label in enumerate(['normal','tumor']):
            for i in range(train_per_class if subset=='train' else 1):
                sid=f'{prefix}_{subset}_{label}_{i}';path=root/'features'/(sid+'.h5')
                x=rng.normal(size=(patches,512)).astype('float32');x[:,:8]+=target*1.5
                with h5py.File(path,'w') as f:
                    f['features']=x;f['patch_ids']=np.array([f'{sid}_patch_{j}'.encode() for j in range(patches)])
                slides.append({'slide_id':sid,'patient_id':sid,'global_patient_id':sid,'label':label})
                split.append({'patient_id':sid,'subset':subset});features.append({'slide_id':sid,'path':str(path)})
    for name,rows in [('slides',slides),('split',split),('features',features)]:pd.DataFrame(rows).to_csv(root/(name+'.csv'),index=False)
    cfg={'schema_version':1,'task_id':'synthetic_protomil_'+prefix,'task_type':'classification','classes':['normal','tumor'],'synthetic':True,
        'training_unit':'slide','evaluation_unit':'slide','defaults':{'fm':'synthetic512_CONCH_shape','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'synthetic512_CONCH_shape':{'dimension':512,'index':'features.csv'}}}
    (root/'task.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));return cfg


def create_demo(output,patches=17,probe_per_class=5):
    if type(patches) is not int or patches<2 or type(probe_per_class) is not int or probe_per_class<5 or 3*patches<probe_per_class:
        raise ValueError('Actual synthetic bags need enough distinct annotated patches for both explicit probe groups')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(2026)
    data=root/'task';extra=root/'additional_SAE_cohort';cfg=_task(data,'main',patches,3,rng);_task(extra,'additional',patches,1,rng)
    archive=data/'actual_probe_RGB_masks.h5';index=data/'probes.jsonl';records=[];bundle=Bundle(data)
    with h5py.File(archive,'w') as f:
        images=f.create_dataset('rgb',shape=(2*probe_per_class,256,256,3),dtype='uint8',chunks=(1,256,256,3),compression='gzip')
        masks=f.create_dataset('mask',shape=(2*probe_per_class,256,256),dtype='uint8',chunks=(1,256,256),compression='gzip')
        counter=0
        for label in [1,0]:
            selected=[r for r in bundle.records('train') if r['target']==label];left=probe_per_class
            for row in selected:
                for j in range(min(left,patches)):
                    image=np.empty((256,256,3),dtype='uint8');image[:]=[220,180,200] if label==0 else [160,90,145];image[0,0]=[counter%251,j%251,128]
                    images[counter]=image;masks[counter]=np.full((256,256),label+1,dtype='uint8')
                    records.append({'slide_id':row['slide_id'],'patch_index':j,'region_label':label,
                        'image':{'path':str(archive),'dataset':'rgb','index':counter},'mask':{'path':str(archive),'dataset':'mask','index':counter}})
                    left-=1;counter+=1
                if left==0:break
            assert left==0
    index.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    schema={'schema_version':1,'provenance':'Explicit synthetic numeric features/RGB/masks for source-engineering verification only; no clinical data or original FM extraction.',
        'encoding':{'model':'CONCH','checkpoint_sha256':'40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c','proj_contrast':True,
            'provenance':'Synthetic512D cached numeric fixture of the original interface; these RGBs were not encoded by CONCH.'},
        'sae_sources':[{'data':str(data),'fm':bundle.fm,'split':'fold0','cohort_id':'synthetic_main'},
            {'data':str(extra),'fm':bundle.fm,'split':'fold0','cohort_id':'synthetic_additional'}],
        'probe_index':str(index),'probe_policy':{'tile_pixels':256,'normal_values':[1],'tumor_values':[2],'normal_min_pixels':5000,'tumor_min_pixels':1000,
            'negative_max_tumor_pixels':0,'provenance':'Explicit synthetic full normal/tumor region masks, not inferred from weak slide labels.'},
        'review_file':str(data/'concept_reviews'/'{sae_state_sha256}.{concept_bank_sha256}.json')}
    manifest=prepare(bundle,schema,data/'protomil_manifest.json');cfg['encoders'][bundle.fm]['protomil']={'manifest':str(manifest)}
    (data/'task.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));return data
