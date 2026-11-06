"""Explicit synthetic raw RGB and annotated TRAIN guide fixtures, no clinical claims."""
from pathlib import Path
import json
import numpy as np,pandas as pd,h5py,yaml
from .data import Bundle
from .macnet_data import CONDITIONS,prepare
from .macnet_source import SWIN22K_SHA256


def create_demo(output,weights,condition='public',public_unlabelled=32):
    if condition not in CONDITIONS or type(public_unlabelled) is not int:raise ValueError('Declare actual original MacNet fixture condition')
    cfg=CONDITIONS[condition];n=cfg['fixed_unlabelled'] or public_unlabelled;k=cfg['guides'];classes=cfg['classes']
    if n<k or n>cfg['max_unlabelled']:raise ValueError('Fixture needs actual distinct native guide origins in its training WSI')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);raw=out/'raw_RGB';raw.mkdir();rng=np.random.default_rng(0)
    labels=[f'class_{i}' for i in range(classes)];slides=[];splits=[];features=[];origin='train_class_'+str(classes-1)+'_0';origin_path=raw/(origin+'.h5')
    # Rich, distinct RGB textures with repeated spatial tiles reduce fixture
    # storage losslessly. They are still actual96x96 RGB inputs to the complete
    # encoder, not frozen features, an identity network or downsized backbone.
    def image():return np.tile(rng.integers(0,256,size=(32,32,3),dtype=np.uint8),(3,3,1))
    guides=np.stack([image() for _ in range(k)])
    for subset in ['train','val','test']:
        for c,label in enumerate(labels):
            for rep in range(2 if subset=='train' else 1):
                sid=f'{subset}_{label}_{rep}';path=raw/(sid+'.h5');ids=[f'{sid}_patch_{j}' for j in range(n)]+[f'{origin}_patch_{j}' for j in range(k)]
                with h5py.File(path,'w') as f:
                    rgb=f.create_dataset('rgb',shape=(n+k,96,96,3),dtype='uint8',chunks=(1,96,96,3),compression='gzip')
                    for j in range(n):rgb[j]=guides[j] if sid==origin and j<k else image()
                    rgb[n:]=guides
                    for name,values in [('patch_ids',ids),('source_slide_ids',[sid]*n+[origin]*k),('global_patient_ids',[sid]*n+[origin]*k),('roles',['unlabelled']*n+['prior_tumor']*k)]:f[name]=np.array([v.encode() for v in values])
                slides.append({'slide_id':sid,'patient_id':sid,'global_patient_id':sid,'label':label});splits.append({'patient_id':sid,'subset':subset});features.append({'slide_id':sid,'path':str(path)})
    for name,rows in [('slides',slides),('split',splits),('raw_index',features)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    task={'schema_version':1,'task_id':'synthetic_macnet_'+condition,'task_type':'classification','classes':labels,'synthetic':True,'training_unit':'slide','evaluation_unit':'slide',
        'defaults':{'fm':'raw_RGB_Swin22K_native768','split':'fold0'},'splits':{'fold0':'split.csv'},'encoders':{'raw_RGB_Swin22K_native768':{'dimension':768,'index':'raw_index.csv'}}}
    (out/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False));index=out/'actual_guide_annotations.jsonl'
    entries=[{'patch_id':f'{origin}_patch_{j}','source_slide_id':origin,'global_patient_id':origin,'region_label':'tumor',
        'image':{'path':str(origin_path),'dataset':'rgb','index':j},'annotator':'synthetic_validation_only',
        'provenance':'Explicit synthetic role decision for software verification; no pathologist or clinical annotation claimed.'} for j in range(k)]
    index.write_text('\n'.join(json.dumps(v) for v in entries)+'\n')
    schema={'schema_version':1,'condition':condition,'guidance_scope':'train_bank','annotation_index':str(index),
        'pretrained_weights':{'path':str(Path(weights).resolve()),'sha256':SWIN22K_SHA256},
        'provenance':'Synthetic original-sized RGB, patch identities, binary/three-grade labels and explicit TRAIN-only guide bank. Not real WSI or paper-score reproduction.'}
    manifest=prepare(Bundle(out),schema,out/'macnet_manifest.json');task['encoders']['raw_RGB_Swin22K_native768']['macnet']={'manifest':str(manifest)}
    (out/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False));return out
