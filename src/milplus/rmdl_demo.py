"""Explicitly synthetic spatial slides for full-shape RMDL software checks."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from .io import digest_file
from .rmdl_pixels import WORKING_MPP
from .rmdl_source import IMAGENET_SHA256


def create_demo(output, weights, side=1899):
    if type(side) is not int or side < 1899:
        raise ValueError('Full RMDL demo retains actual1899 blocks and enough spatial instances; no tiny-grid surrogate')
    weights = Path(weights).resolve()
    if digest_file(weights) != IMAGENET_SHA256:
        raise ValueError('Provide the actual complete official Keras ImageNet backbone initialization')
    root = Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    images = root/'raw';images.mkdir()
    y,x = np.indices((side,side),dtype=np.int32)
    slides = [];splits = [];index = [];records = []
    for subset in ['train','val','test']:
        for target,name in enumerate(['normal','dysplasia','cancer']):
            sid = f'synthetic_only_{subset}_{name}';patient = 'patient_'+sid
            offset = 19*len(slides)+3
            # Reproducible nonconstant RGB; the demo has no clinical meaning.
            pixels = np.stack([(x//3+y//5+offset)%256,(x//7+2*y+offset)%256,
                               (x+y//11+3*offset)%256],axis=-1).astype(np.uint8)
            path = images/(sid+'.h5')
            with h5py.File(path,'w') as file:
                file.create_dataset('rgb',data=pixels,chunks=(256,256,3),compression='gzip',compression_opts=1)
                file.attrs['working_mpp'] = WORKING_MPP;file.attrs['slide_id'] = sid
                if subset == 'train':
                    # Explicit synthetic per-pixel annotations, identical class
                    # definitions across all TRAIN slides, never slide-label fill.
                    mask = np.minimum(x//(side//3),2).astype(np.uint8)
                    file.create_dataset('annotation',data=mask,chunks=(256,256),compression='gzip',compression_opts=1)
            slides.append({'slide_id':sid,'patient_id':patient,'label':name})
            splits.append({'patient_id':patient,'subset':subset})
            index.append({'slide_id':sid,'path':str(path)})
            records.append({'slide_id':sid,'kind':'working_rgb_hdf5','working_mpp':WORKING_MPP,
                            'annotation':str(path) if subset == 'train' else None,'provenance':'procedural synthetic software validation only'})
    for name,rows in [('slides.csv',slides),('split.csv',splits),('index.csv',index)]:
        with (root/name).open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'raw_records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    config = {'schema_version':1,'task_id':'synthetic_rmdl_full_shape_validation','task_type':'classification',
        'synthetic':True,'cohort_id':'synthetic_validation_only','classes':['normal','dysplasia','cancer'],
        'training_unit':'slide','evaluation_unit':'slide','slides':'slides.csv','splits':{'fixed':'split.csv'},
        'defaults':{'fm':'rmdl_native_FConv','split':'fixed'},
        'encoders':{'rmdl_native_FConv':{'dimension':1024,'index':'index.csv','rmdl_raw':{
            'schema_version':1,'kind':'rmdl_original_scale_WSI_and_TRAIN_pixel_masks','records':'raw_records.jsonl',
            'imagenet_weights':str(weights),'image_provenance':'procedural spatial uint8 RGB, no clinical or paper-score claim',
            'annotation_provenance':'explicit synthetic TRAIN masks; validation/test masks are absent'}}}}
    (root/'task.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    return root
