"""Original PIL RGB pixels and lexicographic paths; no resizing or orientation changes."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from .io import digest_file,digest_object,write_json


def pack(table,output):
    table=Path(table).resolve();out=Path(output).resolve()
    frame=pd.read_csv(table,dtype=str,keep_default_na=False)
    if not {'slide_id','path'}<=set(frame) or frame[['slide_id','path']].eq('').any().any():
        raise ValueError('ICMIL tile table requires nonempty slide_id and actual patch path')
    paths=[(Path(p) if Path(p).is_absolute() else table.parent/p).resolve() for p in frame.path]
    if len(set(paths))!=len(paths):raise ValueError('A physical ICMIL patch path cannot be assigned to multiple slides or repeated')
    frame['resolved_path']=[str(p) for p in paths];out.mkdir(parents=True,exist_ok=False);records=[];index=[]
    for sid,group in frame.groupby('slide_id',sort=True):
        original=sorted(group.resolved_path);path=out/(digest_object(sid)+'.h5');sources=[]
        with h5py.File(path,'x') as f:
            pixels=f.create_dataset('rgb',(len(original),256,256,3),dtype=np.uint8,chunks=(1,256,256,3),compression='gzip',compression_opts=1)
            for i,name in enumerate(original):
                p=Path(name);sha=digest_file(p)
                with Image.open(p) as image:
                    if image.mode!='RGB' or image.size!=(256,256):raise ValueError('Original coupled ICMIL expects RGB256 patches; no implicit color conversion, resize or EXIF transformation')
                    value=np.asarray(image)
                if value.dtype!=np.uint8:raise ValueError('Original coupled ICMIL requires original uint8 pixels')
                pixels[i]=value;sources.append({'path':str(p),'sha256':sha})
                if digest_file(p)!=sha:raise ValueError('Original ICMIL patch changed during packing')
            f.create_dataset('patch_ids',data=original,dtype=h5py.string_dtype('utf-8'))
        index.append({'slide_id':sid,'path':str(path)});records.append({'slide_id':sid,'path':str(path),'sha256':digest_file(path),'sources':sources})
    pd.DataFrame(index).to_csv(out/'pixels.csv',index=False)
    write_json(out/'pack_receipt.json',{'state':'completed','table':str(table),'table_sha256':digest_file(table),
        'decoder':'original PIL.Image.open RGB256, no EXIF transpose or resize','within_slide_order':'lexicographic resolved original patch paths',
        'slides':records,'labels_and_patient_partitions':'not inferred; provide audited task bundle'})
    return out/'pixels.csv'
