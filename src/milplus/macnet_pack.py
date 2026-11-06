"""Lossless packing of actual ordered RGB tiles and explicit guide-role metadata."""
from pathlib import Path
import h5py,numpy as np,pandas as pd
from PIL import Image
from .io import digest_file,write_json
from .data import resolve


def pack(table,output):
    table=Path(table).resolve();rows=pd.read_csv(table,dtype=str,keep_default_na=False)
    fields=['slide_id','patch_id','source_slide_id','global_patient_id','role','image_path']
    if set(rows.columns)!=set(fields) or rows.empty or any(not str(v).strip() for v in rows[fields].to_numpy().ravel()):raise ValueError('Supply complete slide/patch/origin/patient/role/image table, without inferred labels or empty identities')
    if not set(rows.role)<={'unlabelled','prior_tumor'}:raise ValueError('Declare actual unlabelled/prior_tumor roles')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);files={};index=[];slugs=set()
    for number,(sid,group) in enumerate(rows.groupby('slide_id',sort=False)):
        # Generated archive names do not interpret slide IDs as file paths.
        path=output/f'bag_{number:08d}.h5';records=group.to_dict('records');shapes=[];sources=[]
        seen_guide=False
        for r in records:
            if r['role']=='prior_tumor':seen_guide=True
            elif seen_guide:raise ValueError('Keep actual unlabelled rows first and original annotated guides last; no implicit row sorting')
            p=resolve(table.parent,r['image_path']).resolve();sha=digest_file(p)
            with Image.open(p) as image:x=np.asarray(image)
            if x.ndim!=3 or x.shape[-1]!=3 or min(x.shape[:2])<96 or x.dtype!=np.uint8:raise ValueError('Materialize actual uint8 RGB tiles of at least96pixels; no implicit image/channel conversion')
            shapes.append(x.shape);sources.append((p,sha));files[str(p)]=sha
        if len(set(shapes))!=1:raise ValueError('One raw archive must have a common actual image size; resizing is the recorded source-model transform')
        with h5py.File(path,'w') as f:
            images=f.create_dataset('rgb',shape=(len(records),*shapes[0]),dtype='uint8',chunks=(1,*shapes[0]),compression='gzip')
            for i,(p,sha) in enumerate(sources):
                with Image.open(p) as im:images[i]=np.asarray(im)
                if digest_file(p)!=sha:raise ValueError('Actual source image changed during packing')
            for field,target in [('patch_id','patch_ids'),('source_slide_id','source_slide_ids'),('global_patient_id','global_patient_ids'),('role','roles')]:
                f[target]=np.array([r[field].encode() for r in records])
        index.append({'slide_id':sid,'path':str(path)})
    result=output/'raw_index.csv';pd.DataFrame(index).to_csv(result,index=False)
    write_json(output/'packing_provenance.json',{'kind':'actual_original_RGB_and_role_bytes_losslessly_packed','table':str(table),'table_sha256':digest_file(table),'source_files':files,
        'archives':{r['path']:digest_file(r['path']) for r in index},'resizing_sampling_zero_padding_or_label_inference':False})
    return result
