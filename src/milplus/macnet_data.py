"""Actual raw RGB/ordered guide/provenance contract for complete source MacNet."""
from pathlib import Path
import json
import h5py,numpy as np,torch
from PIL import Image
from .data import resolve
from .io import digest_file,digest_object,read_json,write_json
from .macnet_source import SWIN22K_SHA256
CONDITIONS={'public':{'classes':2,'guides':5,'max_unlabelled':1995,'fixed_unlabelled':None,'batch_size':1},
    'lscc81':{'classes':3,'guides':81,'max_unlabelled':961,'fixed_unlabelled':961,'batch_size':2},
    'cscc64':{'classes':3,'guides':64,'max_unlabelled':961,'fixed_unlabelled':961,'batch_size':2}}


def strings(value):
    if value.ndim!=1:raise ValueError('Actual MacNet patch/patient/role vectors must be one dimensional')
    return [v.decode() if isinstance(v,bytes) else str(v) for v in value]


def identity(bundle):
    return digest_object({'task':bundle.config['task_id'],'type':bundle.task_type,'classes':bundle.classes,'fm':bundle.fm,'dimension':bundle.dim,
        'training_unit':bundle.config.get('training_unit','slide'),'evaluation_unit':bundle.config.get('evaluation_unit','slide'),'rows':bundle.rows.to_dict('records')})


def _task(bundle,condition):
    if condition not in CONDITIONS:raise ValueError('Declare actual public5, LSCC81 or CSCC64 original guide/bag conditions')
    if bundle.task_type!='classification' or len(bundle.classes)!=CONDITIONS[condition]['classes'] or bundle.dim!=768 or bundle.config.get('training_unit','slide')!='slide' or bundle.config.get('evaluation_unit','slide')!='slide':
        raise ValueError('MacNet requires original binary public/three-grade private slide classification and full768D raw Swin interface')


def _image_ref(root,value):
    if isinstance(value,str):return {'path':str(resolve(root,value).resolve())}
    if not isinstance(value,dict) or set(value)!={'path','dataset','index'} or not isinstance(value['dataset'],str) or not value['dataset'] or type(value['index']) is not int or value['index']<0:raise ValueError('Guide annotation needs an actual PNG or explicit HDF5 RGB source reference')
    return {**value,'path':str(resolve(root,value['path']).resolve())}


def _rgb(ref):
    if set(ref)=={'path'}:
        with Image.open(ref['path']) as f:array=np.asarray(f)
    else:
        with h5py.File(ref['path']) as f:
            d=f[ref['dataset']]
            if ref['index']>=len(d):raise ValueError('Actual guide image index is outside its archive')
            array=d[ref['index']]
    if array.ndim!=3 or array.shape[2]!=3 or min(array.shape[:2])<96 or array.dtype!=np.uint8:raise ValueError('Actual guide RGB must be uint8 and at least96pixels, with no invented channels')
    return array


def _archive(bundle,row,condition,annotations,scope,files):
    path=resolve(bundle.root,row['path']).resolve();rules=CONDITIONS[condition]
    with h5py.File(path) as f:
        required={'rgb','patch_ids','source_slide_ids','global_patient_ids','roles'}
        if not required<=set(f):raise ValueError('MacNet needs actual raw RGB, ordered patch identities/origins and guide roles; no frozen-feature substitute')
        rgb=f['rgb'];
        if rgb.is_virtual or rgb.external or Path(rgb.file.filename).resolve()!=path:raise ValueError('Materialize raw RGB or bind every external source; a standalone archive checksum cannot cover virtual/external pixels')
        n=len(rgb);k=rules['guides'];count=n-k
        if rgb.ndim!=4 or rgb.shape[3]!=3 or min(rgb.shape[1:3])<96 or rgb.dtype!=np.uint8:raise ValueError('Original raw RGB archive shape/dtype changed')
        if count<3 or count>rules['max_unlabelled'] or (rules['fixed_unlabelled'] is not None and count!=rules['fixed_unlabelled']):raise ValueError('Original MacNet unlabelled/guide bag counts changed; preselect real patches explicitly without zero padding')
        ids,sids,pids,roles=[strings(f[name][:]) for name in ['patch_ids','source_slide_ids','global_patient_ids','roles']]
        if any(len(v)!=n or any(not x for x in v) for v in [ids,sids,pids,roles]) or roles!=['unlabelled']*count+['prior_tumor']*k:raise ValueError('Actual MacNet patches must retain unlabelled-first/annotated-guide-last roles')
        if len(set(ids[:count]))!=count or len(set(ids[count:]))!=k:raise ValueError('Actual patches or guidance instances are repeated within the same role')
        if any(s!=row['slide_id'] for s in sids[:count]) or any(p!=row['global_patient_id'] for p in pids[:count]):raise ValueError('Unlabelled raw patches do not belong to the actual WSI patient')
        guides=[]
        for i in range(count,n):
            key=(ids[i],sids[i],pids[i]);annotation=annotations.get(key)
            if annotation is None:raise ValueError('A guide lacks an actual matching tumor-region annotation; never infer it from slide labels')
            if scope=='per_slide_annotated' and (sids[i]!=row['slide_id'] or pids[i]!=row['global_patient_id']):raise ValueError('Per-slide prior annotation belongs to another slide/patient')
            reference=_rgb(annotation['image']);files[annotation['image']['path']]=digest_file(annotation['image']['path'])
            if not np.array_equal(rgb[i],reference):raise ValueError('Supplied guide RGB differs from its actually annotated source image')
            guides.append({'patch_id':ids[i],'source_slide_id':sids[i],'global_patient_id':pids[i],'annotation':annotation})
        info={'path':str(path),'count':n,'unlabelled':count,'guides':guides,'shape':list(rgb.shape),
            'ordered_patch_identity_sha256':digest_object({'patch_ids':ids,'source_slide_ids':sids,'global_patient_ids':pids,'roles':roles}),
            'unlabelled_guide_overlap':sorted(set(ids[:count])&set(ids[count:]))}
    files[str(path)]=digest_file(path);info['sha256']=files[str(path)];return info


def _annotations(bundle,index,scope):
    annotations={};train_slides={r['slide_id']:r['global_patient_id'] for r in bundle.records('train')}
    for line in Path(index).read_text().splitlines():
        a=json.loads(line)
        if not isinstance(a,dict) or set(a)!={'patch_id','source_slide_id','global_patient_id','region_label','image','annotator','provenance'} or a['region_label']!='tumor' or any(not isinstance(a[k],str) or not a[k].strip() for k in ['patch_id','source_slide_id','global_patient_id','annotator','provenance']):raise ValueError('Each actual prior needs its original patch/patient identity, tumor-region annotation, annotator and provenance')
        if a['annotator']=='synthetic_validation_only' and not bundle.config.get('synthetic',False):raise ValueError('Synthetic guidance decisions cannot be claimed as clinical annotation')
        key=(a['patch_id'],a['source_slide_id'],a['global_patient_id'])
        if key in annotations:raise ValueError('Guide annotation identity is repeated')
        if scope=='train_bank' and train_slides.get(a['source_slide_id'])!=a['global_patient_id']:raise ValueError('Fixed guidance bank includes a held-out/non-TRAIN patient')
        annotations[key]={**a,'image':_image_ref(bundle.root,a['image'])}
    return annotations


def prepare(bundle,schema,output):
    keys={'schema_version','condition','guidance_scope','annotation_index','pretrained_weights','provenance'}
    if not isinstance(schema,dict) or set(schema)!=keys or schema['schema_version']!=1 or not isinstance(schema['provenance'],str) or not schema['provenance'].strip():raise ValueError('Declare complete actual MacNet raw data/guidance/initialization schema')
    if 'macnet' in bundle.encoder:raise ValueError('Prepare from unbound actual RGB archives')
    condition=schema['condition'];scope=schema['guidance_scope'];_task(bundle,condition)
    if scope not in {'train_bank','per_slide_annotated'}:raise ValueError('Declare a fixed TRAIN guide bank or additional per-query region annotations')
    weights=schema['pretrained_weights']
    if not isinstance(weights,dict) or set(weights)!={'path','sha256'} or weights['sha256']!=SWIN22K_SHA256:raise ValueError('Declare actual original-compatible ImageNet22K Swin-T checkpoint')
    weights=resolve(bundle.root,weights['path']).resolve()
    if digest_file(weights)!=SWIN22K_SHA256:raise ValueError('Actual Swin-T22K checkpoint bytes changed')
    files={str(weights):SWIN22K_SHA256};index=resolve(bundle.root,schema['annotation_index']).resolve();files[str(index)]=digest_file(index)
    annotations=_annotations(bundle,index,scope)
    rows={r['slide_id']:_archive(bundle,r,condition,annotations,scope,files) for r in bundle.rows.to_dict('records')}
    if scope=='train_bank':
        banks=[[{k:g[k] for k in ['patch_id','source_slide_id','global_patient_id']} for g in r['guides']] for r in rows.values()]
        if not banks or any(b!=banks[0] for b in banks):raise ValueError('The TRAIN guidance bank must be identical across all bags, independent of query labels')
    v={'schema_version':1,'kind':'macnet_original_actual_RGB_and_annotated_guides','identity':identity(bundle),'condition':condition,'guidance_scope':scope,
        'additional_annotations_at_query':scope=='per_slide_annotated','annotation_index':str(index),'pretrained_weights':str(weights),'provenance':schema['provenance'],
        'rows':rows,'files':files,'input_transform':'actual torchvision PIL Resize96, ToTensor, Normalize(mean=.5,std=.5); source dense Swin path',
        'patient_disjoint':True,'guide_region_labels_are_additional_supervision':True,'test_task_labels_not_used_for_fit_or_selection':True}
    v['contract']=digest_object(v);output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    write_json(output,v);return output


class MacNetInputs:
    def __init__(self,bundle):
        cfg=bundle.encoder.get('macnet')
        if not isinstance(cfg,dict) or set(cfg)!={'manifest'}:raise ValueError('Bind encoder.macnet.manifest to actual raw/guidance data')
        self.path=resolve(bundle.root,cfg['manifest']);v=read_json(self.path);self.contract=v.pop('contract',None)
        if self.contract!=digest_object(v) or v.get('kind')!='macnet_original_actual_RGB_and_annotated_guides' or v.get('identity')!=identity(bundle):raise ValueError('Actual MacNet patient/label/partition/FM/guidance identity changed')
        self.bundle=bundle;self.value=v;self.condition=v['condition'];self.rows=v['rows'];self.files=v['files'];self.weights=v['pretrained_weights'];_task(bundle,self.condition)
        if v['guidance_scope'] not in {'train_bank','per_slide_annotated'} or v['additional_annotations_at_query']!=(v['guidance_scope']=='per_slide_annotated'):raise ValueError('Actual MacNet extra supervision scope changed')
        if set(self.rows)!=set(bundle.rows.slide_id):raise ValueError('Actual raw MacNet WSI population changed')
        self.verify_bytes()
        # Reconstruct from the source annotation index and actual arrays; a
        # recomputed manifest checksum cannot forge patient/guide relations.
        annotations=_annotations(bundle,v['annotation_index'],v['guidance_scope'])
        files=dict(self.files);actual={r['slide_id']:_archive(bundle,r,self.condition,annotations,v['guidance_scope'],files) for r in bundle.rows.to_dict('records')}
        if actual!=self.rows or files!=self.files:raise ValueError('Actual original RGB, annotation, guide identity or ordered rows changed')
        if v['guidance_scope']=='train_bank':
            train={r['slide_id']:r['global_patient_id'] for r in bundle.records('train')};banks=[]
            for row in self.rows.values():
                bank=[(g['patch_id'],g['source_slide_id'],g['global_patient_id']) for g in row['guides']]
                if any(train.get(s)!=p for _,s,p in bank):raise ValueError('Fixed guidance bank includes a held-out patient')
                banks.append(bank)
            if any(b!=banks[0] for b in banks):raise ValueError('The guidance bank changed with query labels')

    def verify_bytes(self):
        if self.files.get(self.weights)!=SWIN22K_SHA256:raise ValueError('Actual original Swin initialization identity changed')
        for path,sha in self.files.items():
            if digest_file(path)!=sha:raise ValueError('Actual MacNet raw/guide/annotation/checkpoint bytes changed: '+path)

    def inventory(self,row,deep=False):
        r=self.rows[row['slide_id']]
        if deep and digest_file(r['path'])!=r['sha256']:raise ValueError('Actual original raw MacNet RGB bytes changed')
        return {**r,'guidance_scope':self.value['guidance_scope'],'additional_annotations_at_query':self.value['additional_annotations_at_query']}

    def tensor(self,row):
        from torchvision import transforms
        r=self.rows[row['slide_id']]
        if digest_file(r['path'])!=r['sha256']:raise ValueError('Actual original raw MacNet RGB bytes changed')
        transform=transforms.Compose([transforms.Resize([96,96]),transforms.ToTensor(),transforms.Normalize(mean=.5,std=.5)])
        with h5py.File(r['path']) as f:
            rgb=f['rgb'];result=torch.stack([transform(Image.fromarray(rgb[i])) for i in range(len(rgb))])
        return result
