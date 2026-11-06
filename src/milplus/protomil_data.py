"""Patient/CONCH/probe contracts for the full original concept ProtoMIL pipeline."""
from pathlib import Path
import copy,json
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import torch
from .data import Bundle,resolve,load_array
from .io import read_json,write_json,digest_file,digest_object


def identity(bundle):
    return digest_object({'task_id':bundle.config['task_id'],'type':bundle.task_type,'classes':bundle.classes,'fm':bundle.fm,
        'dimension':bundle.dim,'training_unit':bundle.config.get('training_unit','slide'),
        'evaluation_unit':bundle.config.get('evaluation_unit','slide'),'rows':bundle.rows.to_dict('records')})


def task_config_identity(value):
    c=copy.deepcopy(value)
    for e in c['encoders'].values():e.pop('protomil',None)
    return digest_object(c)


def original_ids(root,row,count):
    p=resolve(root,row['path'])
    if row.get('patch_ids_path'):values=load_array(resolve(root,row['patch_ids_path']),'patch_ids')
    elif p.suffix in {'.h5','.hdf5'}:
        with h5py.File(p) as f:
            if 'patch_ids' not in f:raise ValueError('ProtoMIL needs actual ordered patch identities')
            values=f['patch_ids'][:]
    else:raise ValueError('Provide original patch IDs in HDF5 or patch_ids_path')
    ids=[v.decode() if isinstance(v,bytes) else str(v) for v in values]
    if values.ndim!=1 or len(ids)!=count or len(set(ids))!=count or any(not v for v in ids):raise ValueError('Original patch identity/feature alignment changed')
    return ids


def actual_features(path,dimension=512):
    x=load_array(path)
    if x.ndim!=2 or len(x)<2 or x.shape[1]!=dimension or not np.issubdtype(x.dtype,np.floating) or not np.isfinite(x).all():
        raise ValueError('Original ProtoMIL requires at least two actual finite512D patch feature rows')
    x=np.asarray(x,dtype=np.float32)
    if np.any(np.linalg.norm(x,axis=-1)==0):raise ValueError('Actual CONCH features contain a zero-norm row')
    return torch.from_numpy(x)


def normalized_features(path):
    # Same operation order as author's prepare_sae_data.py, not a distinct FM.
    x=actual_features(path)
    return x/torch.norm(x,dim=-1,keepdim=True)


def image_ref(root,value):
    if isinstance(value,str):return {'path':str(resolve(root,value).resolve())}
    if not isinstance(value,dict) or set(value)!={'path','dataset','index'} or not isinstance(value['dataset'],str) or not value['dataset'] or type(value['index']) is not int or value['index']<0:
        raise ValueError('Probe image/mask must be a PNG path or explicit HDF5 path/dataset/index reference')
    return {**value,'path':str(resolve(root,value['path']).resolve())}


def read_image(ref):
    if set(ref)=={'path'}:
        with Image.open(ref['path']) as f:return np.asarray(f)
    with h5py.File(ref['path']) as f:
        if ref['dataset'] not in f or ref['index']>=len(f[ref['dataset']]):raise ValueError('Probe image/mask HDF5 identity changed')
        return f[ref['dataset']][ref['index']]


def validate_probe(probe,policy):
    image=read_image(probe['image']);mask=read_image(probe['mask']);pixels=policy['tile_pixels']
    if image.shape!=(pixels,pixels,3) or mask.shape!=(pixels,pixels) or image.dtype!=np.uint8 or not np.issubdtype(mask.dtype,np.integer):
        raise ValueError('Actual probe RGB/mask shape, dtype or native tile size changed')
    tumor=int(np.isin(mask,policy['tumor_values']).sum());normal=int(np.isin(mask,policy['normal_values']).sum())
    if probe['region_label']==1:
        if tumor<=policy['tumor_min_pixels']:raise ValueError('Supplied tumor probe does not satisfy its actual annotation threshold')
    elif normal<=policy['normal_min_pixels'] or tumor>policy['negative_max_tumor_pixels']:
        raise ValueError('Supplied normal probe does not satisfy its actual annotation thresholds')
    return {'normal_pixels':normal,'tumor_pixels':tumor}


def _policy(value):
    expected={'tile_pixels','normal_values','tumor_values','normal_min_pixels','tumor_min_pixels','negative_max_tumor_pixels','provenance'}
    if not isinstance(value,dict) or set(value)!=expected or value['tile_pixels']!=256 or any(type(value[k]) is not int or value[k]<0 for k in ['normal_min_pixels','tumor_min_pixels','negative_max_tumor_pixels']):
        raise ValueError('Declare original256pixel probe mask labels, thresholds and actual annotation provenance')
    for key in ['normal_values','tumor_values']:
        if not isinstance(value[key],list) or not value[key] or any(type(v) is not int for v in value[key]):raise ValueError('Probe annotation values must be explicit nonempty integer sets')
    if set(value['normal_values'])&set(value['tumor_values']) or not isinstance(value['provenance'],str) or not value['provenance'].strip():raise ValueError('Probe normal/tumor annotation definitions overlap or lack provenance')


def _encoding(value):
    if not isinstance(value,dict) or set(value)!={'model','checkpoint_sha256','proj_contrast','provenance'} or value['model']!='CONCH' or value['proj_contrast'] is not True or not isinstance(value['checkpoint_sha256'],str) or len(value['checkpoint_sha256'])!=64 or any(c not in '0123456789abcdef' for c in value['checkpoint_sha256']) or not isinstance(value['provenance'],str) or not value['provenance'].strip():
        raise ValueError('Bind actual pretrained512D projected CONCH feature/checkpoint provenance')


def _row(bundle,row,files):
    p=resolve(bundle.root,row['path']).resolve();x=actual_features(p);ids=original_ids(bundle.root,row,len(x));files[str(p)]=digest_file(p)
    result={**row,'path':str(p),'patches':len(x),'ordered_ids_sha256':digest_object(ids),'sha256':files[str(p)]}
    if row.get('patch_ids_path'):
        q=resolve(bundle.root,row['patch_ids_path']).resolve();result['patch_ids_path']=str(q);files[str(q)]=digest_file(q)
    return result


def prepare(bundle,schema,output):
    expected={'schema_version','provenance','encoding','sae_sources','probe_index','probe_policy','review_file'}
    if not isinstance(schema,dict) or set(schema)!=expected or schema['schema_version']!=1 or not isinstance(schema['provenance'],str) or not schema['provenance'].strip():
        raise ValueError('ProtoMIL schema requires version1, complete sources/probes/CONCH/review provenance')
    if 'protomil' in bundle.encoder:raise ValueError('Prepare from unbound actual frozen features')
    if bundle.task_type!='classification' or bundle.dim!=512 or len(bundle.classes)!=2 or bundle.config.get('training_unit','slide')!='slide' or bundle.config.get('evaluation_unit','slide')!='slide':
        raise ValueError('Original concept ProtoMIL uses slide-level binary classification with actual512D CONCH features and patient-disjoint splits')
    _encoding(schema['encoding']);_policy(schema['probe_policy']);files={};rows=[_row(bundle,r,files) for r in bundle.rows.to_dict('records')]
    sources=schema['sae_sources']
    if not isinstance(sources,list) or not sources:raise ValueError('Declare the actual joint SAE TRAIN populations')
    populations=[];metadata=[];seen=set();heldout={r['global_patient_id'] for r in rows if r['subset']!='train'}
    for item in sources:
        if not isinstance(item,dict) or set(item)!={'data','fm','split','cohort_id'} or not isinstance(item['cohort_id'],str) or not item['cohort_id']:
            raise ValueError('Each actual SAE source needs data/fm/split/cohort_id')
        source=Bundle(resolve(bundle.root,item['data']),item['fm'],item['split'])
        if source.dim!=512 or 'protomil' in source.encoder:raise ValueError('SAE sources must be original unbound512D CONCH populations')
        config=source.config;paths=[resolve(source.root,config.get('slides','slides.csv')),resolve(source.root,config['splits'][source.split]),resolve(source.root,source.encoder['index'])]
        hashes={str(p.resolve()):digest_file(p) for p in paths};files.update(hashes)
        metadata.append({'root':str(source.root),'fm':source.fm,'split':source.split,'task_config_sha256':task_config_identity(config),'files':hashes,'cohort_id':item['cohort_id']})
        for row in source.records('train'):
            if row['global_patient_id'] in heldout:raise ValueError('SAE pretraining population overlaps downstream held-out patients')
            r=_row(source,row,files)
            if r['path'] in seen:raise ValueError('Actual SAE population repeats a physical feature file')
            seen.add(r['path']);r['source_root']=str(source.root);r['cohort_id']=item['cohort_id'];populations.append(r)
    if not {str(resolve(bundle.root,r['path']).resolve()) for r in bundle.records('train')}<=seen:raise ValueError('SAE population omits actual downstream TRAIN slides')
    if len(populations)<2:raise ValueError('Original80/20 SAE piece split needs at least two actual TRAIN WSI files')
    main_train={r['slide_id']:r for r in rows if r['subset']=='train'};index=resolve(bundle.root,schema['probe_index']).resolve();files[str(index)]=digest_file(index)
    probes=[];probe_keys=set();ids_cache={}
    for line in index.read_text().splitlines():
        value=json.loads(line)
        if not isinstance(value,dict) or set(value)!={'slide_id','patch_index','region_label','image','mask'} or value['slide_id'] not in main_train or type(value['patch_index']) is not int or type(value['region_label']) is not int or value['region_label'] not in {0,1}:
            raise ValueError('Every annotated probe must be an actual downstream TRAIN patch, never VAL/TEST')
        row=main_train[value['slide_id']];idx=value['patch_index'];key=(row['slide_id'],idx)
        if idx<0 or idx>=row['patches'] or key in probe_keys:raise ValueError('Probe patch row missing, repeated or outside its actual WSI')
        probe_keys.add(key)
        if row['slide_id'] not in ids_cache:ids_cache[row['slide_id']]=original_ids(bundle.root,row,row['patches'])
        p={**value,'patch_id':ids_cache[row['slide_id']][idx],'global_patient_id':row['global_patient_id'],'image':image_ref(bundle.root,value['image']),'mask':image_ref(bundle.root,value['mask'])}
        for kind in ['image','mask']:
            path=p[kind]['path']
            if path not in files:files[path]=digest_file(path)
        p['actual_annotation']=validate_probe(p,schema['probe_policy']);probes.append(p)
    if {p['region_label'] for p in probes}!={0,1}:raise ValueError('Actual annotated probing set needs both normal and tumor regions')
    review=schema['review_file']
    if review is not None and (not isinstance(review,str) or not review.strip()):raise ValueError('Review must be a fixed JSON path or None; new SAE IDs cannot reuse arbitrary historical concepts')
    if review is not None:
        plain=review.replace('{sae_state_sha256}','').replace('{concept_bank_sha256}','')
        if '{' in plain or '}' in plain:raise ValueError('Review filenames accept only actual SAE and concept-bank hash placeholders')
    value={'schema_version':1,'kind':'protomil_complete_actual_inputs','identity':identity(bundle),'encoding':schema['encoding'],'provenance':schema['provenance'],
        'probe_index':str(index),'rows':rows,'sae_population':populations,'source_metadata':metadata,'probes':probes,'probe_policy':schema['probe_policy'],'files':files,
        'review_file':None if review is None else str(resolve(bundle.root,review).resolve()),'test_used_for_fit_or_selection':False}
    value['contract']=digest_object(value);out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    write_json(out,value);return out


class ProtoMILInputs:
    def __init__(self,bundle):
        config=bundle.encoder.get('protomil')
        if not isinstance(config,dict) or set(config)!={'manifest'}:raise ValueError('Bind encoder.protomil.manifest to actual prepared patient/CONCH/probe data')
        self.bundle=bundle;self.path=resolve(bundle.root,config['manifest']);value=read_json(self.path);self.contract=value.pop('contract',None)
        if self.contract!=digest_object(value) or value.get('kind')!='protomil_complete_actual_inputs' or value.get('schema_version')!=1 or value.get('identity')!=identity(bundle):raise ValueError('Actual ProtoMIL manifest, patients, labels, partitions or FM changed')
        _encoding(value['encoding']);_policy(value['probe_policy']);self.value=value;self.files=value['files'];self.rows={r['slide_id']:r for r in value['rows']};self.population=value['sae_population'];self.probes=value['probes'];self.review_file=value['review_file']
        if len(self.rows)!=len(value['rows']) or set(self.rows)!=set(bundle.rows.slide_id):raise ValueError('Actual ProtoMIL slide population changed')
        self.verify_bytes()
        self.validate_semantics()

    def validate_semantics(self):
        validate_semantics(self)

    def verify_bytes(self):
        for name,sha in self.files.items():
            if digest_file(name)!=sha:raise ValueError('Actual ProtoMIL input feature/annotation/metadata bytes changed: '+name)
        import yaml
        for meta in self.value['source_metadata']:
            c=yaml.safe_load((Path(meta['root'])/'task.yaml').read_text())
            if task_config_identity(c)!=meta['task_config_sha256']:raise ValueError('Actual SAE source configuration/encoder/labels changed')

    def features(self,row):
        r=self.rows[row['slide_id']]
        if digest_file(r['path'])!=r['sha256']:raise ValueError('Actual ProtoMIL frozen feature bytes changed')
        x=normalized_features(r['path'])
        if len(x)!=r['patches'] or digest_object(original_ids(self.bundle.root,r,len(x)))!=r['ordered_ids_sha256']:raise ValueError('Actual ProtoMIL patch identities/order changed')
        return x

    def pretraining_features(self,row):
        if row['global_patient_id'] in {r['global_patient_id'] for r in self.rows.values() if r['subset']!='train'}:raise ValueError('SAE decoded downstream held-out patient')
        if digest_file(row['path'])!=row['sha256']:raise ValueError('Actual SAE source feature bytes changed')
        x=normalized_features(row['path'])
        if len(x)!=row['patches'] or digest_object(original_ids(Path(row['source_root']),row,len(x)))!=row['ordered_ids_sha256']:raise ValueError('Actual SAE source patch order changed')
        return x



def validate_semantics(assets):
    bundle=assets.bundle;value=assets.value
    if bundle.task_type!='classification' or bundle.dim!=512 or len(bundle.classes)!=2 or bundle.config.get('training_unit','slide')!='slide' or bundle.config.get('evaluation_unit','slide')!='slide':raise ValueError('Actual ProtoMIL binary slide/CONCH task changed')
    for row in bundle.rows.to_dict('records'):
        declared=assets.rows[row['slide_id']]
        if any(declared.get(k)!=row[k] for k in ['slide_id','patient_id','global_patient_id','label','target','subset']) or declared['path']!=str(resolve(bundle.root,row['path']).resolve()):raise ValueError('Actual ProtoMIL joined patient/label/partition/feature record changed')
        if assets.files.get(declared['path'])!=declared['sha256']:raise ValueError('Actual ProtoMIL feature hash binding changed')
    populations={};physical=set();heldout={r['global_patient_id'] for r in assets.rows.values() if r['subset']!='train'}
    for meta in value['source_metadata']:
        root=Path(meta['root']);cfg=__import__('yaml').safe_load((root/'task.yaml').read_text())
        if cfg['encoders'][meta['fm']]['dimension']!=512:raise ValueError('Actual SAE source encoder dimension changed')
        paths=[resolve(root,cfg.get('slides','slides.csv')),resolve(root,cfg['splits'][meta['split']]),resolve(root,cfg['encoders'][meta['fm']]['index'])]
        if {str(p.resolve()) for p in paths}!=set(meta['files']) or any(assets.files.get(name)!=sha for name,sha in meta['files'].items()):raise ValueError('Actual SAE source metadata files changed')
        slides,splits,features=[pd.read_csv(p,dtype=str,keep_default_na=False) for p in paths]
        joined=slides.merge(splits,on='patient_id',validate='many_to_one').merge(features,on='slide_id',validate='one_to_one')
        for row in joined.to_dict('records'):
            if row['subset']=='train':populations[(str(root),row['slide_id'])]={**row,'global_patient_id':row.get('global_patient_id',row['patient_id'])}
    if len(assets.population)!=len(populations):raise ValueError('Actual SAE declared TRAIN population is incomplete or repeated')
    for row in assets.population:
        actual=populations.get((row['source_root'],row['slide_id']))
        if actual is None or any(row.get(k)!=actual[k] for k in ['patient_id','global_patient_id','subset']) or row['path']!=str(resolve(Path(row['source_root']),actual['path']).resolve()):raise ValueError('Actual SAE source patient/partition/feature record changed')
        if row['global_patient_id'] in heldout:raise ValueError('SAE population overlaps downstream held-out patients')
        if row['path'] in physical or assets.files.get(row['path'])!=row['sha256']:raise ValueError('Actual SAE population repeats files or changes hash binding')
        physical.add(row['path'])
    if not {r['path'] for r in assets.rows.values() if r['subset']=='train'}<=physical:raise ValueError('SAE population omits downstream TRAIN slides')
    index=Path(value['probe_index'])
    if assets.files.get(str(index))!=digest_file(index):raise ValueError('Actual probe source-index hash changed')
    original=[json.loads(line) for line in index.read_text().splitlines()]
    if len(original)!=len(assets.probes):raise ValueError('Actual annotated probe population changed')
    seen=set();ids={}
    for declared,row in zip(assets.probes,original):
        sid=row['slide_id'];item=assets.rows.get(sid)
        if item is None or item['subset']!='train' or declared['global_patient_id']!=item['global_patient_id']:raise ValueError('Actual concept probe belongs to a held-out patient')
        for key in ['slide_id','patch_index','region_label']:
            if declared.get(key)!=row[key]:raise ValueError('Actual original annotated probe record changed')
        i=declared['patch_index'];key=(sid,i)
        if type(i) is not int or i<0 or i>=item['patches'] or key in seen or declared['region_label'] not in {0,1}:raise ValueError('Actual annotated probe patch/label is invalid or repeated')
        seen.add(key)
        for kind in ['image','mask']:
            if declared[kind]!=image_ref(bundle.root,row[kind]) or declared[kind]['path'] not in assets.files:raise ValueError('Actual probe RGB/mask source reference changed')
        if sid not in ids:ids[sid]=original_ids(bundle.root,item,item['patches'])
        if declared['patch_id']!=ids[sid][i]:raise ValueError('Actual annotated probe ordered patch identity changed')
        if declared['actual_annotation']!=validate_probe(declared,value['probe_policy']):raise ValueError('Actual annotated probe mask counts or region semantics changed')
