"""Patient-disjoint ordered binary tasks and actual, ordered frozen FM bags."""
from pathlib import Path

import h5py
import numpy as np

from .data import resolve,load_array
from .io import digest_file,digest_object,read_json,write_json


def identity(bundle):
    return digest_object({'task_id':bundle.config['task_id'],'task_type':bundle.task_type,
        'fm':bundle.fm,'dimension':bundle.dim,'classes':bundle.classes,
        'training_unit':bundle.config.get('training_unit','slide'),
        'evaluation_unit':bundle.config.get('evaluation_unit','slide'),
        'rows':bundle.rows.to_dict('records')})


def patch_ids(bundle,row,count):
    path=resolve(bundle.root,row['path'])
    if row.get('patch_ids_path'):
        values=load_array(resolve(bundle.root,row['patch_ids_path']),'patch_ids')
    elif path.suffix in {'.h5','.hdf5'}:
        with h5py.File(path,'r') as f:
            if 'patch_ids' not in f:raise ValueError('CoMEL needs real patch_ids preserving native feature order')
            values=f['patch_ids'][:]
    else:raise ValueError('CoMEL requires ordered patch_ids or an explicit patch_ids_path')
    ids=[v.decode() if isinstance(v,bytes) else str(v) for v in values]
    if values.ndim!=1 or len(ids)!=count or len(set(ids))!=count or any(not v for v in ids):
        raise ValueError('CoMEL patch identities must be nonempty, unique and match actual feature rows')
    return digest_object(ids)


def prepare(bundle,schema,output):
    if bundle.task_type!='classification' or bundle.config.get('training_unit','slide')!='slide' or bundle.config.get('evaluation_unit','slide')!='slide':
        raise ValueError('CoMEL native protocol needs slide bags, binary tasks and patient-grouped partitions')
    if 'comel' in bundle.encoder:raise ValueError('Prepare from original frozen features, not a previously bound CoMEL bundle')
    if not isinstance(schema,dict) or set(schema)!={'schema_version','tasks','slide_tasks','provenance'} or schema['schema_version']!=1:
        raise ValueError('CoMEL schema requires version1, ordered tasks, slide_tasks and explicit provenance')
    tasks=schema['tasks'];mapping=schema['slide_tasks']
    if not isinstance(tasks,list) or len(tasks)<2 or not isinstance(mapping,dict) or not isinstance(schema['provenance'],str) or not schema['provenance'].strip():
        raise ValueError('Declare at least two ordered real tasks, exact slide mapping and source provenance')
    if any(set(t)!={'id','classes'} or not isinstance(t['id'],str) or not t['id'] or not isinstance(t['classes'],list) or len(t['classes'])!=2 for t in tasks):
        raise ValueError('Each CoMEL task has a unique id and an ordered negative/positive class pair')
    if len({t['id'] for t in tasks})!=len(tasks) or [v for t in tasks for v in t['classes']]!=bundle.classes:
        raise ValueError('Global classes must follow exactly task0:[0,1], task1:[2,3], ...')
    if set(mapping)!=set(bundle.rows.slide_id) or set(mapping.values())!={t['id'] for t in tasks}:
        raise ValueError('CoMEL task mapping must cover exactly all actual slides and declared tasks')
    task_index={t['id']:i for i,t in enumerate(tasks)};patients={};rows=[]
    for row in bundle.rows.to_dict('records'):
        sid=row['slide_id'];task=task_index[mapping[sid]];pid=row['global_patient_id']
        if row['target']//2!=task:raise ValueError('CoMEL global target does not match its binary task offset')
        if pid in patients and patients[pid]!=task:raise ValueError('A global patient appears in multiple continual tasks')
        patients[pid]=task;x,_=bundle.load(row);path=resolve(bundle.root,row['path'])
        if len(x)<2:raise ValueError('CoMEL reverse-attention BPPL requires at least two actual instances')
        record={'slide_id':sid,'task':task,'target':row['target'],'subset':row['subset'],
            'global_patient_id':pid,'path':str(path.resolve()),'sha256':digest_file(path),
            'instances':len(x),'patch_ids_sha256':patch_ids(bundle,row,len(x))}
        for key in ['coords_path','patch_ids_path']:
            if row.get(key):
                p=resolve(bundle.root,row[key]);record[key]=str(p.resolve());record[key+'_sha256']=digest_file(p)
        if row.get('instance_labels_path'):
            if row['subset']=='train':raise ValueError('CoMEL TRAIN uses bag labels only; do not bind TRAIN patch ground truth')
            p=resolve(bundle.root,row['instance_labels_path']);labels=load_array(p,'instance_labels')
            if labels.shape!=(len(x),) or not np.issubdtype(labels.dtype,np.integer) or not set(labels.tolist())<={0,1}:
                raise ValueError('CoMEL optional instance ground truth must be aligned binary integer [N]')
            record.update(instance_labels_path=str(p.resolve()),instance_labels_sha256=digest_file(p))
        rows.append(record)
    for task in range(len(tasks)):
        for subset in ['train','val','test']:
            if {r['target'] for r in rows if r['task']==task and r['subset']==subset}!={2*task,2*task+1}:
                raise ValueError('Every binary task needs both classes in patient-disjoint TRAIN/VAL/TEST')
    value={'schema_version':1,'kind':'comel_patient_disjoint_binary_sequence','identity':identity(bundle),
        'tasks':tasks,'provenance':schema['provenance'],'rows':rows,
        'test_used_for_model_or_threshold_selection':False,'training_patch_ground_truth_used':False}
    value['contract']=digest_object(value);p=Path(output).resolve();p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    write_json(p,value);return p


class CoMELInputs:
    def __init__(self,bundle):
        config=bundle.encoder.get('comel')
        if not isinstance(config,dict) or set(config)!={'manifest'}:raise ValueError('Prepare encoder.comel.manifest with ordered real binary tasks')
        self.bundle=bundle;self.path=resolve(bundle.root,config['manifest']);self.value=read_json(self.path)
        value=dict(self.value);self.contract=value.pop('contract',None)
        if self.contract!=digest_object(value) or value.get('kind')!='comel_patient_disjoint_binary_sequence':
            raise ValueError('CoMEL manifest integrity or kind changed')
        # YAML gains only the manifest reference; the actual original joined
        # metadata, selected FM/dimension and task definition stay identical.
        if value.get('identity')!=identity(bundle):raise ValueError('CoMEL original joined metadata, selected FM or task definition changed')
        if bundle.task_type!='classification' or bundle.config.get('training_unit','slide')!='slide' or bundle.config.get('evaluation_unit','slide')!='slide':
            raise ValueError('CoMEL native sequence requires patient-disjoint slide classification')
        self.tasks=value['tasks'];self.rows={r['slide_id']:r for r in value['rows']}
        if len(self.tasks)<2 or len({t['id'] for t in self.tasks})!=len(self.tasks) or any(len(t['classes'])!=2 for t in self.tasks):
            raise ValueError('CoMEL requires at least two unique binary tasks')
        if set(self.rows)!=set(bundle.rows.slide_id) or len(self.rows)!=len(value['rows']):raise ValueError('CoMEL slide population changed')
        if [v for t in self.tasks for v in t['classes']]!=bundle.classes:raise ValueError('CoMEL class/task ordering changed')
        patients={}
        for row in bundle.rows.to_dict('records'):
            r=self.rows[row['slide_id']]
            if any(r[k]!=row[k] for k in ['subset','target','global_patient_id']) or r['task']!=row['target']//2:
                raise ValueError('CoMEL patient, target, partition or task mapping changed')
            if str(resolve(bundle.root,row['path']).resolve())!=r['path']:raise ValueError('CoMEL bound actual feature path changed')
            if row['global_patient_id'] in patients and patients[row['global_patient_id']]!=r['task']:raise ValueError('CoMEL patient crosses tasks')
            patients[row['global_patient_id']]=r['task']
            for key in ['coords_path','patch_ids_path','instance_labels_path']:
                if bool(row.get(key))!=bool(r.get(key)) or (row.get(key) and str(resolve(bundle.root,row[key]).resolve())!=r[key]):
                    raise ValueError('CoMEL aligned metadata/annotation path changed')
        for task in range(len(self.tasks)):
            for subset in ['train','val','test']:
                if {r['target'] for r in self.rows.values() if r['task']==task and r['subset']==subset}!={2*task,2*task+1}:
                    raise ValueError('CoMEL incomplete binary task partition')
        self.verify_bytes()

    def verify_bytes(self):
        # Integrity hashes do not load/decode held-out or future-task features.
        for r in self.rows.values():
            if digest_file(r['path'])!=r['sha256']:raise ValueError('CoMEL bound feature bytes changed')
            for key in ['coords_path','patch_ids_path','instance_labels_path']:
                sha_key='instance_labels_sha256' if key=='instance_labels_path' else key+'_sha256'
                if r.get(key) and digest_file(r[key])!=r[sha_key]:raise ValueError('CoMEL ordered metadata or instance ground truth changed')

    def records(self,task,subset):
        return [r for r in self.bundle.records(subset) if self.rows[r['slide_id']]['task']==task]

    def features(self,row):
        r=self.rows[row['slide_id']]
        if digest_file(r['path'])!=r['sha256']:raise ValueError('CoMEL feature bytes changed during execution')
        x,_=self.bundle.load(row)
        if len(x)!=r['instances'] or patch_ids(self.bundle,row,len(x))!=r['patch_ids_sha256']:
            raise ValueError('CoMEL ordered patch identities changed')
        return x

    def annotations(self,row):
        r=self.rows[row['slide_id']]
        if r['subset']=='train':raise ValueError('CoMEL TRAIN instance ground truth is not available to the learner')
        if not r.get('instance_labels_path'):return None
        if digest_file(r['instance_labels_path'])!=r['instance_labels_sha256']:raise ValueError('CoMEL patch ground truth changed')
        return load_array(r['instance_labels_path'],'instance_labels')
