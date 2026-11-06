"""Actual ordered feature tasks, including singleton incremental task classes."""
from pathlib import Path

from .data import resolve
from .comel_data import patch_ids
from .io import read_json,write_json,digest_file,digest_object


def identity(bundle):
    return digest_object({'task':bundle.config['task_id'],'type':bundle.task_type,'classes':bundle.classes,
        'fm':bundle.fm,'dimension':bundle.dim,'training_unit':bundle.config.get('training_unit','slide'),
        'evaluation_unit':bundle.config.get('evaluation_unit','slide'),'rows':bundle.rows.to_dict('records')})


def _sequence(bundle,tasks,mapping):
    if bundle.task_type!='classification' or bundle.config.get('training_unit','slide')!='slide' or bundle.config.get('evaluation_unit','slide')!='slide':
        raise ValueError('AKD-PMP needs slide classification and patient-disjoint partitions')
    if not isinstance(tasks,list) or len(tasks)<2 or any(not isinstance(t,dict) or set(t)!={'id','classes'} or not isinstance(t['id'],str) or not t['id'] or not isinstance(t['classes'],list) or not t['classes'] for t in tasks):
        raise ValueError('Declare at least two ordered tasks with unique IDs and nonempty original class lists')
    if len({t['id'] for t in tasks})!=len(tasks) or [c for t in tasks for c in t['classes']]!=bundle.classes:
        raise ValueError('AKD-PMP global classes must be the exact concatenation of ordered task classes')
    if set(mapping)!=set(bundle.rows.slide_id) or set(mapping.values())!={t['id'] for t in tasks}:
        raise ValueError('Each original slide must map to exactly its declared task')
    label_to_task={label:task for task,t in enumerate(tasks) for label in t['classes']};task_ids={t['id']:i for i,t in enumerate(tasks)}
    patients={}
    for r in bundle.rows.to_dict('records'):
        task=task_ids[mapping[r['slide_id']]]
        if label_to_task[r['label']]!=task:raise ValueError('Slide class does not belong to its declared task')
        pid=r['global_patient_id']
        if pid in patients and patients[pid]!=task:raise ValueError('Global patient crosses continual tasks')
        patients[pid]=task
    for t,task in enumerate(tasks):
        for subset in ['train','val','test']:
            observed={r['label'] for r in bundle.records(subset) if mapping[r['slide_id']]==task['id']}
            if observed!=set(task['classes']):raise ValueError('Every task class requires independent TRAIN/VAL/TEST patients')


def prepare(bundle,schema,output):
    if not isinstance(schema,dict) or set(schema)!={'schema_version','tasks','slide_tasks','provenance'} or schema['schema_version']!=1 or not isinstance(schema['provenance'],str) or not schema['provenance'].strip():
        raise ValueError('AKD-PMP preparation needs version1, tasks, slide_tasks and actual provenance')
    if 'akdpmp' in bundle.encoder:raise ValueError('Prepare from original feature inputs, not an already bound AKD-PMP manifest')
    _sequence(bundle,schema['tasks'],schema['slide_tasks']);indices={t['id']:i for i,t in enumerate(schema['tasks'])};rows=[]
    for row in bundle.rows.to_dict('records'):
        x,_=bundle.load(row);p=resolve(bundle.root,row['path'])
        if len(x)<2:raise ValueError('Original CLAM instance constraints require at least two actual patches')
        record={'slide_id':row['slide_id'],'task':indices[schema['slide_tasks'][row['slide_id']]],
            **{k:row[k] for k in ['target','subset','global_patient_id']},'path':str(p.resolve()),
            'sha256':digest_file(p),'patches':len(x),'ordered_ids_sha256':patch_ids(bundle,row,len(x))}
        for key in ['coords_path','patch_ids_path']:
            if row.get(key):
                p=resolve(bundle.root,row[key]);record[key]=str(p.resolve());record[key+'_sha256']=digest_file(p)
        rows.append(record)
    value={'schema_version':1,'kind':'akdpmp_actual_frozen_tasks','identity':identity(bundle),'tasks':schema['tasks'],
        'rows':rows,'provenance':schema['provenance'],'test_used_for_selection':False}
    value['contract']=digest_object(value);out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    write_json(out,value);return out


class AKDInputs:
    def __init__(self,bundle):
        config=bundle.encoder.get('akdpmp')
        if not isinstance(config,dict) or set(config)!={'manifest'}:raise ValueError('Prepare encoder.akdpmp.manifest with ordered real tasks and patch IDs')
        self.bundle=bundle;self.path=resolve(bundle.root,config['manifest']);value=read_json(self.path);self.contract=value.pop('contract',None)
        if self.contract!=digest_object(value) or value.get('kind')!='akdpmp_actual_frozen_tasks' or value.get('schema_version')!=1:
            raise ValueError('AKD-PMP manifest kind/integrity changed')
        if value['identity']!=identity(bundle):raise ValueError('AKD-PMP patient/label/partition/FM metadata changed')
        self.tasks=value['tasks'];self.rows={r['slide_id']:r for r in value['rows']}
        if len(self.rows)!=len(value['rows']) or set(self.rows)!=set(bundle.rows.slide_id):raise ValueError('AKD-PMP actual slide population changed')
        mapping={r['slide_id']:self.tasks[self.rows[r['slide_id']]['task']]['id'] for r in bundle.rows.to_dict('records')}
        _sequence(bundle,self.tasks,mapping)
        for r in bundle.rows.to_dict('records'):
            item=self.rows[r['slide_id']]
            if any(r[k]!=item[k] for k in ['target','subset','global_patient_id']) or str(resolve(bundle.root,r['path']).resolve())!=item['path']:
                raise ValueError('AKD-PMP feature path or original joined metadata changed')
            for key in ['coords_path','patch_ids_path']:
                if bool(r.get(key))!=bool(item.get(key)) or (r.get(key) and str(resolve(bundle.root,r[key]).resolve())!=item[key]):
                    raise ValueError('AKD-PMP actual ordered metadata path changed')
        self.verify_bytes()

    def classes(self,task):
        start=sum(len(t['classes']) for t in self.tasks[:task]);return list(range(start,start+len(self.tasks[task]['classes'])))

    def seen(self,task):return [c for t in range(task+1) for c in self.classes(t)]

    def records(self,task,subset):return [r for r in self.bundle.records(subset) if self.rows[r['slide_id']]['task']==task]

    def verify_bytes(self):
        for row in self.rows.values():
            if digest_file(row['path'])!=row['sha256']:raise ValueError('AKD-PMP actual feature bytes changed')
            for key in ['coords_path','patch_ids_path']:
                if row.get(key) and digest_file(row[key])!=row[key+'_sha256']:raise ValueError('AKD-PMP actual ordered metadata bytes changed')

    def features(self,row):
        item=self.rows[row['slide_id']]
        if digest_file(item['path'])!=item['sha256']:raise ValueError('AKD-PMP actual features changed during execution')
        x,_=self.bundle.load(row)
        if len(x)!=item['patches'] or patch_ids(self.bundle,row,len(x))!=item['ordered_ids_sha256']:
            raise ValueError('AKD-PMP feature row count or real patch order changed')
        return x
