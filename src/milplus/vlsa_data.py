"""Native aligned CONCH patient bags, explicit prompts and TRAIN-only ranks."""
from pathlib import Path

import numpy as np
import torch

from .data import resolve
from .io import digest_file,digest_object,read_json
from .vlsa_source import WEIGHTS_SHA,time_bins


class VLSAInputs:
    def __init__(self,bundle):
        self.bundle=bundle
        if (bundle.task_type!='survival' or bundle.config.get('training_unit')!='patient'
                or bundle.config.get('evaluation_unit')!='patient'
                or bundle.config.get('time_unit') not in {'days','months'}):
            raise ValueError('VLSA requires patient survival with explicit day/month units')
        path=resolve(bundle.root,bundle.encoder['vlsa']['manifest'])
        self.manifest=meta=read_json(path)
        if (meta.get('schema_version')!=1 or meta.get('kind')!='vlsa_native_conch_survival'
                or type(meta.get('synthetic')) is not bool
                or meta['synthetic']!=bool(bundle.config.get('synthetic',False))):
            raise ValueError('VLSA requires its explicit versioned native CONCH survival manifest')
        for key in ['feature_order_provenance','feature_preprocessing','prompt_provenance','weights_provenance']:
            if not isinstance(meta.get(key),str) or not meta[key].strip():
                raise ValueError('VLSA requires explicit '+key)
        if (bundle.dim!=512 or bundle.encoder.get('model')!='conch_ViT-B-16'
                or bundle.encoder.get('weights_sha256')!=WEIGHTS_SHA
                or bundle.encoder.get('feature_preprocessing')!=meta['feature_preprocessing']):
            raise ValueError('VLSA source requires actual aligned full512 CONCH features; no arbitrary FM projection')
        self.weights=resolve(bundle.root,meta['checkpoint']).resolve()
        if digest_file(self.weights)!=WEIGHTS_SHA:
            raise ValueError('VLSA native pretrained checkpoint bytes differ')
        self.rank_prompts=resolve(bundle.root,meta['rank_prompts']).resolve()
        self.query_prompts=resolve(bundle.root,meta['query_prompts']).resolve()
        self.query_key=meta['query_key']
        queries=read_json(self.query_prompts)
        if (not isinstance(self.query_key,str) or not isinstance(queries.get(self.query_key),list)
                or not queries[self.query_key]
                or any(not isinstance(s,str) or not s.strip() for s in queries[self.query_key])):
            raise ValueError('VLSA requires an explicit ordered prognostic query list')
        self.num_queries=len(queries[self.query_key])
        self.files=[path,self.rank_prompts,self.query_prompts]
        self.file_hashes={str(p):digest_file(p) for p in self.files}
        stat=self.weights.stat();self.weights_stat=(stat.st_size,stat.st_mtime_ns)
        train=bundle.rows.loc[bundle.rows.subset=='train'].drop_duplicates('global_patient_id')
        self.bins=time_bins(train.time.to_numpy(),train.event.to_numpy(),meta.get('num_time_bins'))
        self.num_ranks=len(self.bins)-1;self.coordinates=self.bins[:-1].copy()
        self.train_patients=train.global_patient_id.tolist()
        self.records={}
        for row in bundle.rows.to_dict('records'):
            p=resolve(bundle.root,row['path']).resolve()
            self.records[row['slide_id']]={'path':str(p),'sha256':digest_file(p),'bytes':p.stat().st_size}
        self.contract=digest_object({'files':self.file_hashes,'weights_sha256':WEIGHTS_SHA,
            'encoder':{k:v for k,v in bundle.encoder.items() if k!='vlsa'},
            'feature_inventory':self.records,'train_patients':self.train_patients,
            'finite_time_coordinates':self.coordinates.tolist(),'terminal_edge':'+infinity',
            'query_key':self.query_key,'time_unit':bundle.config['time_unit']})

    def check_unchanged(self):
        if any(digest_file(p)!=sha for p,sha in self.file_hashes.items()):
            raise ValueError('VLSA bound prompt/manifest bytes changed')
        stat=self.weights.stat()
        if (stat.st_size,stat.st_mtime_ns)!=self.weights_stat:
            raise ValueError('VLSA bound original pretrained checkpoint changed')

    def patient(self,rows,device='cpu'):
        self.check_unchanged()
        if not rows or len({r['global_patient_id'] for r in rows})!=1:
            raise ValueError('VLSA input is one true patient with all registered slides')
        gid=rows[0]['global_patient_id']
        expected=self.bundle.rows.loc[self.bundle.rows.global_patient_id==gid,'slide_id'].tolist()
        if [r['slide_id'] for r in rows]!=expected:
            raise ValueError('VLSA requires all patient slides in declared stable slide-ID order')
        features=[]
        for row in rows:
            record=self.records[row['slide_id']];p=Path(record['path'])
            if p!=resolve(self.bundle.root,row['path']).resolve() or digest_file(p)!=record['sha256']:
                raise ValueError('VLSA ordered feature bytes/path changed')
            x,_=self.bundle.load(row);features.append(x)
        return torch.cat(features).to(device)[None]

    def label(self,rows,device='cpu'):
        t=float(rows[0]['time']);event=int(rows[0]['event'])
        index=int(np.searchsorted(self.bins,t,side='right')-1)
        if not np.isfinite(t) or t<=0 or not 0<=index<self.num_ranks or event not in {0,1}:
            raise ValueError('VLSA event/time is outside declared intervals')
        return torch.tensor([index],device=device),torch.tensor([event],device=device)
