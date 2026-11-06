"""Explicit real/masked modality presence and TRAIN-only RNA preprocessing.

Missing WSI slots are URI metadata, never fabricated arrays. A patient may lack
one modality, but both missing is an invalid DisPro input. Assays are inputs here
(unlike AMIML mutation labels); outcomes never enter inputs() or token scoring.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import resolve
from .io import digest_file, digest_object, read_json


class DisProInputs:
    def __init__(self, bundle):
        self.bundle = bundle
        if (bundle.task_type != 'survival' or bundle.config.get('training_unit') != 'patient'
                or bundle.config.get('evaluation_unit') != 'patient'
                or bundle.config.get('time_unit') not in {'months','days'}):
            raise ValueError('DisPro requires patient-level survival with actual event/time units')
        config = bundle.encoder['dispro']
        manifest_path = resolve(bundle.root, config['manifest']); meta = read_json(manifest_path)
        if (meta.get('schema_version') != 1 or meta.get('kind') != 'dispro_incomplete_patient_inputs'
                or type(meta.get('synthetic')) is not bool
                or meta['synthetic'] != bool(bundle.config.get('synthetic',False))):
            raise ValueError('DisPro needs its versioned incomplete-patient manifest and matching synthetic marker')
        for key in ['missingness_provenance','assay_preprocessing','pathway_provenance']:
            if not isinstance(meta.get(key),str) or not meta[key].strip():
                raise ValueError('DisPro manifest requires explicit '+key)
        presence_path, expr_path, pathway_path = [resolve(bundle.root,meta[k]) for k in
                                                ['presence','expression','pathways']]
        self.files = [manifest_path,presence_path,expr_path,pathway_path]
        presence = pd.read_csv(presence_path,dtype=str,keep_default_na=False)
        expected = set(bundle.rows.global_patient_id)
        if (not {'global_patient_id','wsi_observed','omics_observed'} <= set(presence)
                or presence.global_patient_id.eq('').any() or presence.global_patient_id.duplicated().any()
                or set(presence.global_patient_id) != expected):
            raise ValueError('DisPro presence requires exactly one row for every task patient')
        if any(not set(presence[k]) <= {'0','1'} for k in ['wsi_observed','omics_observed']):
            raise ValueError('DisPro observed modality flags must be literal0/1, not zero-value inference')
        self.presence = presence.set_index('global_patient_id')[['wsi_observed','omics_observed']].astype(int)
        if (self.presence.sum(1)==0).any():
            raise ValueError('DisPro cannot infer from both modalities missing; no fabricated empty bags')
        observed = set(self.presence.index[self.presence.omics_observed==1])
        with expr_path.open(newline='') as stream: header = next(csv.reader(stream))
        if len(header)!=len(set(header)): raise ValueError('DisPro expression CSV has duplicate columns')
        columns = meta.get('gene_columns')
        if (not isinstance(columns,list) or not columns or len(columns)!=len(set(columns))
                or any(not isinstance(c,str) or not c for c in columns)
                or set(columns)&{'global_patient_id','time','event','subset','label'}):
            raise ValueError('DisPro requires unique explicit ordered gene columns, without identifiers/outcomes')
        self.columns = columns
        frame = pd.read_csv(expr_path,dtype=str,keep_default_na=False)
        if (not {'global_patient_id',*columns} <= set(frame)
                or frame.global_patient_id.eq('').any() or frame.global_patient_id.duplicated().any()
                or not set(frame.global_patient_id)<=expected or not observed<=set(frame.global_patient_id)):
            raise ValueError('DisPro observed RNA patient identity/ordered gene columns do not match task/presence')
        self.values = frame.set_index('global_patient_id').loc[sorted(observed),columns].apply(pd.to_numeric,errors='raise')
        array = self.values.to_numpy(dtype=np.float64)
        if not np.isfinite(array).all() or not np.isfinite(array.astype(np.float32)).all():
            raise ValueError('DisPro observed RNA must be finite float32-representable; no implicit imputation')
        self.pathways = read_json(pathway_path)
        if not isinstance(self.pathways,list) or not self.pathways:
            raise ValueError('DisPro requires a nonempty ordered pathway/gene mapping')
        names, indices = [], []
        for pathway in self.pathways:
            if (not isinstance(pathway,dict) or not isinstance(pathway.get('name'),str)
                    or not pathway['name'].strip() or not isinstance(pathway.get('features'),list)
                    or not pathway['features'] or len(pathway['features'])!=len(set(pathway['features']))
                    or not set(pathway['features'])<=set(columns)):
                raise ValueError('DisPro pathways require real unique ordered genes, no silent absent-gene filtering')
            names.append(pathway['name']);indices.append([columns.index(g) for g in pathway['features']])
        if len(names)!=len(set(names)): raise ValueError('DisPro pathway names must be unique')
        self.group_indices = indices; self.omic_sizes = list(map(len,indices))
        train = set(bundle.rows.loc[bundle.rows.subset=='train','global_patient_id'])
        if not train&observed or not any(self.observed(g,'WSI') for g in train):
            raise ValueError('DisPro two UniPro stages each require observed TRAIN patients for their modality')
        normalization = meta.get('normalization','none')
        if normalization not in {'none','train_standard'}:
            raise ValueError('DisPro supports declared upstream RNA values or explicit TRAIN-only standardization')
        train_values = self.values.loc[sorted(train&observed)].to_numpy(dtype=np.float64)
        self.mean = np.zeros(len(columns));self.scale = np.ones(len(columns))
        if normalization=='train_standard':
            self.mean,self.scale = train_values.mean(0),train_values.std(0)
            self.scale[self.scale<np.finfo(np.float32).eps] = 1.
        self.normalization = normalization
        self.contract = digest_object({'files':{str(p):digest_file(p) for p in self.files},
            'normalization':normalization,'train_observed_rna':sorted(train&observed),
            'mean':self.mean.tolist(),'scale':self.scale.tolist(),'dimension':bundle.dim,
            'ordered_genes':columns,'ordered_pathways':self.pathways})
        for gid,rows in bundle.rows.groupby('global_patient_id',sort=False):
            absent_uri = 'missing://wsi/'+gid
            if any(str(p).startswith('missing://') for p in rows.path):
                if self.observed(gid,'WSI') or len(rows)!=1 or rows.path.iloc[0]!=absent_uri:
                    raise ValueError('DisPro missing-WSI URI is only an explicit absent patient slot, never a slide feature')

    def observed(self, patient, modal):
        if modal not in {'WSI','Omics'}: raise ValueError('DisPro modality must be WSI or Omics')
        return bool(self.presence.loc[patient,'wsi_observed' if modal=='WSI' else 'omics_observed'])

    def inputs(self, rows, device='cpu', mode='available'):
        if (not rows or len({r['global_patient_id'] for r in rows})!=1
                or mode not in {'available','wsi','omics','wsi-omics'}):
            raise ValueError('DisPro inputs require one true patient and an explicit observed-modality mode')
        patient = rows[0]['global_patient_id']
        expected = self.bundle.rows.loc[self.bundle.rows.global_patient_id==patient,'slide_id'].tolist()
        actual = [r['slide_id'] for r in rows]
        if len(actual)!=len(expected) or set(actual)!=set(expected):
            raise ValueError('DisPro requires all registered slides of the patient, without partial bags')
        wsi = self.observed(patient,'WSI') and mode!='omics'
        omics = self.observed(patient,'Omics') and mode!='wsi'
        if not (wsi or omics):
            raise ValueError('DisPro requested modality is genuinely missing for this patient; do not substitute zeros')
        if wsi:
            x = torch.cat([self.bundle.load(r)[0] for r in rows]).to(device)[None]
        else: x = None
        genes = None
        if omics:
            values = (self.values.loc[patient].to_numpy(dtype=np.float64)-self.mean)/self.scale
            if not np.isfinite(values).all() or not np.isfinite(values.astype(np.float32)).all():
                raise ValueError('DisPro transformed RNA is nonfinite')
            vector = torch.tensor(values,dtype=torch.float32,device=device)
            genes = {f'x_omic{i+1}':vector[ids][None] for i,ids in enumerate(self.group_indices)}
        # No labels/censorship/time are forwarded. They belong exclusively to
        # the training loss, and inference code never receives them as inputs.
        return {'x_WSI':x,'x_Omics':genes}

    def inventory(self,row,deep=False):
        patient = row['global_patient_id']
        item = {'slide_id':row['slide_id'],'path':row['path'],'dispro_contract':self.contract,
            'global_patient_id':patient,'wsi_observed':self.observed(patient,'WSI'),
            'omics_observed':self.observed(patient,'Omics')}
        if not item['wsi_observed']:
            item['feature_kind'] = 'explicit_missing_or_masked_WSI_no_array'
            return item
        path = resolve(self.bundle.root,row['path'])
        item.update(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        if deep:
            x,_ = self.bundle.load(row)
            item.update(instances=len(x),sha256=digest_file(path))
            if row.get('sha256') and row['sha256']!=item['sha256']:
                raise ValueError('DisPro observed WSI feature hash changed')
        return item
