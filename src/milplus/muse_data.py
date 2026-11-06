"""Immutable full MUSE language banks, patient splits and frozen patch bags."""
from pathlib import Path

import torch

from .data import resolve
from .focus_network import WEIGHTS_SHA
from .io import digest_file, digest_object, read_json, write_json
from .muse_source import COMMIT, bank_assets

KIND = 'muse_released_complete_knowledge_classification'
CLASSES = {'tcga_brca':['IDC','ILC'], 'tcga_nsclc':['LUAD','LUSC'], 'camelyon_all':['normal','tumor']}


def identity(bundle):
    return digest_object({'fm':bundle.fm,'classes':bundle.classes,
        'encoder':{k:v for k,v in bundle.encoder.items() if k!='muse'},
        'synthetic':bundle.config.get('synthetic',False)})


def spec(bundle, metadata):
    if (bundle.task_type!='classification' or len(bundle.classes)!=2 or bundle.dim not in {512,768,1024,1536}
            or bundle.config.get('training_unit','slide')!='slide'
            or bundle.config.get('evaluation_unit','slide')!='slide'):
        raise ValueError('MUSE requires original binary slide training/evaluation and explicit source-supported feature dimensions')
    if (metadata.get('schema_version')!=1 or metadata.get('kind')!=KIND
            or metadata.get('class_order')!=list(bundle.classes)
            or type(metadata.get('synthetic')) is not bool
            or metadata['synthetic']!=bool(bundle.config.get('synthetic',False))):
        raise ValueError('MUSE actual class order/task/synthetic input binding changed')
    assets=bank_assets(metadata.get('cohort'))
    if not metadata['synthetic'] and list(bundle.classes)!=CLASSES[assets['cohort']]:
        raise ValueError('MUSE clinical classes must explicitly follow the original ordered cohort labels')
    for key in ['feature_preprocessing','feature_provenance','label_provenance','train_subset_provenance']:
        if not isinstance(metadata.get(key),str) or not metadata[key].strip():
            raise ValueError('MUSE requires explicit '+key)
    policy=metadata.get('feature_condition')
    if policy=='native_conch':
        if (bundle.dim!=512 or bundle.encoder.get('model')!='conch_ViT-B-16'
                or bundle.encoder.get('weights_sha256')!=WEIGHTS_SHA):
            raise ValueError('MUSE native CONCH condition requires actual original aligned512D features')
    elif policy=='source_feature_adapter':
        if not isinstance(metadata.get('feature_adapter_experiment'),str) or not metadata['feature_adapter_experiment'].strip():
            raise ValueError('MUSE explicit alternate-FM adapter condition needs its experiment identity')
    elif policy!='synthetic_native_dimension_fixture' or not metadata['synthetic']:
        raise ValueError('MUSE feature condition must identify actual native, explicit source adapter or synthetic inputs')
    weights=resolve(bundle.root,metadata['checkpoint']).resolve()
    if digest_file(weights)!=WEIGHTS_SHA:
        raise ValueError('MUSE requires the actual registered complete pretrained CONCH checkpoint')
    return weights,assets


def record(bundle,row):
    feature,_=bundle.load(row)
    if feature.ndim!=2 or len(feature)<1 or feature.shape[1]!=bundle.dim or not torch.isfinite(feature).all():
        raise ValueError('MUSE requires every finite full ordered source feature bag')
    path=resolve(bundle.root,row['path']).resolve()
    return {**{k:row[k] for k in ['slide_id','global_patient_id','patient_id','subset','target']},
            'path':str(path),'sha256':digest_file(path),'patches':len(feature),'dimension':bundle.dim}


def prepare(bundle,metadata_path,output):
    output=resolve(bundle.root,output)
    if output.exists():raise FileExistsError('MUSE input version exists; preserve it and prepare a new manifest')
    metadata_path=resolve(bundle.root,metadata_path).resolve();metadata=read_json(metadata_path)
    weights,assets=spec(bundle,metadata)
    result={'schema_version':1,'kind':KIND,'source_commit':COMMIT,'encoder_identity':identity(bundle),
        'metadata_path':str(metadata_path),'metadata_sha256':digest_file(metadata_path),'metadata':metadata,
        'checkpoint':str(weights),'weights_sha256':WEIGHTS_SHA,'class_prompts':assets['class_prompts'],
        'knowledge_files':assets['sha256'],'descriptions_per_class':[300,300],
        'records':{r['slide_id']:record(bundle,r) for r in bundle.rows.to_dict('records')},
        'preparer_sha256':digest_file(Path(__file__))}
    result['contract']=digest_object(result);write_json(output,result);return output


class MUSEInputs:
    def __init__(self,bundle,manifest):
        self.bundle,self.path=bundle,resolve(bundle.root,manifest);self.meta=read_json(self.path)
        if (self.meta.get('kind')!=KIND or self.meta.get('source_commit')!=COMMIT
                or self.meta.get('encoder_identity')!=identity(bundle)
                or self.meta.get('preparer_sha256')!=digest_file(Path(__file__))
                or self.meta.get('contract')!=digest_object({k:v for k,v in self.meta.items() if k!='contract'})):
            raise ValueError('MUSE source, class/patient/split, encoder or preparation contract changed')
        self.weights,assets=spec(bundle,self.meta['metadata']);self.cohort=assets['cohort']
        if (assets['sha256']!=self.meta['knowledge_files'] or assets['class_prompts']!=self.meta['class_prompts']
                or set(self.meta['records'])!=set(bundle.rows.slide_id)
                or read_json(self.meta['metadata_path'])!=self.meta['metadata']):
            raise ValueError('MUSE complete knowledge descriptions or actual input metadata changed')
        self.files=[self.path,Path(self.meta['metadata_path']),*assets['paths']]
        self.hashes={str(self.path):digest_file(self.path),self.meta['metadata_path']:self.meta['metadata_sha256'],**assets['sha256']}
        self.contract=self.meta['contract'];self._stats={};self._feature_stats={};self.check_unchanged()

    def check_unchanged(self):
        for name,sha in self.hashes.items():
            path=Path(name);stat=path.stat();signature=(stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
            if self._stats.get(name)!=signature:
                if digest_file(path)!=sha:raise ValueError('MUSE immutable full knowledge or metadata bytes changed')
                self._stats[name]=signature

    def load(self,row):
        self.check_unchanged();bound=self.meta['records'][row['slide_id']]
        path=resolve(self.bundle.root,row['path']).resolve();stat=path.stat()
        if (str(path)!=bound['path'] or any(row[k]!=bound[k] for k in ['slide_id','global_patient_id','patient_id','subset','target'])):
            raise ValueError('MUSE bound slide/patient/label/path changed')
        signature=(stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if self._feature_stats.get(str(path))!=signature:
            if record(self.bundle,row)!=bound:raise ValueError('MUSE complete original feature bytes/order changed')
            self._feature_stats[str(path)]=signature
        return self.bundle.load(row)[0]

    def inventory(self,row,deep=False):
        if deep:self.load(row)
        return {'contract':self.contract,'full_feature':self.meta['records'][row['slide_id']],
                'knowledge_files':self.meta['knowledge_files'],'descriptions_per_class':[300,300]}
