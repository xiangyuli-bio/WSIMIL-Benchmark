"""Ordered tissue/class prompts and native CLIP feature identity for SLIP."""
from collections import Counter

from .data import resolve
from .io import digest_file, digest_object, read_json
from .slip_source import ASSETS


class SLIPInputs:
    def __init__(self,bundle):
        self.path=resolve(bundle.root,bundle.encoder['slip']['manifest'])
        self.manifest=meta=read_json(self.path)
        if (meta.get('schema_version')!=1 or meta.get('kind')!='slip_native_clip_prompts'
                or bundle.task_type!='classification'
                or bundle.config.get('training_unit','slide')!='slide'
                or meta.get('classes')!=bundle.classes
                or type(meta.get('synthetic')) is not bool
                or meta['synthetic']!=bool(bundle.config.get('synthetic',False))):
            raise ValueError('SLIP requires ordered slide-classification prompt metadata and matching synthetic markers')
        self.architecture=meta.get('architecture')
        if self.architecture not in ASSETS:
            raise ValueError('SLIP requires an explicit supported native CLIP architecture')
        _,sha,dim=ASSETS[self.architecture]
        if (bundle.dim!=dim or meta.get('weights_sha256')!=sha
                or bundle.encoder.get('weights_sha256')!=sha
                or bundle.encoder.get('model')!=self.architecture
                or meta.get('feature_preprocessing')!=bundle.encoder.get('feature_preprocessing')):
            raise ValueError('SLIP native CLIP feature dimension/model/weights/preprocessing differs; no cross-FM substitution')
        for key in ['prompt_provenance','feature_preprocessing','few_shot_provenance']:
            if not isinstance(meta.get(key),str) or not meta[key].strip():
                raise ValueError('SLIP requires explicit '+key)
        for key in ['slide_classnames','tissue_classnames']:
            groups=meta.get(key)
            if (not isinstance(groups,list) or not groups
                    or any(not isinstance(g,list) or not g or any(not isinstance(s,str) or not s.strip() for s in g) for g in groups)):
                raise ValueError('SLIP requires nonempty ordered '+key)
        if len(meta['slide_classnames'])!=len(bundle.classes):
            raise ValueError('SLIP ordered class descriptions do not match class indices')
        templates=meta.get('templates')
        if (not isinstance(templates,list) or not templates
                or any(not isinstance(s,str) or s.count('{}')!=1 for s in templates)):
            raise ValueError('SLIP templates must contain one positional class/tissue placeholder')
        shots=meta.get('n_shots')
        if shots not in ['1','2','4','8','16','all']:
            raise ValueError('SLIP n_shots must describe a preselected TRAIN-only few-shot split')
        counts=Counter(r['target'] for r in bundle.records('train'))
        if set(counts)!=set(range(len(bundle.classes))):
            raise ValueError('SLIP prepared TRAIN subset must include every class')
        if shots!='all' and any(n!=int(shots) for n in counts.values()):
            raise ValueError('SLIP few-shot counts must equal the declared number of TRAIN slides per class')
        self.sha=digest_file(self.path)
        self.contract=digest_object({'manifest_sha256':self.sha,
            'encoder':{k:v for k,v in bundle.encoder.items() if k!='slip'},
            'ordered_train_slides':[r['slide_id'] for r in bundle.records('train')]})
        self.files=[self.path]

    def check_unchanged(self):
        if digest_file(self.path)!=self.sha:
            raise ValueError('SLIP prompt/feature/few-shot contract changed after loading')
