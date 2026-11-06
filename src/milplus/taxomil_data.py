"""Explicit original pretrained text asset and coarse/fine clinical taxonomy."""
from pathlib import Path

from .data import resolve
from .io import digest_file, digest_object, read_json
from .taxomil_source import GPT2_REVISION, GPT2_SHA256


class TaxoMILInputs:
    def __init__(self,bundle):
        if (bundle.task_type!='classification' or bundle.hierarchy is None
                or bundle.config.get('training_unit','slide')!='slide'
                or bundle.config.get('evaluation_unit','slide')!='slide'):
            raise ValueError('TaxoMIL requires original coarse/fine slide taxonomy and patient-grouped splits')
        value=bundle.encoder['taxomil']
        if not isinstance(value,dict) or set(value)!={'assets','dataset'}:
            raise ValueError('encoder.taxomil must declare exactly assets and original dataset domain')
        if value['dataset'] not in {'BRACS','PANDA'}:
            raise ValueError('TaxoMIL domain must be explicitly BRACS or PANDA; no invented generic gastric fallback')
        self.bundle=bundle;self.dataset=value['dataset']
        self.assets=resolve(bundle.root,value['assets'])
        self.manifest=self.assets/'milplus_assets.json';self.metadata=read_json(self.manifest)
        if (self.metadata.get('repo')!='openai-community/gpt2' or self.metadata.get('revision')!=GPT2_REVISION
                or self.metadata.get('state')!='passed' or not self.metadata.get('original_full_pretrained_weights')):
            raise ValueError('TaxoMIL requires actual complete pinned GPT2 assets and verification receipt')
        names={'config.json','merges.txt','tokenizer.json','tokenizer_config.json','vocab.json','model.safetensors'}
        if set(self.metadata.get('files',{}))!=names:
            raise ValueError('TaxoMIL original GPT2 asset set is incomplete')
        self.files=[self.manifest]+[self.assets/name for name in sorted(names)]
        self._identity={}
        for name in sorted(names):
            path=self.assets/name;row=self.metadata['files'][name]
            if (not path.is_file() or path.stat().st_size!=row['bytes'] or digest_file(path)!=row['sha256']
                    or name=='model.safetensors' and row['sha256']!=GPT2_SHA256):
                raise ValueError('TaxoMIL original pretrained asset hash/size mismatch: '+name)
            self._identity[name]={'sha256':row['sha256'],'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
        config=read_json(self.assets/'config.json')
        if (config.get('n_embd'),config.get('n_layer'),config.get('n_head'),config.get('vocab_size'))!=(768,12,12,50257):
            raise ValueError('TaxoMIL requires full original GPT2 768D/12layer/12head/50257vocabulary')
        self._manifest_sha=digest_file(self.manifest)
        self.taxonomy=bundle.hierarchy.protocol()
        self.contract=digest_object({'assets':self._identity,'manifest':self._manifest_sha,
            'dataset':self.dataset,'taxonomy':self.taxonomy,'bundle':bundle.fingerprint if hasattr(bundle,'fingerprint') else None})

    def check_inputs(self):
        if digest_file(self.manifest)!=self._manifest_sha or self.bundle.hierarchy.protocol()!=self.taxonomy:
            raise ValueError('TaxoMIL bound language/taxonomy metadata changed')
        for name,row in self._identity.items():
            path=self.assets/name
            if not path.is_file() or path.stat().st_size!=row['bytes'] or path.stat().st_mtime_ns!=row['mtime_ns']:
                raise ValueError('TaxoMIL original pretrained asset changed: '+name)

    def protocol(self):
        return {'contract':self.contract,'model':'openai-community/gpt2','revision':GPT2_REVISION,
            'assets':self._identity,'dataset':self.dataset,'taxonomy':self.taxonomy}
