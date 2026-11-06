"""Bind user-supplied original TITAN files to publicly advertised object IDs.

This module does not download restricted assets, accept access conditions or
replace missing pretrained tensors. Author code remains private at runtime.
"""
import ast
import hashlib
import importlib
import json
from pathlib import Path

import torch

from .io import digest_file, digest_object, read_json, write_json
from .registry import resource


def native_lock():
    value = resource('dyko_titan.lock.json')
    if (value.get('model_id') != 'MahmoodLab/TITAN'
            or value.get('revision') != 'dac6773d9961cfc75503440676ff157a2c6e8d2e'
            or value['files'].get('model.safetensors',{}).get('sha256') != '4e92128317dca5c9e47d8173467105c21e4095f88d3bde9077e7f1e784b195fe'):
        raise ValueError('DyKo actual original TITAN asset identity changed')
    return value


def git_blob_sha1(path):
    path = Path(path)
    digest = hashlib.sha1()
    digest.update(('blob '+str(path.stat().st_size)+'\0').encode())
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def original_files(directory):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError('Provide the local directory of original TITAN assets for which you have access')
    lock = native_lock();records = {}
    missing = [name for name in lock['files'] if not (root/name).is_file()]
    if missing:
        raise FileNotFoundError('Original TITAN files missing: '+', '.join(missing))
    for name,expected in lock['files'].items():
        path = root/name;stat = path.stat()
        if stat.st_size != expected['bytes']:
            raise ValueError('Original TITAN file size differs from its published object: '+name)
        if 'sha256' in expected:
            if digest_file(path) != expected['sha256']:
                raise ValueError('Original TITAN checkpoint SHA256 mismatch: '+name)
        elif git_blob_sha1(path) != expected['git_blob_sha1']:
            raise ValueError('Original TITAN config/code/tokenizer Git-object mismatch: '+name)
        records[name] = {'path':str(path.resolve()),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                         'sha256':digest_file(path)}
    # The verified official code's relative imports must not resolve to extra
    # executable Python files under an ostensibly verified local model folder.
    extras = {p.name for p in root.glob('*.py')}-set(lock['files'])
    if extras:
        raise ValueError('Unbound executable files in the original TITAN model directory: '+', '.join(sorted(extras)))
    return root,records


def prepare(directory, output):
    root,files = original_files(directory)
    lock = native_lock()
    value = {'schema_version':1,'kind':'dyko_verified_original_TITAN_local_assets',
             'model_id':lock['model_id'],'revision':lock['revision'],'directory':str(root),
             'files':files,'public_identity_lock_sha256':digest_object(lock),
             'model_loaded':False,'full_DyKo_validation':False,
             'source_TITAN_revision_was_unpinned':True}
    value['contract'] = digest_object(value)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    write_json(output,value)
    return output


class OriginalTITANAssets:
    def __init__(self, manifest):
        self.manifest = Path(manifest).expanduser().resolve()
        original = read_json(self.manifest);value = dict(original)
        contract = value.pop('contract',None)
        lock = native_lock()
        if (contract != digest_object(value) or value.get('schema_version') != 1
                or value.get('kind') != 'dyko_verified_original_TITAN_local_assets'
                or value.get('model_id') != lock['model_id'] or value.get('revision') != lock['revision']
                or value.get('public_identity_lock_sha256') != digest_object(lock)):
            raise ValueError('DyKo original TITAN asset manifest changed')
        self.root,self.files = original_files(value['directory'])
        if self.files != value['files']:
            raise ValueError('Original TITAN file bytes or physical identity changed after preparation')
        self.value = original;self.contract = contract

    def verify(self, deep=False):
        if read_json(self.manifest) != self.value:
            raise ValueError('Original TITAN asset binding changed')
        for name,record in self.files.items():
            path = self.root/name;stat = path.stat()
            if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns'] or (deep and digest_file(path) != record['sha256']):
                raise ValueError('Actual original TITAN bytes changed: '+name)

    def cache_view(self, directory):
        """Private offline HF view, retaining the original published model ID."""
        self.verify(deep=True)
        directory = Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
        repo = directory/'models--MahmoodLab--TITAN'
        snapshots = repo/'snapshots';snapshots.mkdir(parents=True,exist_ok=True)
        target = snapshots/self.value['revision']
        if target.is_symlink():
            if target.resolve() != self.root:
                raise ValueError('DyKo offline TITAN cache points to different assets')
        elif target.exists():
            raise ValueError('DyKo offline TITAN cache is not the declared original asset view')
        else:
            target.symlink_to(self.root,target_is_directory=True)
        refs = repo/'refs';refs.mkdir(exist_ok=True)
        if (refs/'main').exists() and (refs/'main').read_text().strip() != self.value['revision']:
            raise ValueError('DyKo offline TITAN revision changed')
        (refs/'main').write_text(self.value['revision'])
        return directory

    def load(self):
        """Actual full local AutoModel, including its original text tower.

        Config, executable files, tokenizer and checkpoint have published IDs.
        All model tensors are compared with the actual safetensors payload.
        The original DyKo constructor subsequently chooses its text submodules.
        """
        self.verify(deep=True)
        import transformers
        from safetensors import safe_open
        if transformers.__version__ != '4.46.3':
            raise ValueError('Use the explicit DyKo original-transformers4.46.3 runtime')
        self._prepare_module_cache()
        loaded = transformers.AutoModel.from_pretrained(str(self.root),trust_remote_code=True,
            local_files_only=True,use_safetensors=True,output_loading_info=True)
        if not isinstance(loaded,tuple) or len(loaded) != 2:
            raise ValueError('Original TITAN loader did not expose its pretrained loading evidence')
        model,info = loaded
        if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
            raise ValueError('Original TITAN complete pretrained tensors were not loaded exactly')
        model = model.float().cpu()
        actual = model.state_dict()
        with safe_open(self.root/'model.safetensors',framework='pt',device='cpu') as stored:
            if set(stored.keys()) != set(actual):
                raise ValueError('Original full TITAN state differs from its complete checkpoint key set')
            for key in stored.keys():
                source = stored.get_tensor(key)
                target = actual[key].detach().cpu()
                if source.shape != target.shape or not torch.equal(source.to(target.dtype),target):
                    raise ValueError('Actual original TITAN tensor differs from its checkpoint: '+key)
        text = getattr(model,'text_encoder',None)
        if (text is None or any(not hasattr(text,key) for key in
            ['transformer','positional_embedding','ln_final','text_projection','tokenizer'])
                or not callable(getattr(model,'encode_text',None))
                or tuple(text.text_projection.shape)[-1] != 768):
            raise ValueError('Actual original TITAN contextual text interface is missing')
        model._milplus_original_titan_evidence = {
            'model_id':self.value['model_id'],'revision':self.value['revision'],
            'asset_contract':self.contract,'all_original_model_tensors_exact':True,
            'full_parameters':sum(p.numel() for p in model.parameters()),
            'full_text_parameters':sum(p.numel() for p in text.parameters()),
            'checkpoint_sha256':self.files['model.safetensors']['sha256']}
        return model

    def _prepare_module_cache(self):
        """Populate the full locked relative-import closure for local loading.

        Transformers 4.46.3 does not reliably copy transitive local modules.
        Preserve the official code bytes and reject conflicting cached code.
        """
        from transformers.dynamic_module_utils import HF_MODULES_CACHE, create_dynamic_module
        scripts = {name for name in self.files if name.endswith('.py')}
        for name in sorted(scripts):
            path = self.root/name
            if digest_file(path) != self.files[name]['sha256']:
                raise ValueError('Original TITAN module changed: '+name)
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom) and node.level:
                    dependency = (node.module or '').replace('.','/')+'.py'
                    if node.level != 1 or dependency not in scripts:
                        raise ValueError('Unbound original TITAN relative import: '+name)
        namespace = 'transformers_modules/'+self.root.name
        cache = Path(HF_MODULES_CACHE)
        destination = cache/namespace
        for directory in [cache,cache/'transformers_modules',destination]:
            if directory.is_symlink():
                raise ValueError('DyKo dynamic module cache must not be a symlink')
        create_dynamic_module(namespace)
        for name in sorted(scripts):
            target = destination/name
            if target.is_symlink():
                raise ValueError('DyKo cached module must not be a symlink: '+name)
            if not target.exists():
                with target.open('xb') as stream:
                    stream.write((self.root/name).read_bytes())
            if digest_file(target) != self.files[name]['sha256']:
                raise ValueError('DyKo cached module differs from original: '+name)
        importlib.invalidate_caches()
