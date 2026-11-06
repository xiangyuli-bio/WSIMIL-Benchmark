"""Download immutable actual CLIP checkpoints for original SLIP text branches."""
import hashlib
import os
import urllib.request
import uuid
from pathlib import Path

from .io import digest_file,read_json,write_json
from .slip_source import ASSETS


def prepare_clip(output,architectures=None):
    selected=list(ASSETS) if architectures is None else list(architectures)
    if not selected or len(selected)!=len(set(selected)) or any(a not in ASSETS for a in selected):
        raise ValueError('Select distinct supported native SLIP CLIP architectures')
    root=Path(output).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    manifest=root/'manifest.json'
    if manifest.exists():
        old=read_json(manifest)
        if old.get('repo') not in {None,'OpenAI/CLIP'} or old.get('state')!='passed_exact_original_CLIP_assets':
            raise ValueError('SLIP refuses to overwrite another/unverified asset manifest')
    verified={}
    for architecture,(name,expected,_) in ASSETS.items():
        path=root/name
        if architecture not in selected and not path.exists():continue
        url='https://openaipublic.azureedge.net/clip/models/'+expected+'/'+name
        if path.is_symlink():
            raise ValueError('SLIP refuses a symlink in place of an immutable CLIP checkpoint: '+name)
        if path.exists():
            if path.is_symlink() or digest_file(path)!=expected:
                raise ValueError('SLIP refuses to overwrite an existing unverified CLIP checkpoint: '+name)
        else:
            temporary=root/(name+'.'+uuid.uuid4().hex+'.download')
            try:
                sha=hashlib.sha256()
                with urllib.request.urlopen(url,timeout=60) as response,temporary.open('xb') as target:
                    while True:
                        block=response.read(8*1024*1024)
                        if not block:break
                        target.write(block);sha.update(block)
                    target.flush();os.fsync(target.fileno())
                if sha.hexdigest()!=expected:
                    raise ValueError('SLIP original CLIP download checksum mismatch: '+name)
                os.link(temporary,path)
            finally:temporary.unlink(missing_ok=True)
        verified[name]={'bytes':path.stat().st_size,'sha256':expected,'url':url}
    record={'schema_version':1,'repo':'OpenAI/CLIP','state':'passed_exact_original_CLIP_assets','files':verified,
            'scope':'Actual complete original pretrained checkpoints; text towers reconstructed without reducing hidden width/layers'}
    write_json(manifest,record)
    return {'state':'verified','path':str(root),'manifest':str(manifest),'files':verified}
