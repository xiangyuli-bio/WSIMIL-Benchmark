"""Download only the exact public BioBERT revision reviewed for DisPro."""
import hashlib
import os
import urllib.request
import uuid
from pathlib import Path

from .dispro_source import ASSET_SHA,REPO,REVISION
from .io import digest_file,read_json,write_json


def prepare_biobert(output):
    root=Path(output).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        previous=read_json(root/'manifest.json')
        if previous.get('repo')!=REPO or previous.get('revision')!=REVISION:
            raise ValueError('DisPro refuses to overwrite another asset revision')
    verified={}
    for name,expected in ASSET_SHA.items():
        path=root/name
        if path.exists():
            if path.is_symlink() or digest_file(path)!=expected:
                raise ValueError('DisPro refuses to overwrite existing unverified asset: '+name)
        else:
            url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
            temporary=root/(name+'.'+uuid.uuid4().hex+'.download')
            try:
                sha=hashlib.sha256()
                request=urllib.request.Request(url,headers={'User-Agent':'mil-baseline-puls-dispro'})
                with urllib.request.urlopen(request,timeout=60) as response,temporary.open('xb') as target:
                    while True:
                        block=response.read(8*1024*1024)
                        if not block:break
                        target.write(block);sha.update(block)
                    target.flush();os.fsync(target.fileno())
                if sha.hexdigest()!=expected:
                    raise ValueError('DisPro public pretrained download checksum mismatch: '+name)
                # Exclusive link avoids overwriting a concurrently created user
                # file; the verified temporary inode then loses its extra name.
                os.link(temporary,path)
            finally:
                temporary.unlink(missing_ok=True)
        verified[name]={'bytes':path.stat().st_size,'sha256':expected}
    manifest={'schema_version':1,'repo':REPO,'revision':REVISION,'files':verified,
              'provenance':'Exact public dmis-lab BioBERT-v1.2; full native model, no synthetic weights'}
    write_json(root/'manifest.json',manifest)
    return {'state':'verified','path':str(root),'manifest':str(root/'manifest.json'),
            'repo':REPO,'revision':REVISION,'files':verified}
