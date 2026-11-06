"""Explicit networked-CPU preparation; training never fetches weights online."""
import hashlib
import os
from pathlib import Path
import urllib.request
import uuid

from .icmil_coupled_source import IMAGENET_SHA256
from .io import digest_file,write_json


def fetch(output):
    root=Path(output).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    target=root/'resnet50-11ad3fa6.pth';url='https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
    if target.exists():
        if digest_file(target)!=IMAGENET_SHA256:raise ValueError('Existing original ICMIL ImageNet weights changed; preserve and inspect them')
        return target
    partial=root/('resnet50.'+uuid.uuid4().hex+'.partial');h=hashlib.sha256()
    with urllib.request.urlopen(url,timeout=30) as source,partial.open('xb') as out:
        for block in iter(lambda:source.read(1024*1024),b''):out.write(block);h.update(block)
    if h.hexdigest()!=IMAGENET_SHA256:raise ValueError('Downloaded original encoder checksum mismatch; failed bytes preserved')
    os.replace(partial,target)
    write_json(root/'assets.json',{'url':url,'weights':target.name,'sha256':IMAGENET_SHA256,
        'bytes':target.stat().st_size,'verified':True,'not_downloaded_during_training':True})
    return target
