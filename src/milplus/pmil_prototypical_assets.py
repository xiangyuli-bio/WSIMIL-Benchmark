"""Explicit networked-CPU download of the authentic PMIL ResNet34 initializer."""
from pathlib import Path
import urllib.request

from .io import digest_file,write_json
from .pmil_prototypical_source import IMAGENET_SHA256


def fetch(output):
    root=Path(output).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    target=root/'resnet34-333f7ec4.pth';url='https://download.pytorch.org/models/'+target.name
    if target.exists():
        if digest_file(target)!=IMAGENET_SHA256:raise ValueError('Existing original PMIL weights have different bytes; retain evidence and use a new output')
        return target
    partial=target.with_suffix('.pth.partial')
    with urllib.request.urlopen(url,timeout=60) as source,partial.open('xb') as destination:
        while chunk:=source.read(1024*1024):destination.write(chunk)
    if digest_file(partial)!=IMAGENET_SHA256:raise ValueError('Official PMIL ImageNet download checksum mismatch; partial preserved')
    partial.rename(target);write_json(root/'weights_receipt.json',{'state':'passed','url':url,'sha256':IMAGENET_SHA256,'bytes':target.stat().st_size})
    return target
