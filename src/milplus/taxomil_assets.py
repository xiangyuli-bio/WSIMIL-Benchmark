"""Fetch the actual full original public GPT2 assets on a networked CPU host."""
import hashlib
from pathlib import Path
import urllib.request

from .io import write_json
from .taxomil_source import GPT2_REVISION

FILES={
    'config.json':(665,'0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4'),
    'merges.txt':(456318,'1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'),
    'tokenizer.json':(1355256,'8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'),
    'tokenizer_config.json':(26,'5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f'),
    'vocab.json':(1042301,'196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'),
    'model.safetensors':(548105171,'248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'),
}


def fetch(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);records={}
    for name,(size,expected) in FILES.items():
        url='https://huggingface.co/openai-community/gpt2/resolve/'+GPT2_REVISION+'/'+name
        sha=hashlib.sha256();count=0;target=root/name;partial=root/(name+'.partial')
        with urllib.request.urlopen(url,timeout=120) as response,partial.open('xb') as stream:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                stream.write(chunk);sha.update(chunk);count+=len(chunk)
        if count!=size or sha.hexdigest()!=expected:
            raise ValueError('TaxoMIL original GPT2 asset hash/size mismatch; partial retained: '+name)
        partial.rename(target)
        records[name]={'bytes':size,'sha256':expected,'url':url,'revision':GPT2_REVISION}
    write_json(root/'milplus_assets.json',{'state':'passed','repo':'openai-community/gpt2',
        'revision':GPT2_REVISION,'files':records,'original_full_pretrained_weights':True,'formal_admission':False})
    return root
