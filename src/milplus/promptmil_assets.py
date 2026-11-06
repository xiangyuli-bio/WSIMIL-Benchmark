"""Fetch only the original published Prompt-MIL cohort files, with fixed SHA checks."""
import hashlib
from html.parser import HTMLParser
import os
from pathlib import Path
import urllib.parse
import urllib.request

from .io import digest_file,write_json
from .promptmil_source import WEIGHTS

FILE_IDS={'brca':'1iXooabEQRnhlS2s6NZZ2SkJaCwMrc-Am',
          'bright':'1ZPO1a1jr9UuVrdiHwvBso57VXOTwLChb','crc':'1SdSbIhdKVwXVJhaVBGqpUpisrQ75MOR1'}


class _Form(HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.fields={}
    def handle_starttag(self,tag,attrs):
        item=dict(attrs)
        if tag=='form':self.action=item.get('action')
        if tag=='input'and item.get('name'):self.fields[item['name']]=item.get('value','')


def fetch(output,condition):
    if condition not in WEIGHTS:raise ValueError('Choose original Prompt-MIL brca, bright or crc weights')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    target=root/WEIGHTS[condition]['filename'];expected=WEIGHTS[condition]['sha256'];fid=FILE_IDS[condition]
    if target.exists():
        if digest_file(target)!=expected:raise ValueError('Refuse to overwrite different existing Prompt-MIL weights')
        return target
    url='https://drive.google.com/uc?'+urllib.parse.urlencode({'export':'download','id':fid})
    response=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=40)
    first=response.read(4096)
    if not first.startswith((b'PK',b'\x80')):
        html=first+response.read(2**20);response.close();form=_Form();form.feed(html.decode('utf-8'))
        if (not form.action or urllib.parse.urlparse(form.action).scheme!='https'
            or urllib.parse.urlparse(form.action).hostname!='drive.usercontent.google.com'
            or form.fields.get('id')!=fid):raise ValueError('Original public weight download did not return a checkpoint or standard confirmation form')
        response=urllib.request.urlopen(urllib.request.Request(form.action+'?'+urllib.parse.urlencode(form.fields),headers={'User-Agent':'Mozilla/5.0'}),timeout=40)
        first=response.read(4096)
    if not first.startswith((b'PK',b'\x80')):
        response.close();raise ValueError('Original weight endpoint returned HTML or a non-checkpoint response')
    partial=root/(target.name+'.'+str(os.getpid())+'.part');total=len(first);sha=hashlib.sha256(first)
    with response,partial.open('xb')as out:
        out.write(first)
        for block in iter(lambda:response.read(4*2**20),b''):
            total+=len(block)
            if total>600*2**20:raise ValueError('Unexpected original checkpoint size')
            sha.update(block);out.write(block)
        out.flush();os.fsync(out.fileno())
    if sha.hexdigest()!=expected:raise ValueError('Original Prompt-MIL checkpoint SHA mismatch; partial bytes preserved for diagnosis')
    partial.rename(target)
    write_json(root/(target.name+'.provenance.json'),{'condition':condition,'url':url,'bytes':total,'sha256':expected,
        'origin':'PromptMIL author README published Google Drive folder','authorized_personal_account_required':False})
    return target
