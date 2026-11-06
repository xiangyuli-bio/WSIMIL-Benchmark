"""Public CLI plus small IO doubles; native model proofs live in other tests."""
import hashlib
import io
import json
import subprocess
import sys

import pytest

from milplus import slip_assets


def tiny_assets(monkeypatch):
    body=b'fixture for asset IO only; never a pretrained model'
    sha=hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(slip_assets,'ASSETS',{'clip_vit_b16':('ViT-B-16.pt',sha,512)})
    return body


def test_asset_io_hash_and_idempotent_verified_publication(tmp_path,monkeypatch):
    body=tiny_assets(monkeypatch);calls=[]
    def fetch(url,timeout):calls.append(url);return io.BytesIO(body)
    monkeypatch.setattr(slip_assets.urllib.request,'urlopen',fetch)
    first=slip_assets.prepare_clip(tmp_path/'assets')
    assert first['state']=='verified' and len(calls)==1
    assert slip_assets.prepare_clip(tmp_path/'assets')==first and len(calls)==1


def test_asset_io_bad_hash_never_publishes_model(tmp_path,monkeypatch):
    tiny_assets(monkeypatch)
    monkeypatch.setattr(slip_assets.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'wrong'))
    with pytest.raises(ValueError,match='checksum mismatch'):slip_assets.prepare_clip(tmp_path/'assets')
    assert not list((tmp_path/'assets').iterdir())


def test_asset_io_network_failure_never_publishes_model(tmp_path,monkeypatch):
    tiny_assets(monkeypatch)
    def fail(*a,**k):raise OSError('controlled IO failure')
    monkeypatch.setattr(slip_assets.urllib.request,'urlopen',fail)
    with pytest.raises(OSError,match='controlled'):slip_assets.prepare_clip(tmp_path/'assets')
    assert not list((tmp_path/'assets').iterdir())


@pytest.mark.parametrize('kind',['wrong_file','symlink'])
def test_asset_io_refuses_to_replace_existing_user_files(tmp_path,monkeypatch,kind):
    tiny_assets(monkeypatch);root=tmp_path/'assets';root.mkdir();path=root/'ViT-B-16.pt'
    if kind=='wrong_file':path.write_bytes(b'preserve user file')
    else:path.symlink_to(tmp_path/'nonexistent_user_target')
    with pytest.raises(ValueError):slip_assets.prepare_clip(root)
    assert path.is_symlink() if kind=='symlink' else path.read_bytes()==b'preserve user file'


@pytest.mark.parametrize('architecture',['clip_vit_b16','clip_rn50'])
def test_actual_cli_native_demo_run_and_evaluation(tmp_path,architecture):
    def cli(*args):
        result=subprocess.run([sys.executable,'-m','milplus',*map(str,args)],capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        return result
    data=tmp_path/'task';out=tmp_path/'run'
    cli('demo','--slip','--slip-architecture',architecture,'--patients','20','--output',data)
    cli('check','--data',data,'--method','SLIP','--deep')
    cfg=tmp_path/'config.yaml';cfg.write_text('epochs: 1\nthreads: 1\n')
    cli('run','--data',data,'--method','SLIP','--output',out,'--config',cfg)
    result=json.loads(cli('evaluate','--run',out).stdout)
    assert result['n']==4
    request=json.loads((out/'request.json').read_text())
    assert request['method']=='slip_'+architecture+'_source_repaired'


def test_slip_demo_rejects_mixed_task_conditions(tmp_path):
    result=subprocess.run([sys.executable,'-m','milplus','demo','--slip','--survival','--output',str(tmp_path/'mixed')],capture_output=True,text=True)
    assert result.returncode!=0 and not (tmp_path/'mixed').exists()
