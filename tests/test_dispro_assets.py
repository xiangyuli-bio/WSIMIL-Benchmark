"""Asset IO controls use explicit tiny network responses; never model evidence."""
import hashlib
import io

import pytest

from milplus import dispro_assets as assets
from milplus.io import read_json,write_json


@pytest.fixture
def public_response(monkeypatch):
    bodies={'config.json':b'fixture-config','vocab.txt':b'fixture-vocab','pytorch_model.bin':b'fixture-bytes'}
    monkeypatch.setattr(assets,'ASSET_SHA',{k:hashlib.sha256(v).hexdigest() for k,v in bodies.items()})
    requests=[]
    def open_url(request,timeout):
        requests.append(request.full_url)
        return io.BytesIO(bodies[request.full_url.rsplit('/',1)[-1]])
    monkeypatch.setattr(assets.urllib.request,'urlopen',open_url)
    return bodies,requests


def test_exact_revision_urls_hashes_atomic_files_and_idempotent_verified_reuse(tmp_path,public_response):
    bodies,requests=public_response
    first=assets.prepare_biobert(tmp_path/'assets')
    assert first['state']=='verified' and len(requests)==3
    assert all('/'+assets.REVISION+'/' in u for u in requests)
    for name,body in bodies.items():assert (tmp_path/'assets'/name).read_bytes()==body
    assert assets.prepare_biobert(tmp_path/'assets')==first and len(requests)==3
    assert not list((tmp_path/'assets').glob('*.download'))


def test_bad_existing_file_never_overwritten_and_no_false_complete_manifest(tmp_path,public_response):
    root=tmp_path/'assets';root.mkdir();(root/'config.json').write_bytes(b'private-existing-file')
    with pytest.raises(ValueError,match='overwrite existing'):assets.prepare_biobert(root)
    assert (root/'config.json').read_bytes()==b'private-existing-file'
    assert not (root/'manifest.json').exists()


def test_wrong_download_not_published_as_pretrained_weights(tmp_path,public_response,monkeypatch):
    monkeypatch.setattr(assets.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'incorrect-server-body'))
    with pytest.raises(ValueError,match='checksum mismatch'):assets.prepare_biobert(tmp_path/'assets')
    assert not list((tmp_path/'assets').iterdir())


def test_wrong_manifest_revision_refused_before_any_network_IO(tmp_path,public_response):
    root=tmp_path/'assets';root.mkdir()
    write_json(root/'manifest.json',{'repo':'another/model','revision':'not-the-reviewed-commit'})
    with pytest.raises(ValueError,match='another asset revision'):assets.prepare_biobert(root)
    assert read_json(root/'manifest.json')['repo']=='another/model' and public_response[1]==[]


def test_existing_symlink_is_not_adopted_or_followed(tmp_path,public_response):
    bodies,_=public_response;source=tmp_path/'source';source.write_bytes(bodies['config.json'])
    root=tmp_path/'assets';root.mkdir();(root/'config.json').symlink_to(source)
    with pytest.raises(ValueError,match='overwrite existing'):assets.prepare_biobert(root)
    assert source.read_bytes()==bodies['config.json']
