import json,os
from pathlib import Path
import pytest
from milplus.macnet_demo import create_demo
from milplus.macnet_data import identity
from milplus.data import Bundle
from milplus.io import digest_file,digest_object

WEIGHTS=Path(os.environ.get('MILPLUS_MACNET_WEIGHTS','/root/MIL/development/macnet_v99_assets_r1/swin_tiny_patch4_window7_224_22k.pth'))


def rebind_annotations(data,rows):
    index=data/'actual_guide_annotations.jsonl';index.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    manifest=data/'macnet_manifest.json';value=json.loads(manifest.read_text());value.pop('contract');value['files'][str(index)]=digest_file(index)
    # Recompute all claimed hashes and duplicated metadata, so rejection must
    # follow the semantic annotation contract, not a stale outer checksum.
    lookup={(r['patch_id'],r['source_slide_id'],r['global_patient_id']):r for r in rows}
    for record in value['rows'].values():
        for guide in record['guides']:
            key=tuple(guide[k] for k in ['patch_id','source_slide_id','global_patient_id'])
            if key in lookup:guide['annotation']=lookup[key]
    value['contract']=digest_object(value);manifest.write_text(json.dumps(value)+'\n')


@pytest.mark.parametrize('field,value,message',[
    ('region_label','benign','tumor-region annotation'),
    ('annotator','','annotator and provenance'),
    ('provenance',' ','annotator and provenance'),
    ('extra_field','unreviewed','tumor-region annotation'),
])
def test_rehashed_bound_annotations_keep_the_same_strict_semantics_as_prepare(tmp_path,field,value,message):
    data=create_demo(tmp_path/'input',WEIGHTS);rows=[json.loads(x) for x in (data/'actual_guide_annotations.jsonl').read_text().splitlines()]
    rows[0][field]=value;rebind_annotations(data,rows)
    with pytest.raises(ValueError,match=message):Bundle(data)


def test_rehashed_annotation_index_rejects_duplicate_source_decisions(tmp_path):
    data=create_demo(tmp_path/'input',WEIGHTS);rows=[json.loads(x) for x in (data/'actual_guide_annotations.jsonl').read_text().splitlines()]
    rows.append(dict(rows[0]));rebind_annotations(data,rows)
    with pytest.raises(ValueError,match='identity is repeated'):Bundle(data)
