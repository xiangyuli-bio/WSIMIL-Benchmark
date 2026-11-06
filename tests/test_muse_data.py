"""Actual native-dimension input preparation and immutable patient/language binding."""
import copy
from pathlib import Path

import pytest

from milplus.data import Bundle
from milplus.muse_demo import create_muse_demo
from milplus.registry import inspect_method

WEIGHTS='/mnt/xiangyuli-private-gpfs/mil_baseline_puls_work/model_assets/vlsa_conch/conch.pth'


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    return create_muse_demo(tmp_path_factory.mktemp('muse_data')/'data',WEIGHTS,patients=6,patches=257)


def test_complete_immutable_source_knowledge_and_patient_safe_split(data):
    bundle=Bundle(data);inputs=bundle.muse
    assert bundle.dim==512 and [len(bundle.records(x)) for x in ['train','val','test']]==[2,2,2]
    assert inputs.meta['descriptions_per_class']==[300,300]
    assert bundle.check(deep=True)['patients']==6
    assert inspect_method('MUSE')['id']=='muse_released_source'


@pytest.mark.parametrize('change',['label','path','feature_bytes','metadata','manifest'])
def test_mutation_of_original_input_is_rejected(data,change):
    bundle=Bundle(data);inputs=bundle.muse;row=copy.deepcopy(bundle.records('train')[0])
    if change=='label':
        row['target']=1-row['target']
        with pytest.raises(ValueError):inputs.load(row)
    elif change=='path':
        row['path']=bundle.records('train')[1]['path']
        with pytest.raises(ValueError):inputs.load(row)
    else:
        name=(inputs.meta['records'][row['slide_id']]['path'] if change=='feature_bytes'
              else inputs.meta['metadata_path'] if change=='metadata' else str(inputs.path))
        path=Path(name);original=path.read_bytes()
        try:
            path.write_bytes(original+b'changed')
            with pytest.raises(ValueError):inputs.load(row)
        finally:path.write_bytes(original)
