"""Container SHA cannot secure a mutable external H5 payload; refuse it."""
import h5py
import numpy as np
import pytest

from milplus.data import load_array


@pytest.mark.parametrize('kind',['external_link','soft_alias','virtual_dataset','external_storage'])
def test_unregistered_h5_backing_storage_is_never_hidden_behind_container_sha(tmp_path,kind):
    actual=tmp_path/'actual.h5';container=tmp_path/'features.h5'
    value=np.arange(12,dtype=np.float32).reshape(3,4)
    with h5py.File(actual,'w')as f:f['features']=value
    with h5py.File(container,'w')as f:
        if kind=='external_link':f['features']=h5py.ExternalLink(str(actual),'features')
        elif kind=='soft_alias':
            f['actual']=value;f['features']=h5py.SoftLink('/actual')
        elif kind=='virtual_dataset':
            layout=h5py.VirtualLayout(shape=value.shape,dtype=value.dtype)
            layout[:]=h5py.VirtualSource(str(actual),'features',shape=value.shape)
            f.create_virtual_dataset('features',layout)
        else:
            dataset=f.create_dataset('features',shape=value.shape,dtype=value.dtype,
                external=[(str(tmp_path/'unregistered.bin'),0,h5py.h5f.UNLIMITED)])
            dataset[:]=value
    with pytest.raises(ValueError,match='materialized before feature hashing'):load_array(container)


def test_materialized_h5_preserves_actual_tensor_bytes(tmp_path):
    path=tmp_path/'features.h5';value=np.arange(12,dtype=np.float32).reshape(3,4)
    with h5py.File(path,'w')as f:f['features']=value
    assert np.array_equal(load_array(path),value)
