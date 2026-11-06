import h5py
import numpy as np
import pytest
import yaml

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.icmil_context_data import ContextInputs
from milplus.icmil_context_prepare import prepare
from milplus.io import read_json


def fixture(root,dim=32):
    data=create_demo(root,patients=20,dimension=dim)
    p=data/'task.yaml';c=yaml.safe_load(p.read_text());c['evaluation_unit']='slide';p.write_text(yaml.safe_dump(c))
    return data


def test_train_only_all_instance_PCA_matches_independent_SVD_and_keeps_real_rows(tmp_path):
    data=fixture(tmp_path/'data');source=Bundle(data)
    prepared=prepare(data,tmp_path/'prepared',10,'train_pca25_reference',seed=7)
    actual=ContextInputs(prepared);proof=read_json(prepared/'preparation.json')
    all_train=np.concatenate([source.load(r)[0].numpy() for r in source.records('train')]).astype(np.float64)
    _,s,v=np.linalg.svd(all_train-all_train.mean(0),full_matrices=False)
    with np.load(prepared/'projection.npz',allow_pickle=False) as f:
        assert np.allclose(f['mean'],all_train.mean(0),rtol=0,atol=1e-14)
        assert np.allclose(f['components'].T@f['components'],v[:25].T@v[:25],rtol=1e-10,atol=1e-10)
        assert np.allclose(f['eigenvalues'],s[:25]**2/(len(all_train)-1),rtol=1e-12,atol=1e-12)
    assert proof['pca_train_instances']==len(all_train)
    assert proof['fit_subsets']==['train'] and proof['heldout_labels_used'] is False
    assert proof['historical_author_PCA_reconstruction_claimed'] is False
    assert actual.bundle.dim==25 and actual.bag_size==10
    original={r['slide_id']:r for r in source.rows.to_dict('records')}
    for r in actual.records:
        x,c=actual.bundle.load(r);old_x,old_c=source.load(original[r['slide_id']])
        with h5py.File(r['path']) as f:indices=f['original_row_indices'][:]
        assert np.array_equal(c.numpy(),old_c.numpy()[indices])
        assert len(set(indices))==10


def test_heldout_features_and_labels_cannot_change_PCA_or_TRAIN_support(tmp_path):
    data=fixture(tmp_path/'data');first=prepare(data,tmp_path/'first',10,'train_pca25_reference',seed=8)
    source=Bundle(data)
    for r in source.rows[source.rows.subset!='train'].to_dict('records'):
        with h5py.File(source.root/r['path'],'r+') as f:f['features'][:]+=1000
    slides=source.rows[['slide_id','patient_id','label']].copy()
    mask=source.rows.subset!='train';slides.loc[mask,'label']=slides.loc[mask,'label'].map({'class_0':'class_1','class_1':'class_0'})
    slides.to_csv(data/'slides.csv',index=False)
    second=prepare(data,tmp_path/'second',10,'train_pca25_reference',seed=8)
    with np.load(first/'projection.npz') as a,np.load(second/'projection.npz') as b:
        for k in a.files:assert np.array_equal(a[k],b[k])
    assert np.array_equal(ContextInputs(first).dense('train')[0],ContextInputs(second).dense('train')[0])


@pytest.mark.parametrize('dimension,projection,budget',[(32,'native',10),(16,'train_pca25_reference',10),(16,'native',100)])
def test_no_silent_feature_truncation_padding_or_replacement(tmp_path,dimension,projection,budget):
    data=fixture(tmp_path/'data',dimension)
    with pytest.raises(ValueError):prepare(data,tmp_path/'prepared',budget,projection)


def test_unprepared_variable_length_bags_rejected(tmp_path):
    with pytest.raises(ValueError,match='fixed-size'):ContextInputs(fixture(tmp_path/'data',16))


def test_prepared_feature_tampering_is_rejected(tmp_path):
    p=prepare(fixture(tmp_path/'data',16),tmp_path/'prepared',10,'native')
    asset=next((p/'features').glob('*.h5'))
    with h5py.File(asset,'r+') as f:f['features'][0,0]+=1
    with pytest.raises(ValueError):ContextInputs(p)


def test_patient_leakage_rejected_before_any_projection(tmp_path):
    data=fixture(tmp_path/'data');b=Bundle(data);table=b.rows[['slide_id','patient_id','label']].copy()
    table['global_patient_id']='same_patient';table.to_csv(data/'slides.csv',index=False)
    with pytest.raises(ValueError,match='crosses splits'):prepare(data,tmp_path/'prepared',10,'train_pca25_reference')
