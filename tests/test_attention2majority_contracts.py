"""Input isolation and patient-safe Appendix-I selection; native graphs tested separately."""
import copy
import hashlib

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from milplus.attention2majority_data import AttentionInputs
from milplus.attention2majority_ica import select_large_tumor_train_pool
from milplus.data import Bundle
from milplus.io import read_json,write_json


@pytest.fixture
def raw(tmp_path,monkeypatch):
    # Metadata unit tests use a visibly artificial weight identity. The native
    # graph validator independently requires the unchanged official Keras MD5.
    from milplus import attention2majority_source
    weights=tmp_path/'metadata_unit_test_not_pretrained.h5';weights.write_bytes(b'not real network weights')
    monkeypatch.setattr(attention2majority_source,'RESNET_MD5',hashlib.md5(weights.read_bytes()).hexdigest())
    rows=[];splits=[];indices=[];pool=[]
    for subset in ['train','val','test']:
        for target in range(2):
            name=f'{subset}_{target}';path=tmp_path/(name+'.h5')
            with h5py.File(path,'w') as f:
                f.create_dataset('pixels',data=np.full((2,224,224,3),target*255,dtype=np.uint8))
                f.create_dataset('patch_ids',data=np.array([name+'_0',name+'_1'],dtype=h5py.string_dtype()))
            rows.append({'slide_id':name,'patient_id':name,'global_patient_id':'unit:'+name,'label':str(target)})
            splits.append({'patient_id':name,'subset':subset});indices.append({'slide_id':name,'path':path.name})
            if subset=='train':pool.append(name)
    pd.DataFrame(rows).to_csv(tmp_path/'slides.csv',index=False)
    pd.DataFrame(splits).to_csv(tmp_path/'split.csv',index=False)
    pd.DataFrame(indices).to_csv(tmp_path/'images.csv',index=False)
    write_json(tmp_path/'review.json',{'kind':'reviewed_large_tumor_TRAIN_pool','selected_train_slide_ids':pool,
        'heldout_labels_used':False,'review_provenance':'Synthetic metadata-unit-test pool'})
    schema={'schema_version':1,'kind':'attention2majority_uint8_rgb_hdf5','mode':'camelyon_binary',
        'image_provenance':'synthetic metadata test','patch_provenance':'explicit original unit patch identities',
        'weights':str(weights),'discriminator_train_pool':pool,'discriminator_pool_evidence':'review.json'}
    c={'schema_version':1,'task_id':'attention_unit','task_type':'classification','classes':['0','1'],
        'defaults':{'fm':'rgb','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'rgb':{'dimension':3,'index':'images.csv','attention2majority_raw':schema}}}
    (tmp_path/'task.yaml').write_text(yaml.safe_dump(c));return tmp_path,c


def save(raw):
    root,c=raw;(root/'task.yaml').write_text(yaml.safe_dump(c));return root


def test_raw_inventory_rejects_frozen_feature_interpretation(raw):
    root,_=raw;b=Bundle(root);inventory=b.check(deep=True)
    assert inventory['slides']==inventory['patients']==6
    assert all(r['instances']==2 and r['sha256'] for r in inventory['features'])
    with pytest.raises(ValueError,match='complete native Keras'):b.load(b.records('train')[0])
    with pytest.raises(ValueError,match='frozen spatial'):b.check(deep=True,require_coords=True)


def test_heldout_discriminator_pool_is_rejected(raw):
    _,c=raw;c['encoders']['rgb']['attention2majority_raw']['discriminator_train_pool']=['train_0','test_1']
    with pytest.raises(ValueError,match='exclusively from TRAIN'):AttentionInputs(save(raw))


def test_pool_evidence_must_bind_actual_selection_and_heldout_policy(raw):
    root,_=raw;p=root/'review.json';proof=read_json(p)
    proof['heldout_labels_used']=True;write_json(p,proof)
    with pytest.raises(ValueError,match='actual TRAIN identities'):AttentionInputs(root)
    proof['heldout_labels_used']=False;proof['selected_train_slide_ids'].reverse();write_json(p,proof)
    with pytest.raises(ValueError,match='actual TRAIN identities'):AttentionInputs(root)


def test_unselected_binding_is_restricted_to_initial_train_features(raw):
    _,c=raw;s=c['encoders']['rgb']['attention2majority_raw'];s.pop('discriminator_train_pool');s.pop('discriminator_pool_evidence')
    root=save(raw);initial=AttentionInputs(root,require_pool=False)
    assert initial.pool==[] and initial.pool_evidence['kind']=='unselected_initial_TRAIN_features_only'
    with pytest.raises(ValueError,match='explicit positive/negative'):AttentionInputs(root)


def test_changed_raw_bytes_and_patch_id_alignment_are_rejected(raw):
    root,_=raw;inputs=AttentionInputs(root)
    with h5py.File(root/'train_0.h5','r+') as f:f['pixels'][0,0,0,0]=123
    with pytest.raises(ValueError,match='original RGB'):inputs.check(deep=True)
    with h5py.File(root/'train_0.h5','r+') as f:f['patch_ids'][1]='train_0_0'
    with pytest.raises(ValueError,match='unique aligned'):AttentionInputs(root)


def test_patient_leakage_is_rejected_before_any_image_updates(raw):
    root,_=raw;p=root/'slides.csv';rows=pd.read_csv(p,dtype=str)
    rows.loc[rows.slide_id=='test_0','global_patient_id']='unit:train_0';rows.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Global patient identity crosses splits'):AttentionInputs(root)


def test_registered_family_routes_explicit_native_conditions():
    from types import SimpleNamespace
    from milplus.registry import inspect_method,resolve_for_bundle
    assert inspect_method('Attention2majority')['id']=='attention2majority_camelyon_source_repaired'
    assert resolve_for_bundle('Attention2majority',SimpleNamespace(classes=['low','mid','high']))['id']=='attention2majority_kidney_paper_reference'
    assert inspect_method('attention2majority_camelyon_paper_reference')['class_count']==2


def test_other_mil_cannot_interpret_native_rgb_as_frozen_features(raw):
    from milplus.registry import check_compatibility,inspect_method
    with pytest.raises(ValueError,match='cannot replace frozen FM'):
        check_compatibility(inspect_method('abmil'),Bundle(raw[0]))


def test_public_training_rejects_hidden_instance_reduction_before_launch(raw):
    from milplus.engine import train
    root,_=raw
    with pytest.raises(ValueError,match='preserves native original stage optimizers'):
        train(root,'Attention2majority',root/'forbidden_run',config={'max_instances':8})
    assert not (root/'forbidden_run').exists()


def test_appendix_prepare_rejects_heldout_template_before_native_extraction(raw):
    from milplus.attention2majority_prepare import prepare
    root,_=raw
    with pytest.raises(ValueError,match='TRAIN-positive template'):
        prepare(root,root/'forbidden_prepare','test_1','train_0','unmixing_rows')
    assert not (root/'forbidden_prepare').exists()


def ica_case():
    rng=np.random.default_rng(781);basis=np.linalg.qr(rng.normal(size=(1024,2)))[0]
    template=(rng.laplace(size=(128,2))@basis.T).astype(np.float32)
    rows=[{'slide_id':'normal','target':0,'subset':'train'}]
    features={'normal':rng.normal(scale=.01,size=(51,1024)).astype(np.float32)}
    for i in range(15):
        name=f'tumor_{i:02d}';rows.append({'slide_id':name,'target':1,'subset':'train'});features[name]=template.copy()
    rows.append({'slide_id':'heldout','target':1,'subset':'test'})
    return rows,features


@pytest.mark.parametrize('space',['unmixing_rows','mixing_columns'])
def test_complete_native_ica_pool_retains_train_identity_and_explicit_choices(space):
    rows,features=ica_case()
    proof=select_large_tumor_train_pool(features,rows,'tumor_00','normal',component_space=space)
    assert len(proof['selected_positive_slide_ids'])==12
    assert len(proof['all_TRAIN_tumor_fractions'])==15 and 'heldout' not in proof['all_TRAIN_tumor_fractions']
    assert proof['heldout_labels_used'] is False and proof['component_space']==space
    assert np.asarray(proof['components']).shape==(2,1024) and len(proof['mean'])==1024
    assert all(proof['all_TRAIN_tumor_fractions'][s]>=.1 for s in proof['selected_positive_slide_ids'])


@pytest.mark.parametrize('defect',['heldout_template','heldout_features','missing_train','missing_dimension'])
def test_ica_rejects_leakage_incomplete_or_non_native_inputs(defect):
    rows,features=ica_case();template='tumor_00'
    if defect=='heldout_template':template='heldout'
    if defect=='heldout_features':features['heldout']=features['tumor_00']
    if defect=='missing_train':features.pop('tumor_01')
    if defect=='missing_dimension':features['tumor_01']=features['tumor_01'][:,:512]
    with pytest.raises(ValueError):
        select_large_tumor_train_pool(features,rows,template,'normal',component_space='unmixing_rows')
