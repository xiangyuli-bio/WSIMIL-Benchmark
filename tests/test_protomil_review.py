import copy
from types import SimpleNamespace
import pytest
from milplus.protomil_stages import read_review
from milplus.io import write_json


def setup(tmp_path,synthetic=True):
    assets=SimpleNamespace(review_file=str(tmp_path/'review.json'),bundle=SimpleNamespace(config={'synthetic':synthetic}))
    bank={'sae_state_sha256':'a'*64,'bank_sha256':'b'*64,'active_concepts':[5,103,2047]}
    review={'schema_version':1,'sae_state_sha256':bank['sae_state_sha256'],'concept_bank_sha256':bank['bank_sha256'],
        'reviewer':'named_reviewer','provenance':'Declared supplied inspection; unit fixture only','removed_concepts':[103],'concept_names':{'103':'supplied description'}}
    return assets,bank,review


def test_intervention_waits_for_actual_review_while_nonintervened_condition_is_explicit(tmp_path):
    assets,bank,_=setup(tmp_path)
    assert read_review(assets,bank,True) is None
    assert read_review(assets,bank,False)=={'removed_concepts':[],'review_sha256':None,'review_scope':'non-intervened original concept condition'}


@pytest.mark.parametrize('field,value',[('sae_state_sha256','c'*64),('concept_bank_sha256','c'*64),('removed_concepts',[9]),('removed_concepts',[103,103]),('concept_names',{}),('reviewer',''),('provenance','')])
def test_wrong_SAE_probing_bank_or_unreviewed_concepts_are_rejected(tmp_path,field,value):
    assets,bank,review=setup(tmp_path);review[field]=value;write_json(assets.review_file,review)
    with pytest.raises(ValueError):read_review(assets,bank,True)


def test_synthetic_review_never_admitted_for_unmarked_clinical_data(tmp_path):
    assets,bank,review=setup(tmp_path,False);review['reviewer']='synthetic_validation_only';write_json(assets.review_file,review)
    with pytest.raises(ValueError,match='Synthetic concept'):read_review(assets,bank,True)


def test_actual_supplied_review_identity_and_byte_hash_are_exported(tmp_path):
    assets,bank,review=setup(tmp_path);write_json(assets.review_file,review);result=read_review(assets,bank,True)
    assert result['removed_concepts']==[103] and len(result['review_sha256'])==64 and result['review_path']==assets.review_file
