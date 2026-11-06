"""Published model identity and missing/wrong-asset rejection, without fake weights."""
import pytest

from milplus.feather_assets import native_lock,prepare


@pytest.mark.parametrize('encoder',['conch_v15','uni','uni_v2'])
def test_missing_original_weight_file_cannot_produce_a_manifest(tmp_path,encoder):
    output = tmp_path/'binding.json'
    with pytest.raises(FileNotFoundError,match='original FEATHER24K'):
        prepare(encoder,tmp_path/'missing.bin',output)
    assert not output.exists()


@pytest.mark.parametrize('encoder',['conch_v15','uni','uni_v2'])
def test_unrelated_or_random_checkpoint_is_rejected_before_tensor_loading(tmp_path,encoder):
    checkpoint = tmp_path/'pytorch_model.bin';checkpoint.write_bytes(b'not the original pretrained model')
    output = tmp_path/'binding.json'
    with pytest.raises(ValueError,match='original pretrained FEATHER24K variant'):
        prepare(encoder,checkpoint,output)
    assert not output.exists()


def test_native_encoder_identities_distinguish_feature_spaces_and_release_sizes():
    lock = native_lock()
    assert {key:value['native_dim'] for key,value in lock['checkpoints'].items()} == {
        'conch_v15':768,'uni':1024,'uni_v2':1536}
    assert lock['pretraining_release_scale_label'] == '24K'
    assert lock['exact_pretraining_cohort_size_verified'] is False
    hashes = [row['files']['model.safetensors']['sha256'] for row in lock['checkpoints'].values()]
    assert len(set(hashes)) == 3
    assert lock['other_CONCHv15_file_identity_not_proven_feature_equivalent']['same_dim_does_not_establish_interchangeability']
