"""Separate literal-source and explicitly reconstructed Attention2majority conditions."""
from .attention2majority_source import COMMIT

REQUIRED=['native_RGB_patch_bags','original_Keras_ImageNet_ResNet50_weights',
    'TRAIN_only_discriminator_pool','aligned_original_patch_ids','single_slide_bag']
PROFILES=[]
for name,classes,recipe in [
    ('attention2majority_camelyon_source_repaired',2,
     'Complete actual Keras ResNet50 conv4/GAP1024 TRAIN weak-patch training, signedint8/Caffe/extra255 source preprocessing, '
     'Adam.001/150epochs/batch104, literal top/bottom30percent including source -0; actual binary1024/512 gatedMIL, '
     'class-weightedBCE+allL2, Adam.0002/inverse-time/150max/TRAINaccuracy-patience10 and selected-last; syntax/orchestration repair explicit'),
    ('attention2majority_camelyon_paper_reference',2,
     'Complete original Keras RGB224 ResNet50 conv4/GAP1024 task adaptation, named uint8/Caffe/Adam.0002 reference '
     '150max/TRAINaccuracy-patience10, actual all-patch extraction and top/bottom30percent with named zero-tail repair; '
     'literal binary1024/512 gatedMIL/class-weightedBCE+L2/Adam/inverse-time/source stopping and selected-last'),
    ('attention2majority_kidney_paper_reference',3,
     'Complete original Keras RGB128 ResNet50 conv4/GAP1024 three-grade task adaptation, native200sample/slide/batch104, '
     '150max/TRAINaccuracy-patience10, all-patch majority/confidence queue K2000; complete commented author three independent1024/512heads, '
     'concat3072/three logits/class-weighted categoricalCE+L2 and source Adam/inverse-time/selected-last')]:
    PROFILES.append({'id':name,'family':'attention2majority','recipe':recipe,'tasks':['classification'],
        'required_inputs':REQUIRED,'class_count':classes,'external_source':'attention2majority',
        'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run',
        'status':'implemented_unvalidated','backend':'attention2majority_native_tensorflow'})
METHOD={'id':'attention2majority','name':'Attention2majority','group':'E_encoder_training',
    'scope_origin':'user_164_admission_list','source_url':'https://github.com/JoeSu666/Attention2majority',
    'source_status':'git_commit_locked_explicit_orchestration_reconstruction','source_commit':COMMIT,
    'license':'AGPL-3.0_private_pinned_author_source_loading','profiles':[p['id'] for p in PROFILES],
    'status':'adapted','required_inputs':REQUIRED,'paper_reproduction':'not_run','real_wsi_validation':'not_run',
    'requirements_review':'Complete original TensorFlow CNN/filter/MIL stages, original source graph/selector/pixel and every gradient/Adam oracles; '
        'Appendix-I actual initial ImageNet/ICA TRAIN selection, complete checkpoint/input/runtime contracts. Source syntax, '
        'signed-pixel/extra-scale/learning-rate/zero-tail and missing orchestration choices explicitly disclosed. '
        'Native-budget, independent-wheel and whole-release acceptance remain separate; real WSI/paper scores not run.',
    'blocker':None}
