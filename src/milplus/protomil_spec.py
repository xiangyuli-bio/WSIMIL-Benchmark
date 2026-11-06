"""Original MICCAI2025 concept ProtoMIL, four disclosed training/review conditions."""
from .protomil_source import COMMIT,DTN_COMMIT
from .protomil_driver import PROFILES as RECIPES
PROFILES=[{'id':name,'family':'protomil','tasks':['classification'],'required_inputs':[
    'patch_features','single_slide_bag','ordered_patch_ids','actual_CONCH512_projection_features',
    'TRAIN_only_joint_SAE_populations','TRAIN_region_annotated_RGB256_probes']+(['actual_SAE_bound_concept_review'] if recipe['intervened'] else []),
    'class_count':2,'external_source':'protomil','status':'adapted_under_verification',
    'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run',
    'recipe':recipe,'core_mechanisms':['full512_to2048_sparse_autoencoder_with_gradient_projection_hook_and_unit_norm_decoder',
        'original4096_batch_AdamWithReset_resampler_and_activation_L1','original80_20_SAE_piece_split_with_only_declared_TRAIN_patients',
        'actual_normal_tumor_region_probes_and_full_group_forward','original_top10_per_concept_real_RGB_inspection',
        'full2048_concept_gated_MIL_CE_plus_mean_classifier_L1','complete_SAE_and_MIL_weights_gradients_Adam_RNG_recovery'],
    'reference_repairs':['explicit_reviewed_original_SAE_dependency_commit_'+DTN_COMMIT,'patient_disjoint_partitions',
        'independent_TRAIN_VAL_generators','TEST_deferred_to_explicit_evaluation','local_logging_no_external_telemetry',
        'lossless_float32_concept_cache','paper_first_valid_VAL_selection_or_disclosed_released_last_epoch']}
    for name,recipe in sorted(RECIPES.items())]
METHOD={'id':'protomil','name':'ProtoMIL','group':'H_extra_supervision','scope_origin':'user_164_admission_list',
    'source_url':'https://github.com/ss-sun/ProtoMIL','source_commit':COMMIT,
    'source_status':'full_protocol_under_verification','license':'no_LICENSE_in_pinned_tree_private_source_loading_no_redistribution',
    'status':'adapted_under_verification','profiles':[r['id'] for r in PROFILES],
    'required_inputs':sorted({k for p in PROFILES for k in p['required_inputs']}),
    'paper_reproduction':'not_run','real_wsi_validation':'not_run','blocker':None,
    'requirements_review':'Full original512D CONCH /2048D SAE with actual projection hooks and AdamWithReset/resampler/metrics; joint TRAIN-only source piece split; independently annotated normal/tumor RGB256 probes; original concept activation inspection; actual SAE/probe-bank-bound user review for intervention; full gated concept MIL with CE and mean classifier L1. Separate paper60/200 and released-default200/1 epoch profiles. All software/wheel/native/global gates remain required; synthetic checks do not reproduce clinical scores.'}
