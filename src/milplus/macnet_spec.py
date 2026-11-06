"""One original raw-image MacNet family with six disclosed source/repair profiles."""
from .macnet_driver import PROFILES as RECIPES
from .macnet_data import CONDITIONS
from .macnet_source import COMMIT,SWIN22K_SHA256
PROFILES=[{'id':name,'family':'macnet','tasks':['classification'],
    'required_inputs':['raw_RGB','single_slide_bag','ordered_patch_ids','patient_disjoint_splits','actual_annotated_tumor_guides','actual_Swin_Tiny_ImageNet22K_initialization'],
    'class_count':CONDITIONS[cfg['condition']]['classes'],'external_source':'macnet','status':'adapted_under_verification',
    'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run','requires_cuda':True,'recipe':cfg,
    'core_mechanisms':['full_original27499482parameter_Swin_feature_encoder_at96RGB',
        'full_original768D_GFK_PCA_GSVD_and_kernel_sqrt','three_original_manifold_adaptive_clusters',
        'actual_5_81_or64_prior_guides_and_proxy_weighted_original_bag_features','CE_plus_minimum_minus_maximum_manifold_distance',
        'literal_author_RMSprop_recreated_each_epoch_with_original_mutable_groups','native100epoch_training_and_source_validation_selection',
        'complete_cached_adaptive_matrix_model_gradient_optimizer_scaler_RNG_recovery'],
    'reference_repairs':['source_tabs_expanded8','wrong_source_import_namespace_isolated','literal_cuda0_to_selected_current_device',
        'single_device_full_encoder_instead_of4GPU_DataParallel','source_unregistered_cached_matrix_explicitly_checkpointed',
        'TEST_deferred_to_explicit_evaluation','first_valid_VAL_selection_initialization']+(['adaptive_diagonal_refreshed_each_forward_explicit_repair'] if cfg['adaptive']=='refresh_repaired' else []),
    'input_supervision':'additional_TRAIN_guide_bank_or_explicit_per_query_region_annotations',
    'source_caveats':['released_metric_uses_exp_of_matrix_elements_for_cluster_assignment_and_signed_diagonal_for_proxy_distance',
        'refresh_repaired_does_not_claim_to_reconstruct_the_paper_PSD_metric','source_mutated_RMSprop_group_LR_semantics_preserved_by_default'],
    'pretrained_weight_sha256':SWIN22K_SHA256}
    for name,cfg in sorted(RECIPES.items())]
METHOD={'id':'macnet','name':'MacNet','group':'E_encoder_training','scope_origin':'user_164_admission_list',
    'source_url':'https://github.com/Prince-Lee-PathAI/MacNet','source_commit':COMMIT,'source_status':'complete_raw_protocol_under_verification',
    'license':'no_LICENSE_in_pinned_tree_private_source_loading_no_redistribution','status':'adapted_under_verification',
    'profiles':[r['id'] for r in PROFILES],'required_inputs':PROFILES[0]['required_inputs'],'paper_reproduction':'not_run',
    'real_wsi_validation':'not_run','blocker':None,
    'requirements_review':'Complete actual pretrained22K Swin-Tiny,768D GFK/GSVD/kernel sqrt, adaptive three-cluster proxy weighting and CE+min-max loss; original reset-RMSprop epochs and source validation selection. Public5/LSCC81/CSCC64 guide layouts; separate source-cached and explicit refresh-diagonal repair conditions. Actual cached matrix checkpointed, no raw encoder/guide substitution. All CLI/wheel/native100epoch/global gates still required; no real WSI/paper-score reproduction claim.'}
