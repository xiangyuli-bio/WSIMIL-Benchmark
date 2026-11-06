"""One original QPMIL-VL method family, two explicit author query options."""
from .qpmilvl_source import COMMIT
from .qpmilvl_driver import PROFILES as RECIPES
PROFILES=[{'id':name,'family':'qpmilvl','tasks':['classification'],'required_inputs':[
    'patch_features','single_slide_bag','ordered_patch_ids','patient_disjoint_ordered_tasks',
    'actual_CONCH512_projection_features','original_full_CONCH_weights','clinical_class_descriptions_and_templates'],
    'class_count':None,'external_source':'qpmilvl','status':'adapted_under_verification',
    'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run',
    'recipe':recipe,'core_mechanisms':['differentiable_full110M_CONCH_text_tower','original20key24token5matched_pool',
        'native_minibatch_consensus_and_frequency_penalty','current_task_only_tunable_vectors',
        'CE_matching_class_similarity_objectives','original16bag_update_clip_zero_gradient_Adam',
        'native_plateau_and_early_stop_final_task_model'],
    'reference_repairs':['patient_disjoint_VAL','single_process_feature_IO','no_external_telemetry_or_TSNE',
        'explicit_heldout_evaluation','numpy2_Inf_constant','strict_used_pretrained_checkpoint_keys']}
    for name,recipe in sorted(RECIPES.items())]
METHOD={'id':'qpmilvl','name':'QPMIL-VL','group':'I_continual','scope_origin':'user_164_admission_list',
    'source_url':'https://github.com/can-can-ya/QPMIL-VL','source_commit':COMMIT,
    'source_status':'full_protocol_under_verification','license':'CC-BY-NC-ND-4.0_private_source_loading_no_weights_redistribution',
    'status':'adapted_under_verification','profiles':[r['id'] for r in PROFILES],
    'required_inputs':PROFILES[0]['required_inputs'],'paper_reproduction':'not_run','real_wsi_validation':'not_run','blocker':None,
    'requirements_review':'Full original CoCa construction and RNG ordering, real110M12layer/768hidden/12head CONCH text tower,20/24/5 prototype pool, native16bag losses/update, class ensembles, full continual frequency/optimizer/scheduler/gradient/RNG recovery. Explicit source max/mean query variants; all software/wheel/native/global gates required.'}
