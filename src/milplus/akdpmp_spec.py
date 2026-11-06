"""Four complete author recipes, one AKD-PMP method family."""
from .akdpmp_source import COMMIT
from .akdpmp_driver import PROFILES as RECIPES

PROFILES=[{'id':name,'family':'akdpmp','tasks':['classification'],'required_inputs':[
    'patch_features','single_slide_bag','ordered_patch_ids','patient_disjoint_ordered_tasks'],
    'class_count':None,'external_source':'akdpmp','status':'adapted_under_verification',
    'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run',
    'recipe':recipe,'core_mechanisms':['original_attention_KD','selected_frozen_teacher','native_memory_reservoir',
        'native_TCGA_pseudo_bag_or_skin_full_bag','original_proportional_interleaved_replay'],
    'reference_repairs':['patient_disjoint_VAL','single_process_feature_IO','explicit_heldout_evaluation',
        'smooth_SVM_device_closure','numpy2_Inf_constant']} for name,recipe in sorted(RECIPES.items())]
METHOD={'id':'akdpmp','name':'AKD-PMP','group':'I_continual','scope_origin':'user_164_admission_list',
    'source_url':'https://github.com/lxr2/akd-pmp','source_commit':COMMIT,'source_status':'full_protocol_under_verification',
    'license':'no_LICENSE_in_pinned_tree_private_source_loading','status':'adapted_under_verification',
    'profiles':[r['id'] for r in PROFILES],'required_inputs':PROFILES[0]['required_inputs'],
    'paper_reproduction':'not_run','real_wsi_validation':'not_run','blocker':None,
    'requirements_review':'Both complete CLAM-SB and TransMIL source models, original smooth SVM, literal source AKD/PMP training/refill AST; TCGA balanced compressed memory and skin unbalanced full-bag recipes; selected teachers, full replay/state/RNG recovery. All software/wheel/native/global gates required.'}
