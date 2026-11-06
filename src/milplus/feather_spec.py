"""Original-pretrained FEATHER code profiles, without a research acceptance claim."""

EXECUTION_CONDITIONS = {
    f'feather_{encoder}_{mode}_reference': {'encoder': encoder, 'mode': mode}
    for encoder in ['conch_v15', 'uni', 'uni_v2']
    for mode in ['finetune_val20', 'finetune_no_val10', 'frozen_knn20']
}
DEFAULT_PROFILE = 'feather_uni_finetune_val20_reference'
REQUIRED_INPUTS = ['native_CONCHv15_UNI_or_UNIv2_features',
                   'authorized_original_FEATHER24K_weights', 'patient_disjoint_splits']
PROFILES = [{
    'id': identifier, 'family': 'feather',
    'recipe': f"Original FEATHER24K512/384 backbone, {condition['encoder']}, {condition['mode']}; "
              'explicit downstream training/selection/KNN reference choices',
    'tasks': ['classification', 'ordinal'], 'required_inputs': REQUIRED_INPUTS,
    'class_count': None, 'external_source': 'feather', **condition,
    'status': 'adapted_under_verification', 'formal_admission': False,
    'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run',
} for identifier, condition in EXECUTION_CONDITIONS.items()]
METHOD = {
    'id': 'feather', 'name': 'FEATHER / MIL-Lab models', 'group': 'J_pretrained_slide',
    'scope_origin': 'user_164_admission_list',
    'source_url': 'https://github.com/mahmoodlab/MIL-Lab',
    'source_status': 'full_original_pretrained_adapter_code_integrated_under_verification',
    'source_commit': 'dfc3c17ca6876a8f9bd33a9162c40f2c22631d9d',
    'license': 'CC-BY-NC-ND-4.0_private_source_loading_no_redistribution',
    'profiles': list(EXECUTION_CONDITIONS), 'status': 'adapted_under_verification',
    'default_execution_profile': DEFAULT_PROFILE, 'required_inputs': REQUIRED_INPUTS,
    'formal_admission': False, 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run',
    'requirements_review': 'Full original512/384 model, checked local pretrained loading, '
        'native feature/patient binding, isolated20epoch-with-VAL and10epoch-without-VAL '
        'finetuning, frozen512D KNN20, recovery and evaluation are publicly integrated. '
        'Family execution uses the prepared encoder with finetune_val20. '
        'Unpublished trainer/selection/AMP/sampling/KNN details remain explicit references. '
        'No random architecture is presented as pretrained FEATHER.',
    'prerequisites': ['Run setup and prepare-feather-assets with the authorized original24K checkpoint.',
                      'Bind its matching native visual features and patients using prepare-feather.',
                      'Use the separate feather runtime dependencies; frozen KNN requires at least20 TRAIN slides.'],
    'validation_blockers': ['Actual original-pretrained loading and complete source/wheel/native-budget validation remain pending.'],
    'blocker': None,
}
