"""Public full DSPA2025 reference; the distinct DSPA-U variant is NOT implemented."""

EXECUTION_CONDITIONS = {
    'dspamil_2025_released_mask_reference': 'released_zero_logits',
    'dspamil_2025_paper_mask_reference': 'paper_post_softmax_no_renormalization_reference',
}
DEFAULT_PROFILE = 'dspamil_2025_released_mask_reference'
REQUIRED_INPUTS = ['native_UNI1024_patch_features', 'patient_disjoint_splits',
                   'TRAIN_only_measured_Gleason_pattern_targets',
                   'pattern_and_training_grade_expert_concepts', 'explicit_text_to_UNI_alignment']
PROFILES = [{
    'id': identifier, 'family': 'dspamil',
    'recipe': f'Complete162690072parameter PAT/Teacher/PTG/Student DSPA2025, {mask}; '
              'explicit equation/head/selection references, full50epoch AdamW recipe',
    'tasks': ['ordinal'], 'required_inputs': REQUIRED_INPUTS, 'class_count': 6,
    'external_source': 'dspamil', 'mask_convention': mask,
    'status': 'adapted_under_verification', 'formal_admission': False,
    'DSPA_U_implemented': False, 'original_expert_assets_verified': False,
    'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run',
} for identifier, mask in EXECUTION_CONDITIONS.items()]
METHOD = {
    'id': 'dspamil', 'name': 'DSPA-MIL / DSPA-U-MIL', 'group': 'H_extra_supervision',
    'scope_origin': 'user_164_admission_list',
    'source_url': 'https://github.com/AlexNmSED/DSPA-MIL',
    'source_status': 'incomplete_author_source_with_full2025_reference_adapter_integrated',
    'source_commit': '75d95d410374771a3eba4bd84fdfadc06acf0eed', 'license': 'MIT',
    'profiles': list(EXECUTION_CONDITIONS), 'status': 'adapted_under_verification',
    'default_execution_profile': DEFAULT_PROFILE, 'required_inputs': REQUIRED_INPUTS,
    'supported_variants': ['DSPA-MIL2025_explicit_reference'],
    'unsupported_variants': ['DSPA-U-MIL'], 'DSPA_U_implemented': False,
    'original_expert_assets_verified': False, 'formal_admission': False,
    'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run',
    'requirements_review': 'Complete151120896parameter PAT plus full-width dual heads '
        '(162690072 total), native input binding, TRAIN-only measured pattern supervision, '
        'all published2025 losses,50epoch reference trainer, full-state checkpoints, '
        'isolated subprocess and Student-only evaluation are publicly integrated. '
        'Unreleased normalization, KL/detach, head and training choices must be explicitly '
        'configured. This is not the incomplete literal author trainer or a DSPA-U implementation.',
    'prerequisites': ['Prepare actual UNI features, six ordered grades, measured TRAIN patterns and justified concept/alignment assets.',
                      'Run prepare-dspamil and supply configs/dspamil_2025_full_reference.yaml with reviewed provenance.',
                      'Use the dspamil Torch2.5.1 runtime; inference uses only the Student path.'],
    'validation_blockers': ['Original expert assets/alignment and complete full-model source/wheel/native validation remain pending.',
                            'DSPA-U specification and uncertainty implementation remain unavailable.'],
    'blocker': None,
}
