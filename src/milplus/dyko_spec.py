"""Public code profiles; original TITAN assets and research validation are separate."""

EXECUTION_CONDITIONS = {
    f'dyko_{cohort.lower()}_{suffix}': {'cohort': cohort, 'faiss_backend': backend}
    for cohort in ['CAMELYON16', 'NSCLC', 'RCC', 'UBC']
    for suffix, backend in [
        ('cpu_faiss_source_reference', 'cpu_faiss_reference'),
        ('cuda_faiss_requested_reference', 'cuda_faiss'),
    ]
}
DEFAULT_PROFILE = 'dyko_nsclc_cpu_faiss_source_reference'
REQUIRED_INPUTS = [
    'native_aligned768D_patch_features', 'actual_pretrained_TITAN_text_tower',
    'released_cohort_concept_bank_and_class_prompts',
    'patient_safe_source_split_and_fewshot_manifest',
    'native_GPU_FAISS_or_explicit_CPU_reference',
]
PROFILES = [{
    'id': identifier, 'family': 'dyko',
    'recipe': f"Full original TITAN DyKo, {condition['cohort']}, {condition['faiss_backend']}; "
              '16-shot default, Adam/accumulation4, VAL accuracy, max1000/patience40',
    'tasks': ['classification'], 'required_inputs': REQUIRED_INPUTS,
    'class_count': {'CAMELYON16': 2, 'NSCLC': 2, 'RCC': 3, 'UBC': 5}[condition['cohort']],
    'external_source': 'dyko', **condition, 'status': 'adapted_under_verification',
    'formal_admission': False, 'gpu_validation': 'not_run',
    'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run',
} for identifier, condition in EXECUTION_CONDITIONS.items()]
METHOD = {
    'id': 'dyko', 'name': 'DyKo', 'group': 'G_vision_language',
    'scope_origin': 'user_164_admission_list',
    'source_url': 'https://github.com/junjianli106/DyKo',
    'source_status': 'full_original_asset_adapter_code_integrated_under_verification',
    'source_commit': 'a8b43645840cf283c334b07c42cf4ddf230f644a',
    'license': 'no_LICENSE_in_pinned_tree_private_loading_only',
    'profiles': list(EXECUTION_CONDITIONS), 'status': 'adapted_under_verification',
    'default_execution_profile': DEFAULT_PROFILE, 'required_inputs': REQUIRED_INPUTS,
    'formal_admission': False, 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run',
    'requirements_review': 'Full model, native inputs, original-asset binding, isolated training, '
        'checkpoint recovery and evaluation are integrated into the public engine. '
        'Family execution selects the prepared cohort and explicitly uses CPU FAISS; '
        'CUDA FAISS requires its separately named profile and real GPU FAISS. '
        'No replacement text model or random pretrained weights are used.',
    'prerequisites': ['Run setup and prepare-dyko-assets with authorized original TITAN files.',
                      'Bind native visual provenance and patients with prepare-dyko.',
                      'Use the separate dyko runtime dependencies; GPU FAISS is not in the CPU extra.'],
    'validation_blockers': ['Complete original-pretrained source/wheel/native-budget validation remains pending.'],
    'blocker': None,
}
