"""Full raw-LPA/SP/TOD execution conditions; validation state stays explicit."""
from .lnplmil_source import COMMIT

PROFILES = [{'id': name, 'family': 'lnplmil', 'tasks': ['classification'],
    'required_inputs': ['actual_RGB224', 'observed_TRAIN_patch_labels', 'original_ImageNet_ResNet18',
                        'ordered_patch_ids', 'measured_level0_coordinates', 'patient_disjoint_splits', 'single_slide_bag'],
    'class_count': 2, 'external_source': 'lnplmil', 'status': 'adapted_under_verification',
    'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run',
    'reference_reconstruction': True, 'formal_admission': False,
    'condition': condition, 'native_recipe': {'weak_epochs': 50, 'weak_batch': 256, 'TOD_max_epochs': 200,
        'TOD_patience': 10, 'TOD_accumulation': 2, 'TopK': 200, 'HNSW_radius': 50, 'ROI_ratio': .4,
        'HNSW_canonical_index_build_threads': 1, 'candidate_score_tie_order': 'level0_XY_before_original_TopK'}}
    for name, condition in [('lnplmil_released_with_reference_weak', 'released_source_with_reference_weak'),
                            ('lnplmil_paper_with_reference_weak', 'paper_reference_with_reference_weak')]]

METHOD = {'id': 'lnplmil', 'name': 'LNPL-MIL', 'group': 'L_composite', 'scope_origin': 'user_164_admission_list',
    'source_url': 'https://github.com/szc19990412/LNPL-MIL', 'source_commit': COMMIT,
    'source_status': 'complete_raw_LPA_source_and_paper_reference_pipeline_under_verification',
    'license': 'no_LICENSE_in_pinned_tree_private_source_loading_no_redistribution',
    'profiles': [p['id'] for p in PROFILES], 'status': 'adapted_under_verification',
    'required_inputs': PROFILES[0]['required_inputs'], 'paper_reproduction': 'not_run',
    'real_wsi_validation': 'not_run', 'formal_admission': False, 'blocker': None,
    'requirements_review': 'Full observed-LPA ResNet18, separate frozen ImageNet512D features, actual SP/Top200 '
        'and full four-layer TOD/Lookahead-RAdam/gradacc2. Original weak-training config is unpublished; '
        'explicit50epoch SGD reference choices are disclosed. Released one-shot/convolution replacement/CE overwrite '
        'and paper disjoint/residual/auxiliary conditions are distinct. Public runs explicitly use canonical single-thread '
        'HNSW index construction; original default-thread operators remain available as separate components. '
        'Binary slide diagnosis only; COAD survival '
        'is not claimed. Native, installed-wheel, whole-library and formal release gates remain.'}

# Public metadata only: preserve every structured native training parameter.
from .catalog_contracts import native_profile
PROFILES = [native_profile(p, class_count=2) for p in PROFILES]
