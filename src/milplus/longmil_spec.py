"""Original public-v1 and complete declared NeurIPS-v2 reference conditions."""
from .longmil import PROFILES as CONDITIONS
from .longmil_source import COMMIT

PROFILES = [{'id': name, 'family': 'longmil', 'tasks': [condition[1]],
    'required_inputs': ['patch_features', 'longmil_physical_xy', 'patient_disjoint_splits'] +
                       (['full_original_ALiBi_asset'] if condition[0] == 'v1' else []),
    'external_source': 'longmil', 'status': 'adapted_under_verification',
    'gpu_validation': 'not_run', 'real_wsi_validation': 'not_run', 'paper_reproduction': 'not_run',
    'formal_admission': False, 'reference_reconstruction': True,
    'version_condition': condition[0],
    'native_recipe': {'hidden': 384, 'local_blocks': 2, 'global_blocks': 1,
        'heads': 4 if condition[0] == 'v1' else 1, 'FFN': 1536,
        'max_epochs': 200, 'patience': 20, 'lr': .0001, 'weight_decay': .01,
        'optimizer': 'Adam_reference', 'complete_bags': True,
        'local_radius': 10 if condition[0] != 'v1' else 'source_global_minimum_mask'}}
    for name, condition in CONDITIONS.items()]

METHOD = {'id': 'longmil', 'name': 'LongMIL', 'group': 'B_spatial_graph',
    'scope_origin': 'user_164_admission_list', 'source_url': 'https://github.com/invoker-LL/Long-MIL',
    'source_commit': COMMIT, 'source_status': 'original_v1_and_complete_paper_v2_reference_under_verification',
    'license': 'no_LICENSE_in_pinned_tree_private_source_loading_no_redistribution',
    'profiles': [p['id'] for p in PROFILES], 'status': 'adapted_under_verification',
    'required_inputs': ['patch_features', 'longmil_physical_xy', 'patient_disjoint_splits'],
    'formal_admission': False, 'paper_reproduction': 'not_run', 'real_wsi_validation': 'not_run',
    'blocker': None, 'requirements_review': 'Public v1 full90000x90000int8 asset/4heads/384width/'
        'source joint-minimum mask and source eval-dropout retained; separate eval repair. NeurIPS-v2 '
        'original trainer/kernels remain unpublished. Explicit full paper reference implements '
        'two exact2D radius10 sparse/chunked local blocks, actual2x2 observed-tissue pooling, '
        'global ALiBi2D block, native1head/384width/FF1536/dropout0 and classification/full-cohort '
        'patient-survival trainers. All unspecified pooling/optimizer/training choices disclosed. '
        'Actual CUDA xFormers is mandatory; no silent dense/identity fallback. '
        'Full native budgets, source/wheel and global acceptance remain separate gates.'}

# Public metadata only: preserve every structured native training parameter.
from .catalog_contracts import native_profile
PROFILES = [native_profile(p, class_count=None) for p in PROFILES]
