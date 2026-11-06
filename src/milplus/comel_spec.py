"""CoMEL is a complete continual protocol, not one task's GDAT class."""
from .comel_source import COMMIT

PROFILES=[]
for name,repair in [('comel_owlora_source_patient_reference',False),('comel_owlora_svd_basis_repaired',True)]:
    PROFILES.append({'id':name,'family':'comel','tasks':['classification'],'required_inputs':
        ['patch_features','single_slide_bag','ordered_patch_ids','patient_disjoint_continual_binary_tasks'],
        'class_count':None,'external_source':'comel','status':'adapted_under_verification',
        'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run',
        'svd_basis_assignment_repaired':repair,
        'recipe':'native GDAT512/two blocks/eight heads; BPPL; orthogonal gradient projection; rank16/48; Adam1e-4/wd1e-5; cosine100; fixed final task snapshots',
        'reference_repairs':'patient-disjoint VAL; task count equals manifest; stable available-class evaluation; VAL-only localization thresholds',
        'source_semantics':'Original missing SVD-basis assignment retained' if not repair else 'Explicitly assign U/Vh/S to otherwise zero-up initial LoRA basis'})

METHOD={'id':'comel','name':'CoMEL','group':'I_continual','scope_origin':'user_164_admission_list',
    'source_url':'https://github.com/Hyun1A/CoMEL','source_commit':COMMIT,
    'source_status':'complete_continual_reference_under_verification',
    'license':'no_LICENSE_in_pinned_tree_private_source_loading','status':'adapted_under_verification',
    'profiles':[r['id'] for r in PROFILES],'required_inputs':PROFILES[0]['required_inputs'],
    'paper_reproduction':'not_run','real_wsi_validation':'not_run',
    'requirements_review':'Actual pinned GDAT/BPPL/OWLoRA training methods; complete ordered task sequence and dynamic heads; exact active/inactive gradients, optimizer/scheduler/tau2/RNG recovery; original vs explicit SVD-basis repair separated. Software, source oracle, installed-wheel and native gates required.',
    'blocker':None}
