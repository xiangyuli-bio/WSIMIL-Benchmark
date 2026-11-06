"""Explicit complete reference reconstruction, separate from absent author filter code."""
PROFILES=[{'id':name,'family':'nciemil','recipe':
    'Task-trained full pretrained Swin-Tiny RGB256/max-instance binary pretraining with explicit missing author hyperparameters; complete highK/lowK selection; actual native1024D/four-head graph, literal source batch4/SGD2e-4/wd1e-5/100epochs, harmonic VAL/latest ties and full recovery; '+variant,
    'tasks':['classification'],'required_inputs':['patch_features','nciemil_tasktrained_selected_cache','single_slide_bag'],
    'class_count':None,'external_source':'nciemil','gpu_validation':'not_run','real_wsi_validation':'not_run',
    'status':'implemented_unvalidated','paper_reproduction':'not_run'} for name,variant in [
    ('nciemil_full_reference_released_kv','released interleaved half-length KV'),
    ('nciemil_full_reference_independent_kv_repaired','explicit independent full-token KV repair')]]
METHOD={'id':'nciemil','name':'NcIEMIL','group':'L_composite','scope_origin':'user_164_admission_list',
    'source_url':'https://github.com/polyethylene16/NcIEMIL','source_commit':'780d8b4c04ef6266e552bfd4c47b76470f7510fd',
    'source_status':'complete_reference_reconstruction_under_verification',
    'license':'no_LICENSE_in_pinned_tree_private_source_loading','profiles':[p['id'] for p in PROFILES],
    'status':'adapted','required_inputs':['patch_features','nciemil_tasktrained_selected_cache','single_slide_bag'],
    'paper_reproduction':'not_run','real_wsi_validation':'not_run','blocker':None,
    'requirements_review':'Complete task-trained Swin-Tiny max-instance stage, all-patch score/identity/feature cache, source native aggregation and full-state recovery implemented. Missing author filtering trainer/checkpoint/optimizer/budget are explicit reference choices, never paper-score reproduction. Full native budgets and public CLI acceptance remain required.'}
