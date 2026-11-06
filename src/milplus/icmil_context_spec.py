"""Pretrained in-context ICMIL, separate from iteratively coupled ICMIL."""
from .icmil_context_source import COMMIT,WEIGHTS

REQUIRED=['fixed_length_patch_features_at_most25D','binary_slide_labels','patient_grouped_split',
    'actual_released_ICMIL_8000epoch_checkpoint']
PROFILES=[{'id':'icmil_context_'+seed+'_source','family':'icmil_context',
    'recipe':'Complete released8,257,552-parameter PFN/Perceiver pretrained8000epoch seed '+seed+
        '; all labelled TRAIN bags as context, original TRAIN-only F64 normalization, all unlabelled queries, '
        'binary logits; zero downstream gradient updates and no validation-selected weights; '
        'explicit fixed-bag/native<=25D or named TRAIN-only PCA25 reference preparation',
    'tasks':['classification'],'required_inputs':REQUIRED,'class_count':2,'external_source':'icmil_context',
    'status':'implemented_unvalidated','gpu_validation':'not_run','real_wsi_validation':'not_run',
    'paper_reproduction':'not_run','backend':'icmil_context_original_pretrained'} for seed in WEIGHTS]
METHOD={'id':'icmil_context','name':'ICMIL In-context','group':'J_pretrained_slide',
    'scope_origin':'user_164_admission_list','source_url':'https://github.com/injurise/ICMIL',
    'source_commit':COMMIT,'source_status':'git_commit_locked','license':'MIT_with_third_party_notices',
    'profiles':[p['id'] for p in PROFILES],'status':'adapted','required_inputs':REQUIRED,
    'paper_reproduction':'not_run','real_wsi_validation':'not_run','blocker':None,
    'requirements_review':'Actual original three released checkpoints, full pretrained source graph and author TCGA50split reference inference; '
        'new complete unified fixed-context fit/evaluation, exact input binding and explicit projection preparation. '
        'Software/native/independent-wheel/full-library acceptance remains separate. No new WSI cohort or paper-score reproduction claimed.'}
