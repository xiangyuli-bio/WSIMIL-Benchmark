"""Complete RGB Prompt-MIL configurations; no cached-vector or random-backbone fallback."""
from .promptmil import RECIPES
from .promptmil_source import COMMIT

PROFILES=[{'id':name,'family':'promptmil','recipe':'Original complete RGB/DINO prompt-gradient replay with '+recipe['edition']+' cohort recipe',
    'tasks':['classification'],'required_inputs':['promptmil_actual_RGB224','single_slide_bag','patient_disjoint_splits'],
    'external_source':'promptmil','class_count':None,'status':'adapted_under_verification','formal_admission':False,
    'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run','native_recipe':dict(recipe)}
    for name,recipe in RECIPES.items()]
METHOD={'id':'promptmil','name':'Prompt-MIL','group':'E_encoder_training','scope_origin':'user_164_admission_list',
    'source_url':'https://github.com/cvlab-stonybrook/PromptMIL','source_commit':COMMIT,
    'source_status':'full_original_RGB_DINO_DSMIL_and_three_pass_trainer_with_source_and_declared_paper_reference_adapters',
    'license':'No repository-wide LICENSE; private pinned author loading only; DINO upstream Apache2.0 headers retained',
    'profiles':[p['id']for p in PROFILES],'status':'adapted_under_verification','formal_admission':False,'blocker':None,
    'required_inputs':['actual_RGB224_uint8_and_ordered_ids','original_author_cohort_DINO_teacher_weights','patient_disjoint_splits'],
    'paper_reproduction':'not_run','real_wsi_validation':'not_run',
    'requirements_review':'Full192D/depth12/3head pretrained DINO, one shallow prompt or explicit full deep-prompt option, '
        'dual DSMIL bag/max loss and retained-feature-gradient RGB recomputation. Complete original three cohort teachers '
        'bound by SHA and every native backbone key. Source8-slide unnormalized accumulation with cross-epoch tail gradient '
        'carry and final-epoch evaluation is preserved. Paper-budget reference explicitly flushes tail and uses validation '
        'source Sigmoid macroOVR AUC; normalized shared probabilities exported separately. No per-epoch test monitoring. '
        'All pending prompt/classifier gradients, optimizer/scheduler and RNG retained on resume. '
        'Source/paper40or50epoch/native chunk512or256 and all full-process gates remain separately verified.'}
