"""Complete two-stage MExD paper references, not an MoE-only approximation."""
from .mexd_graph import COHORTS
from .mexd_source import COMMIT

PROFILES=[{'id':'mexd_'+cohort+'_paper_reference','family':'mexd','tasks':['classification'],
    'required_inputs':['patch_features','single_slide_bag','patient_disjoint_splits'],
    'external_source':'mexd','class_count':spec['classes'],'status':'adapted_under_verification',
    'formal_admission':False,'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run',
    'reference_reconstruction':True,'native_recipe':{'feature_dimensions':[384,768],'adapter_width':512,
        'adapters':2,'layers_each':2,'heads':8,'landmarks':256,'experts':spec['experts'],
        'observed_background_class':spec['observed_background'],'moe_epochs':100,'diffusion_epochs':200,
        'diffusion_steps':500,'posterior_samples':100,'moe_lr':2e-4,'diffusion_lr':1e-3,
        'moe_weight_decay':1e-5,'diffusion_weight_decay':0.,'diffusion_clip_norm':1.,
        'batch_size':1,'selection':'fixed_final_epoch'}}for cohort,spec in COHORTS.items()]
PROFILES.append({**PROFILES[1],'id':'mexd_class_port_reference','class_count':None,
    'native_recipe':{**PROFILES[1]['native_recipe'],'experts':'C_if_observed_class0background_else_Cplus1',
                     'observed_background_class':'explicit_optional_task.class_roles.background',
                     'output_class_port':True}})
METHOD={'id':'mexd','name':'MExD','group':'L_composite','scope_origin':'user_164_admission_list',
    'source_url':'https://github.com/JWZhao-uestc/MExD','source_commit':COMMIT,
    'source_status':'complete_original_adapters_denoiser_and_diffusion_with_declared_full_paper_reference_MoE_trainer',
    'license':'no_LICENSE_in_pinned_tree_private_source_loading_only',
    'profiles':[p['id']for p in PROFILES],'status':'adapted_under_verification','formal_admission':False,
    'required_inputs':['actual_ordered_frozen_features','patient_disjoint_splits','single_slide_bag'],
    'paper_reproduction':'not_run','real_wsi_validation':'not_run','blocker':None,
    'requirements_review':'Both complete512D/two-layer8head256landmark TransMIL/PPEG adapters, '
        '2/3/4expert routes, negative and bag-matched expert supervision, confidence-weighted condition, '
        'full original conditional denoiser and shifted500step diffusion preserved. Complete two-stage '
        '100epoch RAdam plus200epoch Adam/cosine training, frozen stage2 auxiliary,100posterior paths '
        'and paired uncertainty are implemented. Incompatible released pretrain/inference graphs and '
        'missing hardcoded checkpoints are not silently substituted: unified paperEq2/3/9, initialization, '
        'empty-expert and selection choices are declared reference reconstructions. Full software/native/ '
        'source/wheel and final gates remain independent; model-only checks are insufficient.'}

# Public metadata only: preserve every structured native training parameter.
from .catalog_contracts import native_profile
PROFILES = [native_profile(p, class_count=p['class_count']) for p in PROFILES]
