"""Full published IIB components plus explicitly named reconstructed trainers."""
from .iibmil import CONDITIONS
from .iibmil_source import COMMIT

RECIPES={'luad_gm':(32,8,16,.0002,2),'tcga_nsclc':(256,3,16,.00001,2),'tcga_rcc':(256,4,4,.00001,3)}
PROFILES=[{'id':name,'family':'iibmil','tasks':['classification'],
    'required_inputs':['patch_features','single_slide_bag','patient_disjoint_splits'],
    'external_source':'iibmil','status':'adapted_under_verification','formal_admission':False,
    'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run',
    'reference_reconstruction':True,'condition':condition,
    'native_recipe':{'input_dim':1280,'hidden':RECIPES[condition][0],'depth':RECIPES[condition][1],
        'batch_size':RECIPES[condition][2],'lr':RECIPES[condition][3],'classes':RECIPES[condition][4],
        'tokens':500,'epochs':300,'confidence_warmup':10,'confidence_ema_denominator':200,
        'prototype_ema':.99,'optimizer':'AdamW','weight_decay':.00001,'clip_norm':.01,
        'lr_drop':100,'lr_gamma':.1,'bag_loss_weight':1.,'instance_loss_weight':1.}}
    for name,condition in CONDITIONS.items()]
METHOD={'id':'iibmil','name':'IIB-MIL','group':'A_frozen','scope_origin':'user_164_admission_list',
    'source_url':'https://github.com/TencentAILabHealthcare/IIB-MIL','source_commit':COMMIT,
    'source_status':'complete_original_network_data_confidence_and_prototype_with_reference_training_under_verification',
    'license':'no_LICENSE_in_pinned_tree_private_source_loading_no_redistribution',
    'profiles':[p['id'] for p in PROFILES],'status':'adapted_under_verification','formal_admission':False,
    'required_inputs':['patch_features','patient_disjoint_splits','single_slide_bag'],
    'paper_reproduction':'not_run','real_wsi_validation':'not_run','blocker':None,
    'requirements_review':'Complete original residual-transformer/five-query decoder, original TRAIN confidence '
        'initialization, PartialLoss/linear200EMA and class-adjusted top10percent prototype update. '
        'Original500-row sampling/padding and ignored attention mask retained. Full author trainer is absent: '
        'equal dual-loss coefficients,10epoch confidence warmup, per-batch lagged update order, valid-instance '
        'supervision mask and validation selection are declared reconstruction choices. All3full native '
        'configurations, FM-width and output-class ports retain source widths/depths/FFNs. '
        'Checkpoint includes complete confidence/prototypes/scheduler/actual sampled rows and RNG. '
        'Native300epoch/source/wheel/global/final gates remain separate; no paper-score reproduction claim.'}

# Public metadata only: preserve every structured native training parameter.
from .catalog_contracts import native_profile
PROFILES = [native_profile(p, class_count=p['native_recipe']['classes']) for p in PROFILES]
