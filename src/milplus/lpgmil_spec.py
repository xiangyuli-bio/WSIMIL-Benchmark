"""Complete LPGMIL references; no random prototype or attention-only fallback."""
from .lpgmil_source import COMMIT

PROFILES=[{'id':name,'family':'lpgmil','tasks':['classification'],
    'required_inputs':['patch_features','single_slide_bag','patient_disjoint_splits','lpgmil_train_prototypes'],
    'external_source':'lpgmil','class_count':2 if 'tls1024'in name else None,
    'status':'adapted_under_verification','formal_admission':False,'gpu_validation':'not_run',
    'real_wsi_validation':'not_run','paper_reproduction':'not_run','reference_reconstruction':True,
    'native_recipe':{'hidden':512,'layers':2,'state':16,'conv':4,'expand':2,'heads':8,
        'prototypes':6,'topk':1000,'masked_top':10,'mask_fraction':.6,'epochs':50,
        'optimizer':'Adam','lr':1e-4,'weight_decay':1e-5,'all_parameter_l1':1e-5,
        'input_dim':1024,'batch_size':1,'grad_clip':None,'scheduler':None,'selection':'last_epoch'}}
    for name in ['lpgmil_tls1024_reference','lpgmil_fm_width_reference']]
METHOD={'id':'lpgmil','name':'LPGMIL','group':'H_extra_supervision','scope_origin':'user_164_admission_list',
    'source_url':'https://github.com/FPXMU/LPGMIL','source_commit':COMMIT,
    'source_status':'complete_pinned_native_graph_and_dual_loss_with_explicit_TRAIN_prototype_reconstruction',
    'license':'no_LICENSE_in_pinned_tree_private_loading_only; Mamba1.1.1 Apache2.0',
    'profiles':[p['id']for p in PROFILES],'status':'adapted_under_verification','formal_admission':False,
    'required_inputs':['original_ordered_features_and_patch_ids','TRAIN_only_actual_nucleus_counts',
        'nucleus_detector_provenance','patient_disjoint_splits'],
    'paper_reproduction':'not_run','real_wsi_validation':'not_run','blocker':None,
    'requirements_review':'Full512D/two_original_Mamba1.1.1layers/state16/conv4/expand2, '
        'six_masked8head_prototype_branches/top1000/reaggregated_slide_head and all losses retained. '
        'Released query width, constructor and output contracts repaired explicitly. Missing original '
        'proto_vector.pt/preparer replaced by named TRAIN-only actual lymphocyte-candidate Ward reference '
        'with explicit thresholds, detector provenance and complete candidate/patch/feature/split hashes. '
        'Original first-head hardtopk, source all-parameterL1 and actual one-bag optimizer updates preserved. '
        'All native/source/wheel/full-budget and whole-library gates remain independently required.'}

# Public metadata only: preserve every structured native training parameter.
from .catalog_contracts import native_profile
PROFILES = [native_profile(p, class_count=p['class_count']) for p in PROFILES]
