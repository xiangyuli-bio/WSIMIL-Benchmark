"""All three released base-MIL branches of the complete coupled pipeline."""
from .icmil_coupled_source import COMMIT

REQUIRED=['original_RGB256_patch_bags','original_ImageNet_ResNet50_weights','patient_grouped_split','single_slide_bag']
PROFILES=[]
for kind,name,recipe in [
    ('abmil','icmil_coupled_abmil_patient_reference','original1024/512/128 gatedABMIL, Adam.0002'),
    ('maxpool','icmil_coupled_maxpool_patient_reference','original1024/512 maxpoolMIL with literal forward/tie-index and exact deterministic K1 CUDA backward, Adam.0001'),
    ('dtfd','icmil_coupled_dtfd_legacy14_patient_reference','original1024/512/128/five-pseudo-group/two-tier DTFD, two unmodified originalAdam14.0001 with first optimizer before second backward')]:
    PROFILES.append({'id':name,'family':'icmil_coupled','recipe':'Complete original ResNet50conv4/1024 RGB256 extraction, '+recipe+
        '/200epochs/wd.0001/clip5; full teacher/student encoder + reduction/classifier threeKL1/.5/.5, '
        'Adam.00001/wd.0001/200epochs/batch100/VALteacherMAE; selected encoder re-extraction, fresh ABMIL or previous-MIL warm-start Max/DTFD retraining; '
        'explicit patient-disjoint/ImageNet-bootstrap/first-valid-selection/singleton-VAL repair/fixed TEST permutation reference protocol',
        'tasks':['classification'],'required_inputs':REQUIRED,'class_count':2,'external_source':'icmil_coupled',
        'status':'implemented_unvalidated','gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run',
        'backend':'icmil_coupled_complete_reference'})
METHOD={'id':'icmil_coupled','name':'ICMIL Iteratively Coupled','group':'E_encoder_training',
    'scope_origin':'user_164_admission_list','source_url':'https://github.com/Dootmaan/ICMIL','source_commit':COMMIT,
    'source_status':'git_commit_locked_main_MICCAI2023','license':'MIT_private_author_definitions_and_BSD_PyTorch14_Adam',
    'profiles':[p['id'] for p in PROFILES],'status':'adapted','required_inputs':REQUIRED,
    'paper_reproduction':'not_run','real_wsi_validation':'not_run','blocker':None,
    'requirements_review':'Full raw encoder/MIL/threeKL distillation/feedback/retraining pipeline for all3author base branches. '
        'Actual independent Torch1.4 original DTFD loop validates gradients/weights/Adam states; original Adam body retained with BSD notice. '
        'Source patient split/test usage/initialization and selection differences are explicit reference protocol choices. '
        'Complete native200epoch phase budgets, installed-package and whole-library release acceptance remain separate.'}
