import json
from importlib.resources import files


def resource(name):
    return json.loads(files("milplus").joinpath("resources", name).read_text())


def methods():
    return resource("methods.json")


def profiles():
    return resource("profiles.json")


def normalize(name):
    return "".join(c for c in name.lower() if c.isalnum())


def inspect_method(name):
    if normalize(name) in {'dspau', 'dspaumil', 'dspamilu'}:
        raise ValueError('DSPA-U-MIL is not implemented; use an explicit DSPA-MIL2025 reference profile instead')
    if normalize(name) == 'dyko':
        from .dyko_spec import DEFAULT_PROFILE
        name = DEFAULT_PROFILE
    if normalize(name) in {'dspamil', 'dspamil2025', 'dspamildspaumil'}:
        from .dspamil_spec import DEFAULT_PROFILE
        name = DEFAULT_PROFILE
    if normalize(name) in {'feather', 'feathermillabmodels'}:
        from .feather_spec import DEFAULT_PROFILE
        name = DEFAULT_PROFILE
    if normalize(name) == 'fcmil' and name != 'fcmil':
        name = 'fcmil_original_benchmark100_cr_auc_repaired'
    if normalize(name) == 'nnmil' and name != 'nnmil':
        name = 'nnmil_author_100_bacc_train_planned_locked_test'
    if normalize(name) in {'mrblock', 'mrabmil', 'mrblockmrabmil'} and name != 'mrblock':
        name = 'mrblock_author_200_mr_abmil_earlystop_locked_test'
    if normalize(name) == 'carmil' and name != 'carmil':
        name = 'carmil_author_100_attmil_l1_locked_test_repaired'
    if normalize(name) == 'attrimil' and name != 'attrimil':
        name = 'attrimil_author_200_main_repaired'
    if normalize(name) == 'asmil' and name != 'asmil':
        name = 'asmil_author_50_literal_nan_teacher_locked_test'
    if normalize(name) == 'pneamil' and name != 'pneamil':
        name = 'pneamil_author_val_selector_repaired'
    if normalize(name)=='rmdl':name='rmdl_paper_mean_full_reference'
    if normalize(name)=="mexd":name="mexd_class_port_reference"
    if normalize(name)=="lpgmil":name="lpgmil_tls1024_reference"
    if normalize(name)=='promptmil':name='promptmil_brca_dino_dsmil_paper_reference'
    if normalize(name)=='iibmil':name='iibmil_tcga_nsclc_complete_reference'
    if normalize(name)=='longmil':name='longmil_v2_paper_classification_reference'
    if normalize(name)=='lnplmil':name='lnplmil_paper_with_reference_weak'
    if normalize(name) in {'hybridsupervision','hybridsupervisionlearning'}:name='hybrid_supervision_paper_full_reference'
    if normalize(name)=="mggmmil":name="mggmmil_uaf_original_graph_patient_reference"
    if normalize(name)=='macnet':name='macnet_public_source_cached_patient_reference'
    if normalize(name)=='protomil':name='protomil_concept_paper_patient_reference'
    if normalize(name)=='kmpmil':name='kmpmil_conch_max_query_patient_reference'
    if normalize(name)=='qpmilvl':name='qpmilvl_conch_max_query_patient_reference'
    if normalize(name)=='akdpmp':name='akdpmp_transmil_tcga_patient_reference' 
    # A technical family ID requests its metadata and complete architecture
    # choices. A display-name alias or explicit profile requests execution.
    if name == 'icmil_coupled':
        family = next(r for r in methods() if r['id'] == name)
        return {**family, 'kind': 'family', 'runnable': False,
                'execution_profiles': [p for p in profiles() if p['family'] == name],
                'default_execution_profile': 'icmil_coupled_abmil_patient_reference'}
    if normalize(name)=='comel':name='comel_owlora_source_patient_reference' 
    if normalize(name)=='pmilprototypical':
        name='pmil_prototypical_raw_patient_reference'
    if normalize(name) in {'icmilcoupled','icmiliterativelycoupled','iterativelycoupledmil'}:
        name='icmil_coupled_abmil_patient_reference'
    if normalize(name) in {'icmilcontext','icmilincontext'}:
        name='icmil_context_c5trd795_source'
    if normalize(name)=='attention2majority':
        name='attention2majority_camelyon_source_repaired'
    if normalize(name)=='nciemil':
        name='nciemil_full_reference_released_kv'
    if normalize(name)=='taxomil':
        name='taxomil_gpt2_all_losses_key_repaired'
    if normalize(name) == 'gmmamba':
        name = 'gmmamba_mambamil_bimamba_source'
    if normalize(name) == 'muse':
        name = 'muse_released_source'
    if normalize(name) == 'libramil':
        name = 'libramil_source_p4'
    if normalize(name) == 'm4':
        name = 'm4_multilabel_source'
    if normalize(name) == 'convl':
        name = 'convl_yaml_fp16_source_repaired'
    if normalize(name) in {'centerembeddings', 'milwithcenterembeddings', 'chikonmil'}:
        name = 'center_embeddings_bag_classifier_repaired'
    if normalize(name) == 'miladapter':
        name = 'miladapter_taskres_abmil'
    if normalize(name) == 'conmil':
        name = 'conmil_posneg_source'
    if normalize(name) == 'sparseconvmil':
        name = 'sparseconvmil_raw_source_c32'
    if normalize(name) in {'c2c', 'clustertoconquer'}:
        name = 'c2c_cluster_source'
    if normalize(name) == 'sammil':
        name = 'sammil_attn_objective_repaired'
    if normalize(name) == 'lrmil':
        name = 'lrmil_source_repaired'
    if normalize(name) == 'kmgmamba':
        name = 'kmgmamba_source'
    if normalize(name) == 'ahdmil':
        name = 'ahdmil_brca_source_repaired'
    if normalize(name) == 'hdmil':
        name = 'hdmil_brca_source'
    if normalize(name) == 'mgtrans':
        name = 'mgtrans_source_repaired'
    if normalize(name) == 'zoommil':
        name = 'zoommil_bright_source_repaired'
    if normalize(name) == 'augdiff':
        name = 'augdiff_amil_schedule_repaired_reconstruction'
    if normalize(name)=='vlsa':
        name='vlsa_conch_source_repaired'
    if normalize(name)=='slip':
        name='slip_clip_vit_b16_source_repaired'
    if normalize(name)=='dispro':
        name='dispro_source_repaired'
    if normalize(name) == 'amiml':
        name = 'amiml_paper_reference'
    if normalize(name) == 'dtmil':
        name = 'dtmil_repaired_reference_train'
    if normalize(name) in {'pamilpolicy','pamildynamicpolicy'}:
        name = 'pamil_policy_paper_reference'
    if normalize(name) == 'focus':
        name = 'focus_source'
    if normalize(name) == 'codmil':
        name = 'codmil_source_repaired'
    if normalize(name) == 'ins':
        name = 'ins_frozen_source_repaired'
    if normalize(name) == 'sclwc':
        name = 'sclwc_paper_reference'
    if normalize(name) == 'deepgraphsurv':
        name = 'deepgraphsurv_agcn'
    if normalize(name) == 'triagemil':
        name = 'triagemil_source'
    if normalize(name) == 'c2mil':
        name = 'c2mil_source'
    if normalize(name) == 'gatmamba':
        name = 'gatmamba_source'
    if normalize(name) == 'protosurv':
        name = 'protosurv_repaired'
    if normalize(name) == 'pathrwkv':
        name = 'pathrwkv_repaired'
    if normalize(name) == 'c2aug':
        name = 'c2aug_transmil_paper_reference'
    if normalize(name)=='agdrmil':
        name='agdrmil_yaml_source'
    if normalize(name)=='h2mil':
        name='h2mil_repaired'
    if normalize(name)=='hipt':
        name='hipt_lgp_frozen_source'
    if normalize(name) == 'simil':
        name = 'simil_full_vjp'
    if normalize(name) in {'mamba2dmil', '2dmambamil'}:
        name = 'mamba2dmil_paper'
    if normalize(name) == 'ttis':
        name = 'ttis_abmil_source'
    if normalize(name) in {'msattention', 'multiattentionmil', 'multiscaleattentionmilwibawa'}:
        name = 'msattention_repaired'
    if normalize(name) in {'pmilshapley', 'pmilshapleypseudobag'}:
        name = 'pmil_shapley_abmil_repaired'
    if normalize(name) == 'mammoth':
        name = 'mammoth_abmil'
    aliases = {"cdpmil": "cdpmil_source", "rankmix": "rankmix_dsmil", "sdfp": "sdfp_abmil", "mfcmil": "mfcmil_abmil", "clam": "clam_sb", "mildropout": "mil_dropout_abmil", "packmil": "packmil_abmil", "dtfdmil": "dtfd", "abmilgated": "gated_abmil",
               "ilramil": "ilra", "acmil": "acmil_ga", "aem": "aem_abmil", "mhim": "mhim_abmil", "mhimmil": "mhim_abmil",
               "psemix": "psemix_abmil", "prodiv": "prodiv_abmil", "remix": "remix_abmil",
               "mmiltransformer": "mmil", "hagmil": "hagmil_ce", 'smmil': 'smmil_early', 'psamil': 'psamil',
               'bayesmil': 'bayesmil_apcrf', 'caprmil': 'caprmil_mean', 'efficientmil': 'efficientmil_gru'}
    aliases['recomil'] = 'recomil_single'
    aliases['mambamil'] = 'mambamil_srmamba'
    aliases['mfemil'] = 'mfemil_abmil'
    aliases['pamoe'] = 'pamoe_transmil'
    aliases['pathomics'] = 'pathomics_concat_source'
    aliases['mmp'] = 'mmp_panther_source'
    aliases['pibd'] = 'pibd_source'
    aliases['advmil'] = 'advmil_esat'
    aliases['hmil'] = 'hmil_source_repaired'
    aliases['smmilemulticlass'] = 'smmile'
    aliases['pamilprototype'] = 'pamil_prototype_repaired'
    aliases['pamilprototypeattention'] = 'pamil_prototype_repaired'
    aliases['monai'] = 'monai_transformer_tutorial'
    aliases['monaimil'] = 'monai_transformer_tutorial'
    aliases['dependencyawaremilmonai'] = 'monai_transformer_tutorial'
    aliases['stablemil'] = 'stablemil_classification_repaired'
    aliases['setmil'] = 'setmil_repaired'
    aliases['gtp'] = 'gtp_repaired'
    aliases['probsamil'] = 'probsa_abmil_probmean'
    aliases['micromil'] = 'micromil_frozen'
    aliases['lossattention'] = 'loss_attention_classical'
    aliases['lossbasedattention'] = 'loss_attention_classical'
    name = aliases.get(normalize(name), name)
    for p in profiles():
        if normalize(p["id"]) == normalize(name):
            family = next(r for r in methods() if r["id"] == p["family"])
            return {**p, "source_url": family["source_url"], "runnable": True}
    matches = [r for r in methods() if normalize(name) in {normalize(r["id"]), normalize(r["name"])}]
    if len(matches) != 1:
        raise ValueError(f"Unknown/ambiguous method {name}; use `milplus methods`")
    # A catalog's descriptive display name uses the existing family/default
    # route. Re-enter only for a different normalized name, so pending families
    # still terminate here and no new architecture default is invented.
    family = matches[0]
    if normalize(name) == normalize(family["name"]) and normalize(name) != normalize(family["id"]):
        return inspect_method(family["id"])
    return {**matches[0], "runnable": False}


def require_runnable(name):
    spec = inspect_method(name)
    # Execution still accepts the technical family ID and uses its documented
    # full ABMIL pipeline. Inspecting a family does not silently discard its
    # independent MaxPool and DTFD alternatives.
    if spec.get('kind') == 'family' and spec.get('default_execution_profile'):
        spec = inspect_method(spec['default_execution_profile'])
    if not spec["runnable"]:
        raise ValueError(f"{spec['id']}: {spec['status']}. {spec['blocker']}")
    return spec


def canonical_catalog_display_name(name):
    """Resolve a registered display alias while preserving explicit profile IDs."""
    normalized = normalize(name)
    if any(normalize(profile["id"]) == normalized for profile in profiles()):
        return name
    matches = [record for record in methods() if normalize(record["name"]) == normalized]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous catalog method {name}; select an explicit execution profile")
    return matches[0]["id"] if matches else name


def resolve_for_bundle(name, bundle):
    if normalize(name) == "fcmil" and name != "fcmil":
        return require_runnable("fcmil_original_benchmark100_cr_auc_repaired")
    if normalize(name) == "nnmil" and name != "nnmil":
        return require_runnable("nnmil_author_100_bacc_train_planned_locked_test")
    # Bundle-sensitive choices (task type, class count, native input condition)
    # must also work when the caller uses the full catalog display name.
    if normalize(name) in {"mrblock", "mrabmil", "mrblockmrabmil"} and name != "mrblock":
        return require_runnable("mrblock_author_200_mr_abmil_earlystop_locked_test")
    if normalize(name) == "carmil" and name != "carmil":
        return require_runnable("carmil_author_100_attmil_l1_locked_test_repaired")
    if normalize(name) == "attrimil" and name != "attrimil":
        return require_runnable("attrimil_author_200_main_repaired")
    if normalize(name) == "asmil" and name != "asmil":
        return require_runnable("asmil_author_50_literal_nan_teacher_locked_test")
    if normalize(name) == "pneamil" and name != "pneamil":
        return require_runnable("pneamil_author_val_selector_repaired")
    name = canonical_catalog_display_name(name)
    if normalize(name) == 'dyko' and getattr(bundle, 'dyko', None) is not None:
        return require_runnable('dyko_' + bundle.dyko.cohort.lower() + '_cpu_faiss_source_reference')
    if normalize(name) == 'feather' and getattr(bundle, 'feather', None) is not None:
        return require_runnable('feather_' + bundle.feather.encoder + '_finetune_val20_reference')
    if normalize(name)=='promptmil' and getattr(bundle,'promptmil',None)is not None:
        return require_runnable('promptmil_'+bundle.promptmil.condition+'_dino_dsmil_paper_reference')
    if normalize(name)=='iibmil':
        return require_runnable('iibmil_tcga_rcc_complete_reference' if len(bundle.classes)>2 else 'iibmil_tcga_nsclc_complete_reference')
    if normalize(name)=='longmil':
        return require_runnable('longmil_v2_paper_survival_reference' if bundle.task_type=='survival' else 'longmil_v2_paper_classification_reference')
    """Resolve SMMILe's declared task route; explicit execution IDs keep their meaning."""
    if normalize(name)=='macnet' and getattr(bundle,'macnet',None) is not None:
        return require_runnable('macnet_'+bundle.macnet.condition+'_source_cached_patient_reference')
    if normalize(name)=='attention2majority':
        return require_runnable('attention2majority_kidney_paper_reference' if len(bundle.classes)==3 else 'attention2majority_camelyon_source_repaired')
    if normalize(name) == 'conmil' and len(bundle.classes) != 2:
        return require_runnable('conmil_posneg_multiclass_repaired')
    if normalize(name) in {'c2c', 'clustertoconquer'} and len(bundle.classes) != 2:
        return require_runnable('c2c_cluster_multiclass_repaired')
    if normalize(name)=='slip' and bundle.slip is not None:
        return require_runnable('slip_'+bundle.slip.architecture+'_source_repaired')
    if normalize(name) == 'stablemil':
        return require_runnable('stablemil_survival_repaired' if bundle.task_type == 'survival' else 'stablemil_classification_repaired')
    if normalize(name) in {'pamilprototype', 'pamilprototypeattention'}:
        return require_runnable('pamil_prototype_multilabel' if bundle.task_type == 'multilabel' else 'pamil_prototype_repaired')
    if normalize(name) == 'smmile':
        variant = bundle.config.get('smmile_variant', 'multilabel' if bundle.task_type == 'multilabel' else 'multiclass')
        choices = {'binary': 'smmile_binary', 'multiclass': 'smmile', 'multilabel': 'smmile_multilabel'}
        if not isinstance(variant, str) or variant not in choices:
            raise ValueError('task.smmile_variant must be binary, multiclass or multilabel')
        return require_runnable(choices[variant])
    return require_runnable(name)


def check_compatibility(spec, bundle):
    if spec['id'].startswith('dyko_'):
        if getattr(bundle,'dyko',None) is None:raise ValueError('DyKo requires original aligned visual/concept/TITAN inputs; run prepare-dyko-assets and prepare-dyko')
        if bundle.dyko.cohort != spec['cohort']:
            raise ValueError('DyKo profile cohort differs from the prepared concept/class inputs')
        bundle.dyko.verify(deep=True)
        return
    if getattr(bundle,'dyko',None) is not None:
        raise ValueError('Bound DyKo inputs require the explicit few-shot protocol; prepare the same selected TRAIN population for comparator methods')
    if spec['id'].startswith('feather_'):
        if getattr(bundle, 'feather', None) is None:
            raise ValueError('FEATHER requires original24K weights and native features; run prepare-feather-assets and prepare-feather')
        if bundle.feather.encoder != spec['encoder']:
            raise ValueError('FEATHER profile encoder differs from the prepared pretrained/native feature pairing')
        bundle.feather.verify(deep=True)
        return
    if getattr(bundle, 'feather', None) is not None:
        raise ValueError('Bound FEATHER original-pretrained inputs require an explicit FEATHER protocol')
    if spec['id'].startswith('dspamil_'):
        if getattr(bundle, 'dspamil', None) is None:
            raise ValueError('DSPA-MIL requires native UNI/concept alignment and TRAIN-only pattern targets; run prepare-dspamil')
        bundle.dspamil.verify(deep=True)
        return
    if getattr(bundle, 'dspamil', None) is not None:
        raise ValueError('Bound DSPA concept/pattern inputs require the explicit DSPA-MIL2025 reference protocol')
    if spec['id'].startswith('rmdl_'):
        if getattr(bundle,'rmdl',None) is None:
            raise ValueError('Complete RMDL requires actual original-scale WSI and TRAIN-only pixel annotations')
        bundle.rmdl.check(deep=True)
        return
    if getattr(bundle,'rmdl',None) is not None:
        raise ValueError('RMDL WSI/pixel-supervision inputs cannot be consumed as a frozen FM feature bag')
    if spec["id"].startswith("mexd_"):
        from .mexd import check_inputs
        check_inputs(bundle,spec["id"])
        return
    if spec["id"].startswith("lpgmil_"):
        from .lpgmil import check_inputs
        check_inputs(bundle,spec["id"])
        return
    if spec['id'].startswith('promptmil_'):
        from .promptmil import check_inputs
        check_inputs(bundle,spec['id']);bundle.check(deep=True)
        return
    if getattr(bundle,'promptmil',None)is not None:raise ValueError('Prompt-MIL raw pixels require their explicit complete RGB prompt training profile')
    if spec['id'].startswith('iibmil_'):
        from .iibmil import check_inputs
        check_inputs(bundle)
        return
    if spec['id'].startswith('longmil_'):
        from .longmil import input_spec
        input_spec(bundle, spec['id'], deep=True)
        bundle.check(deep=True, require_coords=True)
        return
    if spec['id'].startswith('lnplmil_'):
        if getattr(bundle,'lnplmil',None) is None:raise ValueError('LNPL needs actual RGB224/TRAIN patch annotations and the full weak/scoring/SP/TOD protocol')
        bundle.check(deep=True)
        return
    if getattr(bundle,'lnplmil',None) is not None:raise ValueError('LNPL raw LPA data require an explicit complete LNPL execution condition')
    if spec['id'].startswith('hybrid_supervision_'):
        if getattr(bundle,'hybrid_supervision',None) is None:raise ValueError('Hybrid supervision needs actual512RGB/TRAINfine masks/Top16/DLA34 protocol')
        bundle.check(deep=True)
        return
    if getattr(bundle,'hybrid_supervision',None) is not None:raise ValueError('Hybrid fine-supervision raw data need their explicit complete hybrid pipeline')
    if spec["id"].startswith("mggmmil_"):
        from .mggmmil_data import GraphInputs
        GraphInputs(bundle).verify(deep=True)
        return
    if spec['id'].startswith('macnet_'):
        from .macnet_driver import PROFILES
        if getattr(bundle,'macnet',None) is None:raise ValueError('MacNet requires complete raw RGB and actual annotated guides; frozen768D features cannot substitute')
        if PROFILES[spec['id']]['condition']!=bundle.macnet.condition:raise ValueError('Actual public5/LSCC81/CSCC64 MacNet bag and guide conditions differ')
        bundle.macnet.verify_bytes()
        return
    if getattr(bundle,'macnet',None) is not None:raise ValueError('MacNet raw RGB/guidance data require their explicit complete end-to-end protocol')
    if spec['id'].startswith('protomil_'):
        if getattr(bundle,'protomil',None) is None:raise ValueError('ProtoMIL needs actual CONCH features, TRAIN-only SAE sources and region-annotated concept probes; run prepare-protomil')
        bundle.protomil.verify_bytes()
        return
    if getattr(bundle,'protomil',None) is not None:raise ValueError('ProtoMIL prepared concept data require the explicit complete SAE/concept/MIL protocol')
    if spec['id'].startswith('kmpmil_'):
        if getattr(bundle,'kmpmil',None) is None:raise ValueError('KMP-MIL needs actual CONCH/text/semantic descriptor/patient/task bindings')
        bundle.kmpmil.verify_bytes()
        return
    if getattr(bundle,'kmpmil',None) is not None:raise ValueError('KMP-MIL tasks require their explicit complete continual calibration protocol')
    if spec['id'].startswith('qpmilvl_'):
        if getattr(bundle,'qpmilvl',None) is None:raise ValueError('QPMIL-VL needs actual CONCH/text/patient/task bindings')
        bundle.qpmilvl.verify_bytes()
        return
    if getattr(bundle,'qpmilvl',None) is not None:raise ValueError('QPMIL-VL tasks require their explicit complete continual prompt protocol')
    if spec['id'].startswith('akdpmp_'):
        if getattr(bundle,'akdpmp',None) is None:raise ValueError('AKD-PMP needs prepared actual ordered tasks and original patch IDs')
        bundle.akdpmp.verify_bytes()
        return
    if getattr(bundle,'akdpmp',None) is not None:raise ValueError('AKD-PMP continual tasks require their explicit complete replay/distillation protocol')
    if spec['id'].startswith('comel_'):
        if getattr(bundle,'comel',None) is None:raise ValueError('CoMEL needs prepared ordered binary tasks, actual frozen features and patch IDs')
        bundle.comel.verify_bytes()
        return
    if getattr(bundle,'comel',None) is not None:raise ValueError('CoMEL continual task data require their explicit complete continual protocol')
    if spec['id']=='pmil_prototypical_raw_patient_reference':
        if getattr(bundle,'pmil_prototypical',None) is None:
            raise ValueError('PMIL needs actual prepared RGB256 and original ResNet34 weights')
        bundle.check(deep=True)
        return
    if getattr(bundle,'pmil_prototypical',None) is not None:
        raise ValueError('PMIL raw RGB256 images cannot replace frozen FM features or another raw-input contract')
    if spec['id'].startswith('icmil_coupled_'):
        if getattr(bundle,'icmil_coupled',None) is None:
            raise ValueError('Coupled ICMIL needs actual prepared RGB256 bags and original ImageNet weights')
        bundle.check(deep=True)
        return
    if getattr(bundle,'icmil_coupled',None) is not None:
        raise ValueError('Coupled ICMIL actual raw RGB256 patches cannot replace frozen FM features or another raw-input contract')
    if spec['id'].startswith('icmil_context_'):
        from .icmil_context_data import ContextInputs
        ContextInputs(bundle)
    if spec['id'].startswith('attention2majority_'):
        from .attention2majority_data import AttentionInputs
        from .attention2majority_driver import PROFILES
        assets=AttentionInputs(bundle)
        if assets.classes!=PROFILES[spec['id']]['classes']:
            raise ValueError('Attention2majority binary and three-grade source conditions require their native task classes')
        return
    if 'attention2majority_raw' in bundle.encoder:
        raise ValueError('Attention2majority native RGB bags cannot replace frozen FM features or another method raw-input protocol')
    if spec['id'].startswith('nciemil_') and getattr(bundle,'nciemil',None) is None:
        raise ValueError('NcIEMIL requires actual TRAIN-only Swin filtering/extraction and native highK/lowK cache; run prepare-nciemil')
    if spec['id'].startswith('taxomil_') and (getattr(bundle,'taxomil',None) is None or bundle.dim!=1024):
        raise ValueError('TaxoMIL needs explicit GPT2 assets, coarse/fine taxonomy and source-native1024D bags')
    if spec['id'].startswith('gmmamba_'):
        from .spatial_attention import spatial_metadata
        spatial_metadata(bundle)
        if (bundle.task_type!='classification' or len(bundle.classes) not in {2,3}
                or bundle.config.get('training_unit','slide')!='slide'
                or bundle.config.get('evaluation_unit','slide')!='slide'):
            raise ValueError('GMMamba needs original binary/three-class slide training and source slide-level selection')
        if spec['id']=='gmmamba_mambamil_bimamba_source' and bundle.dim!=512:
            raise ValueError('Released GMMamba requires512D features; other FMs need explicit FM-projection repair')
    if spec['id'].startswith('muse_') and getattr(bundle, 'muse', None) is None:
        raise ValueError('MUSE requires full original knowledge, text checkpoint and bound source feature inputs')
    if spec['id'].startswith('libramil_') and getattr(bundle, 'libramil', None) is None:
        raise ValueError('Libra-MIL requires bound actual text and native512D feature inputs')
    if spec['id'].startswith('m4_') and (bundle.task_type != 'multilabel'
            or bundle.config.get('training_unit','slide') != 'slide'
            or bundle.config.get('evaluation_unit','slide') != 'slide'):
        raise ValueError('M4 requires complete binary endpoint labels, original slide training/evaluation and patient-grouped splits')
    if spec['id'].startswith('convl_') and getattr(bundle, 'convl', None) is None:
        raise ValueError('ConVL requires actual native512 CONCH patient survival inputs and original concept/bin assets in encoder.convl.manifest')
    if spec['id'].startswith('center_embeddings_') and getattr(bundle, 'center_embeddings', None) is None:
        raise ValueError('Center Embeddings requires original measured256RGB patch bags, historical assets and encoder.center_embeddings.manifest')
    if spec['id'].startswith('miladapter_') and getattr(bundle, 'miladapter', None) is None:
        raise ValueError('MIL-Adapter requires actual author text prototypes and TRAIN-only support pool in encoder.miladapter.manifest')
    if spec['id'].startswith('conmil_') and getattr(bundle, 'conmil', None) is None:
        raise ValueError('ConMIL requires actual native CONCH text, aligned visual features and TRAIN-only prompt provenance in encoder.conmil.manifest')
    if spec['id'].startswith('sparseconvmil_raw_') and getattr(bundle, 'sparseconvmil', None) is None:
        raise ValueError('SparseConvMIL needs actual spatial RGB bags in encoder.sparseconvmil.manifest')
    if getattr(bundle, 'sparseconvmil', None) is not None and not spec['id'].startswith('sparseconvmil_raw_'):
        raise ValueError('SparseConvMIL raw RGB images cannot be substituted for frozen FM features')
    if spec['id'].startswith('c2c_cluster_') and getattr(bundle, 'c2c', None) is None:
        raise ValueError('C2C requires actual RGB patch bags and original ResNet18 initialization in encoder.c2c.manifest')
    if getattr(bundle, 'c2c', None) is not None and not spec['id'].startswith('c2c_cluster_'):
        raise ValueError('C2C original RGB patches cannot enter a method expecting frozen FM feature tensors')
    if spec['id'].startswith('sammil_') and getattr(bundle, 'sammil', None) is None:
        raise ValueError('SAM-MIL needs actual SAM-vit-H regions, original areas and patch membership in encoder.sammil.manifest')
    if spec['id'].startswith('lrmil_') and (getattr(bundle, 'lrmil', None) is None or bundle.dim != 768):
        raise ValueError('LRMIL requires actual native768 CONCH teacher intermediates and measured LR RGB in encoder.lrmil.manifest')
    if spec['id'].startswith('hdmil_') and bundle.hdmil is None:
        raise ValueError('HDMIL needs registered measured low-RGB paired to high-resolution frozen features')
    if spec['id'].startswith('kmgmamba_') and bundle.kmgmamba is None:
        raise ValueError('KMG-Mamba requires actual measured dual-scale features in encoder.kmgmamba.manifest')
    if spec['id'].startswith('mgtrans_') and bundle.mgtrans is None:
        raise ValueError('MG-Trans requires actual measured dual-scale alignment in encoder.mgtrans.manifest')
    if spec['id'].startswith('zoommil_') and bundle.zoommil is None:
        raise ValueError('ZoomMIL requires actual complete three-scale assets in encoder.zoommil.manifest')
    if 'augdiff_trained_dae' in spec.get('required_inputs', []) and bundle.augdiff is None:
        raise ValueError('AugDiff requires encoder.augdiff.manifest bound to a complete actual seven-view DAE pretraining run')
    if spec['id'].startswith('vlsa_') and bundle.vlsa is None:
        raise ValueError('VLSA requires native512 aligned CONCH patient features and explicit prognostic/rank text in encoder.vlsa')
    if 'slip_native_clip_prompts' in spec.get('required_inputs',[]) and bundle.slip is None:
        raise ValueError('SLIP requires actual native CLIP feature identity and ordered prompt manifest in encoder.slip')
    if spec['id'].startswith('dispro_') and getattr(bundle,'dispro',None) is None:
        raise ValueError('DisPro requires explicit incomplete-modality WSI/RNA patient inputs')
    if getattr(bundle,'dispro',None) is not None and not spec['id'].startswith('dispro_'):
        raise ValueError('DisPro incomplete-modality patients cannot enter a method that requires complete frozen inputs')
    if spec['id'].startswith('amiml_') and bundle.config.get('training_unit') != 'patient':
        raise ValueError('AMIML requires true patient bags, not independent slide labels')
    if spec['id'].startswith('dtmil_'):
        from .spatial_attention import spatial_metadata
        spatial_metadata(bundle)
        if bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('DT-MIL requires individual slide coordinate frames')
        if spec['id'] == 'dtmil_source_crop_reference_train' and len(bundle.classes) != 2:
            raise ValueError('DT-MIL source-crop recipe retains binary source labels; use repaired reference for C-way tasks')
    if spec['id'].startswith('pamil_policy_'):
        order = bundle.encoder.get('row_order_provenance')
        if not isinstance(order, str) or not order.strip():
            raise ValueError('PAMIL policy needs actual encoder.row_order_provenance for interpolation')
    if 'focus_assets' in spec.get('required_inputs', []) and bundle.focus is None:
        raise ValueError('FOCUS requires actual pretrained CONCH, ordered prompts and stable feature row assets in encoder.focus.manifest')
    if 'codmil_assets' in spec.get('required_inputs', []) and bundle.codmil is None:
        raise ValueError('CoD requires bound actual physical scales/maps and ordered text in encoder.codmil.manifest')
    if spec['id'].startswith('mamba2dmil_'):
        from .mamba2d_data import PhysicalGrid
        grid = PhysicalGrid(bundle)
        for row in bundle.rows.to_dict('records'):grid.inputs(row)
    if ('raw_regions' in spec.get('required_inputs', [])) != (bundle.raw_regions is not None):
        raise ValueError('Method input contract requires streamed real WSI regions or frozen/whole-level inputs, without substitution')
    if ('raw_images' in spec.get('required_inputs', [])) != (bundle.raw_images is not None):
        raise ValueError('Method input contract requires either full raw RGB images or frozen features, without substitution')
    if bundle.task_type not in spec['tasks']:
        raise ValueError(f"{spec['id']} does not support task {bundle.task_type}")
    if 'training_instance_labels' in spec.get('required_inputs', []) and bundle.pamil is None:
        raise ValueError('PAMIL prototype requires frozen TRAIN instance annotations in encoder.pamil')
    if 'smmile_assets' in spec.get('required_inputs', []) and bundle.smmile is None:
        raise ValueError('SMMILe requires frozen physical superpixels in encoder.smmile')
    if 'ttis_assets' in spec.get('required_inputs', []) and bundle.ttis is None:
        raise ValueError('TTIS requires prepared physical clusters and semantic assets in encoder.ttis')
    if 'pathexpert_assets' in spec.get('required_inputs', []) and bundle.simil is None:
        raise ValueError('SI-MIL requires real aligned PathExpert descriptors in encoder.simil.manifest')
    if 'hipt_pyramid' in spec.get('required_inputs', []) and bundle.hipt is None:
        raise ValueError('HIPT requires real complete physical pyramids and actual pretrained teacher weights in encoder.hipt.manifest')
    if 'deepgraphsurv_assets' in spec.get('required_inputs', []) and bundle.deepgraphsurv is None:
        raise ValueError('DeepGraphSurv requires its actual original feature graph assets in encoder.deepgraphsurv.manifest')
    if 'triagemil_assets' in spec.get('required_inputs', []) and bundle.triagemil is None:
        raise ValueError('TRIAGE-MIL requires actual MASS-selected indices and semantic hypergraphs in encoder.triagemil.manifest')
    if 'c2mil_assets' in spec.get('required_inputs', []) and bundle.c2mil is None:
        raise ValueError('C²MIL requires real graph/thumbnail and TRAIN reference assets in encoder.c2mil.manifest')
    if 'gatmamba_assets' in spec.get('required_inputs', []) and bundle.gatmamba is None:
        raise ValueError('GAT-Mamba requires real physical/subtype graph assets in encoder.gatmamba.manifest')
    if 'protosurv_assets' in spec.get('required_inputs', []) and bundle.protosurv is None:
        raise ValueError('ProtoSurv requires bound physical graphs and category/prototype assets in encoder.protosurv.manifest')
    if 'h2mil_hierarchy' in spec.get('required_inputs',[]) and bundle.h2mil is None:
        raise ValueError('H2-MIL requires actual thumbnail/coarse/fine features and physical parents in encoder.h2mil.manifest')
    if spec['id'].startswith('mfemil_'):
        if bundle.task_type == 'survival':
            bundle.check(deep=True, require_coords=True)
        elif bundle.config.get('training_unit', 'slide') != 'slide':
            raise ValueError('Classification MFE-MIL requires slide-bag training')
    if 'dual_scale' in spec.get('required_inputs', []) and bundle.dual_scale is None:
        raise ValueError('ReCo-MIL dual profiles require explicit high/low scale feature indices')
    if spec['id'].startswith('vilamil') and bundle.dual_scale is not None and bundle.dual_scale.dim != bundle.dim:
        raise ValueError('ViLa high/low features must share the registered image/text dimension')
    if 'tissue_prototypes' in spec.get('required_inputs', []) and bundle.pamoe is None:
        raise ValueError('PAMoE requires explicit tissue prototypes in encoder.pamoe')
    if bundle.omics is not None and bundle.omics.normalization == 'train_global_minmax_preserve_zero' and spec['id'] != 'pibd_source':
        raise ValueError('train_global_minmax_preserve_zero is currently supported only by PIBD')
    if 'regions4x4' in spec.get('required_inputs', []) and bundle.regions is None:
        raise ValueError('AdvMIL requires explicit physical 4x4 regions')
    if 'adversarial_training' in bundle.config and not spec['id'].startswith('advmil_'):
        raise ValueError('adversarial_training label visibility is only implemented for AdvMIL')
    if 'hierarchical_labels' in spec.get('required_inputs', []) and bundle.hierarchy is None:
        raise ValueError('HMIL requires explicit hierarchy in task.yaml')
    if 'concept_anchors' in spec.get('required_inputs', []) and (bundle.vision_language is None or bundle.vision_language.kind != 'concept_anchors'):
        raise ValueError('CATE requires explicit concept_anchors')
    if 'learned_dual_scale_prompts' in spec.get('required_inputs', []) and (bundle.vision_language is None or bundle.vision_language.kind != 'learned_dual_scale_prompts' or bundle.dual_scale is None):
        raise ValueError('ViLa requires learned_dual_scale_prompts and dual scale features')
    if 'per_slide_coords'  in spec.get('required_inputs', []):
        bundle.check(deep=True, require_coords=True)
    if spec['id'] == 'scmil' and bundle.config.get('time_unit') not in {'months', 'days'}:
        raise ValueError('SCMIL task must declare time_unit: months or days')
    if 'slide_canvas' in spec.get('required_inputs', []):
        from .bayesian import check_bayes_bundle
        check_bayes_bundle(bundle)
    if 'spatial_metadata' in spec.get('required_inputs', []):
        from .spatial_attention import spatial_metadata
        spatial_metadata(bundle)
    if spec.get('class_count') is not None and len(bundle.classes) != spec['class_count']:
        raise ValueError(f"{spec['id']} requires {spec['class_count']} classes")
    for kind in ('aligned_centers', 'quadtree'):
        if kind in spec.get('required_inputs', []) and (bundle.multiscale is None or bundle.multiscale.kind != kind):
            raise ValueError(f"{spec['id']} requires multiscale kind {kind}")
    if ('single_slide_bag' in spec.get('required_inputs', []) and
            bundle.config.get('training_unit', 'slide') == 'patient' and
            (bundle.rows.groupby('global_patient_id').size() > 1).any()):
        raise ValueError(f"{spec['id']} requires individual slide bags; aggregate probabilities for patient evaluation")
    if 'regression_targets' in spec.get('required_inputs', []) and bundle.regression_targets is None:
        raise ValueError(f"{spec['id']} requires explicit regression targets")
    if 'omics' in spec.get('required_inputs', []) and bundle.omics is None:
        raise ValueError(f"{spec['id']} requires patient-aligned omics in task.yaml")
    if 'pathways' in spec.get('required_inputs', []) and (bundle.omics is None or not bundle.omics.groups):
        raise ValueError(f"{spec['id']} requires explicit pathway/signature definitions")
    if spec['id'] in {'mcat', 'cmta', 'motcat', 'motcat_micro'} and len(bundle.omics.groups) != 6:
        raise ValueError(f"{spec['id']} requires exactly six molecular signatures")
    if 'spatial_graph' in spec.get('required_inputs', []):
        bundle.check(deep=True, require_coords=True)
    if 'coords' in spec.get('required_inputs', []):
        if bundle.config.get('training_unit', 'slide') == 'patient':
            counts = {}
            for row in bundle.records('train'):
                key = row['global_patient_id']
                counts[key] = counts.get(key, 0) + 1
            if any(n > 1 for n in counts.values()):
                raise ValueError('Spatial training cannot merge coordinate frames from different slides')
        bundle.check(deep=True, require_coords=True)


def coverage():
    from collections import Counter
    records, entries = methods(), profiles()
    families = {p["family"] for p in entries}
    return {"planned_records": len(records), "implemented_families": len(families),
            "execution_profiles": len(entries), "pending_families": len(records) - len(families),
            "profile_validation": dict(Counter(p['status'] for p in entries)),
            "groups": {group: {'planned':sum(r['group']==group for r in records),
                               'adapted':sum(r['group']==group and r['id'] in families for r in records)}
                       for group in sorted({r['group'] for r in records})},
            "paper_reproductions": sum(r["paper_reproduction"] != "not_run" for r in records),
            "pending_ids": [r["id"] for r in records if r["id"] not in families],
            "unsupported_variants": {r["id"]: r["unsupported_variants"] for r in records
                                     if r.get("unsupported_variants")},
            "note": "Profiles include variants. Registration, executable adaptation and paper reproduction are separate."}
