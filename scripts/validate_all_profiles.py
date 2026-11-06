"""Run a small synthetic experiment for each supported profile; record genuine execution status."""
import argparse
import json
import os
from pathlib import Path

from milplus.demo import create_demo, create_dual_scale_demo, create_hierarchy_demo, create_region_demo
from milplus.engine import evaluate, train
from milplus.io import digest_file, read_json, write_json
from milplus.provenance import software_identity
from milplus.registry import profiles


def retain_metadata(root,run):
    """Prune only weights generated inside this new, completed synthetic verification run.

    Call after evaluate has actually reloaded the selected checkpoint and exported
    predictions. This is an explicit storage mode, never a training default.
    """
    root,run=Path(root).resolve(),Path(run)
    if run.is_symlink() or run.resolve().parent!=root or any(p.is_symlink() for p in run.rglob('*')):
        raise ValueError('Metadata retention requires a direct run owned by the fresh verification directory')
    request=read_json(run/'request.json');status=read_json(run/'status.json')
    if request.get('synthetic') is not True or status.get('state')!='completed':
        raise ValueError('Metadata retention is restricted to completed synthetic verification runs')
    if not (run/'test_predictions.csv').is_file() or not (run/'test_metrics.json').is_file():
        raise ValueError('Actual selected-checkpoint evaluation must precede metadata retention')
    candidates=sorted(p for p in run.rglob('*') if p.is_file() and (p.suffix in {'.pt','.pth'}
        or (p.parent.name=='checkpoints' and (p.suffix=='.index' or '.data-' in p.name))))
    receipt={'mode':'metadata_only','synthetic':True,'evaluation_completed_before_pruning':True,
             'weights':[{'path':str(p.relative_to(run)),'bytes':p.stat().st_size,'sha256':digest_file(p)} for p in candidates],
             'limitation':'Weights intentionally not retained; further inference/resume requires a new training run'}
    write_json(run/'weight_receipts.json',receipt)
    for path in candidates:path.unlink()
    return {'pruned_weight_bytes':sum(p['bytes'] for p in receipt['weights']),
            'weight_receipts':str((run/'weight_receipts.json').relative_to(root))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--profile',action='append',help='Validate only these explicit IDs; repeatable, no evidence reuse')
    p.add_argument('--metadata-only',action='store_true',help='After actual reload/evaluation, retain hashes and metadata in place of synthetic weights')
    p.add_argument('--convl-native-receipt', help='Actual passed current-wheel full CUDA receipt; required to verify CUDA-only ConVL FP16 while the remaining profile matrix runs on CPU')
    p.add_argument('--taxomil-assets',help='Complete actual pinned GPT2 assets; also MILPLUS_TAXOMIL_ASSETS')
    p.add_argument('--nciemil-data',help='Complete actual TRAIN-only Swin selected-bag cache from prepare-nciemil; also MILPLUS_NCIEMIL_SELECTED_BUNDLE')
    p.add_argument('--attention2majority-weights',help='Full original Keras no-top ImageNet weights; also MILPLUS_ATTENTION2MAJORITY_WEIGHTS')
    p.add_argument('--pmil-prototypical-weights',help='Actual original ResNet34 ImageNet weights; also MILPLUS_PMIL_WEIGHTS')
    p.add_argument('--qpmilvl-weights',help='Actual full CONCH checkpoint; also MILPLUS_QPMILVL_WEIGHTS')
    p.add_argument('--qpmilvl-sha256',help='Actual CONCH checkpoint SHA256; also MILPLUS_QPMILVL_SHA256')
    p.add_argument('--macnet-weights',help='Actual full official Swin-T ImageNet22K checkpoint; also MILPLUS_MACNET_WEIGHTS')
    p.add_argument('--macnet-native-receipt',help='Passed current-wheel all6full100epoch CUDA evidence when the remaining matrix runs on CPU')
    p.add_argument('--kmpmil-weights',help='Actual full CONCH checkpoint; also MILPLUS_KMPMIL_WEIGHTS')
    p.add_argument('--kmpmil-sha256',help='Actual CONCH checkpoint SHA256; also MILPLUS_KMPMIL_SHA256')
    args=p.parse_args()
    selected={s['id'] for s in profiles()}
    if args.profile:
        unknown=set(args.profile)-selected
        if unknown:raise ValueError('Unknown validation profiles: '+str(sorted(unknown)))
        selected=set(args.profile)
    macnet_native=None
    if any(mid.startswith('macnet_') for mid in selected) and args.device=='cpu':
        from importlib.util import module_from_spec,spec_from_file_location
        info=spec_from_file_location('_milplus_macnet_native_receipt',Path(__file__).with_name('macnet_native_receipt.py'));helper=module_from_spec(info);info.loader.exec_module(helper)
        receipt=args.macnet_native_receipt or os.environ.get('MILPLUS_MACNET_NATIVE_RECEIPT')
        if not receipt:raise ValueError('MacNet needs actual complete raw CUDA training; provide passed current independent-wheel all6native receipt for CPU matrix accounting or run selected profiles on CUDA. No CPU/frozen-feature fallback.')
        macnet_native=helper.verify_native_receipt(receipt,software_identity())
    convl_native = None
    if 'convl_yaml_fp16_source_repaired' in selected and args.device == 'cpu':
        from importlib.util import module_from_spec, spec_from_file_location
        helper_spec = spec_from_file_location('_milplus_convl_native_receipt', Path(__file__).with_name('convl_native_receipt.py'))
        helper = module_from_spec(helper_spec); helper_spec.loader.exec_module(helper)
        verify_native_receipt = helper.verify_native_receipt
        receipt = args.convl_native_receipt or os.environ.get('MILPLUS_CONVL_NATIVE_RECEIPT')
        if not receipt:
            raise ValueError('ConVL source FP16 requires actual CUDA; provide its passed current-wheel full native receipt, or run on actual CUDA. No CPU precision fallback is permitted.')
        convl_native = verify_native_receipt(receipt, software_identity())
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    classification=create_demo(root/'classification',patients=20)
    context_data=None
    if any(mid.startswith('icmil_context_') for mid in selected):
        import yaml
        from milplus.icmil_context_prepare import prepare as prepare_context
        original=create_demo(root/'icmil_context_input',patients=20,dimension=32)
        task=yaml.safe_load((original/'task.yaml').read_text());task['evaluation_unit']='slide'
        (original/'task.yaml').write_text(yaml.safe_dump(task))
        context_data=prepare_context(original,root/'icmil_context_prepared',10,'train_pca25_reference')
    macnet_data={}
    if any(mid.startswith('macnet_') for mid in selected) and macnet_native is None:
        from milplus.macnet_demo import create_demo as create_mac_demo
        from milplus.macnet_driver import PROFILES as MAC_RECIPES
        weights=args.macnet_weights or os.environ.get('MILPLUS_MACNET_WEIGHTS')
        if not weights:raise ValueError('MacNet requires actual original-compatible22K Swin-T weights and complete raw/guidance data')
        for condition in sorted({MAC_RECIPES[mid]['condition'] for mid in selected if mid.startswith('macnet_')}):
            macnet_data[condition]=create_mac_demo(root/('macnet_'+condition),weights,condition)
    protomil_data=None
    if any(mid.startswith('protomil_') for mid in selected):
        from milplus.protomil_demo import create_demo as create_proto_demo
        protomil_data=create_proto_demo(root/'protomil_input')
    kmpmil_data=None
    if any(mid.startswith('kmpmil_') for mid in selected):
        from milplus.kmpmil_demo import create_demo as create_kmp_demo
        weights=args.kmpmil_weights or os.environ.get('MILPLUS_KMPMIL_WEIGHTS')
        sha=args.kmpmil_sha256 or os.environ.get('MILPLUS_KMPMIL_SHA256')
        if not weights or not sha:raise ValueError('KMP-MIL needs actual full CONCH checkpoint and SHA256; no random text substitute')
        kmpmil_data=create_kmp_demo(root/'kmpmil_input',weights,sha)
    qpmilvl_data=None
    if any(mid.startswith('qpmilvl_') for mid in selected):
        from milplus.qpmilvl_demo import create_demo as create_qpmil_demo
        weights=args.qpmilvl_weights or os.environ.get('MILPLUS_QPMILVL_WEIGHTS')
        sha=args.qpmilvl_sha256 or os.environ.get('MILPLUS_QPMILVL_SHA256')
        if not weights or not sha:raise ValueError('QPMIL-VL needs actual pretrained full CONCH weights and their SHA256; no random/cached text substitute')
        qpmilvl_data=create_qpmil_demo(root/'qpmilvl_input',weights,sha)
    akdpmp_data={}
    if any(mid.startswith('akdpmp_') for mid in selected):
        from milplus.akdpmp_demo import create_demo as create_akd_demo
        for preset in ['tcga','skin']:
            akdpmp_data[preset]=create_akd_demo(root/('akdpmp_'+preset),preset=preset)
    comel_data=None
    if any(mid.startswith('comel_') for mid in selected):
        from milplus.comel_demo import create_demo as create_comel_demo
        comel_data=create_comel_demo(root/'comel_input',tasks=3,patches=17,dimension=1024)
    pmil_prototypical_data=None
    if 'pmil_prototypical_raw_patient_reference' in selected:
        from milplus.pmil_prototypical_demo import create_demo as create_pmil_demo
        weights=args.pmil_prototypical_weights or os.environ.get('MILPLUS_PMIL_WEIGHTS')
        if not weights:raise ValueError('Prototypical PMIL needs actual original ResNet34 weights and full raw max-MIL/AP/dual-loss stages')
        pmil_prototypical_data=create_pmil_demo(root/'pmil_prototypical_input',weights)
    coupled_data=None
    if any(mid.startswith('icmil_coupled_') for mid in selected):
        from milplus.icmil_coupled_demo import create_demo as create_coupled_demo
        weights=os.environ.get('MILPLUS_ICMIL_COUPLED_WEIGHTS')
        if not weights:raise ValueError('Coupled ICMIL software probe needs the actual full original ImageNet weights')
        coupled_data=create_coupled_demo(root/'icmil_coupled',weights)
    attention2majority_data={}
    if any(mid.startswith('attention2majority_') for mid in selected):
        from milplus.attention2majority_demo import create_attention2majority_demo
        weights=args.attention2majority_weights or os.environ.get('MILPLUS_ATTENTION2MAJORITY_WEIGHTS')
        if not weights:raise ValueError('Attention2majority software gate needs the actual original ImageNet weights and complete native RGB/CNN/filter/MIL stages')
        for classes in sorted({3 if 'kidney' in mid else 2 for mid in selected if mid.startswith('attention2majority_')}):
            attention2majority_data[classes]=create_attention2majority_demo(root/('attention2majority_'+str(classes)+'class'),weights,classes)
    nciemil_data=None
    if any(mid.startswith('nciemil_') for mid in selected):
        from milplus.data import Bundle
        nciemil_data=args.nciemil_data or os.environ.get('MILPLUS_NCIEMIL_SELECTED_BUNDLE')
        if not nciemil_data:raise ValueError('NcIEMIL needs actual task-trained Swin filtering/selection cache, not synthetic random frozen features')
        actual_bundle=Bundle(nciemil_data)
        if actual_bundle.nciemil is None:raise ValueError('NcIEMIL complete filtering provenance is missing')
    taxomil_data = {}
    if any(mid.startswith('taxomil_') for mid in selected):
        from milplus.taxomil_demo import create_taxomil_demo
        assets=args.taxomil_assets or os.environ.get('MILPLUS_TAXOMIL_ASSETS')
        if not assets:raise ValueError('TaxoMIL requires actual complete GPT2 assets, no small/random decoder fallback')
        for mid in sorted(mid for mid in selected if mid.startswith('taxomil_')):
            dataset='PANDA' if 'paper_lr' in mid else 'BRACS'
            taxomil_data[mid]=create_taxomil_demo(root/('input_'+mid),assets,dataset,patients=36 if dataset=='PANDA' else 42)
    gmmamba_data = {}
    if any(mid.startswith('gmmamba_') for mid in selected):
        from milplus.gmmamba_demo import create_gmmamba_demo
        for mid in sorted(mid for mid in selected if mid.startswith('gmmamba_')):
            dimension=512 if mid=='gmmamba_mambamil_bimamba_source' else 1024
            classes=2 if dimension==512 else 3
            gmmamba_data[mid]=create_gmmamba_demo(root/('input_'+mid),patients=3*classes,dimension=dimension,classes=classes)
    muse_data = None
    if 'muse_released_source' in selected:
        from milplus.muse_demo import create_muse_demo
        from milplus.conmil_text import default_weights
        muse_data = create_muse_demo(root/'muse_input', default_weights(), patients=6, patches=257)
    libramil_data = None
    if any(mid.startswith('libramil_') for mid in selected):
        from milplus.libramil_demo import create_libramil_demo
        from milplus.conmil_text import default_weights
        libramil_data = create_libramil_demo(root/'libramil_input', default_weights(), patients=30, patches=257)
    m4_data = None
    if 'm4_multilabel_source' in selected:
        from milplus.m4_demo import create_m4_demo
        m4_data = create_m4_demo(root/'m4_input',patients=20,dimension=2048,endpoints=10)
    convl_data = None
    if 'convl_yaml_fp32_reference' in selected or ('convl_yaml_fp16_source_repaired' in selected and convl_native is None):
        from milplus.convl_demo import create_convl_demo
        from milplus.conmil_text import default_weights
        convl_data = create_convl_demo(root/'convl_input', default_weights(), patients=20)
    hdmil_data = {}
    if any(mid.startswith(('hdmil_', 'ahdmil_')) for mid in selected):
        from milplus.hdmil_demo import create_hdmil_demo
        for classes in sorted({3 if mid in {'hdmil_rcc_source','ahdmil_rcc_source_repaired'} else 2 for mid in selected if mid.startswith(('hdmil_', 'ahdmil_'))}):
            hdmil_data[classes] = create_hdmil_demo(root/f'hdmil_input_{classes}', patients=10*classes, classes=classes)
    center_embeddings_data = None
    if any(mid.startswith('center_embeddings_') for mid in selected):
        from milplus.center_embeddings_demo import create_center_embeddings_demo
        center_embeddings_data = create_center_embeddings_demo(root/'center_embeddings_input', patients=6, patches=33)
    miladapter_data = None
    if any(mid.startswith('miladapter_') for mid in selected):
        from milplus.miladapter_demo import create_miladapter_demo
        miladapter_data = create_miladapter_demo(root/'miladapter_input', patients=20, patches=33)
    conmil_data = {}
    if any(mid.startswith('conmil_') for mid in selected):
        from milplus.conmil_demo import create_conmil_demo
        for classes in sorted({3 if mid.endswith('multiclass_repaired') else 2 for mid in selected if mid.startswith('conmil_')}):
            conmil_data[classes] = create_conmil_demo(root/f'conmil_input_{classes}', patients=5*classes, classes=classes, patches=33)
    sparseconvmil_data = {}
    if any(mid.startswith('sparseconvmil_raw_') for mid in selected):
        from milplus.sparseconvmil_demo import create_sparseconvmil_demo
        for classes in sorted({3 if mid.endswith('_c128') else 2 for mid in selected if mid.startswith('sparseconvmil_raw_')}):
            sparseconvmil_data[classes] = create_sparseconvmil_demo(root/f'sparseconvmil_input_{classes}', patients=5*classes, classes=classes)
    c2c_data = {}
    if any(mid.startswith('c2c_cluster_') for mid in selected):
        from milplus.c2c_demo import create_c2c_demo
        for classes in sorted({3 if mid.endswith('multiclass_repaired') else 2 for mid in selected if mid.startswith('c2c_cluster_')}):
            c2c_data[classes] = create_c2c_demo(root/f'c2c_input_{classes}', patients=5*classes, classes=classes, patches=64, patch_pixels=64)
    sammil_data = {}
    if any(mid.startswith('sammil_') for mid in selected):
        from milplus.sammil_demo import create_sammil_demo
        for classes in sorted({2 if mid.endswith('source_control') else 3 for mid in selected if mid.startswith('sammil_')}):
            sammil_data[classes] = create_sammil_demo(root/f'sammil_input_{classes}', patients=10*classes, dimension=1024, classes=classes, patches=32)
    lrmil_data = None
    if any(mid.startswith('lrmil_') for mid in selected):
        from milplus.lrmil_demo import create_lrmil_demo
        lrmil_data = create_lrmil_demo(root/'lrmil_input', patients=18, low_patches=8)[0]
    kmgmamba_data = {}
    if any(mid.startswith('kmgmamba_') for mid in selected):
        from milplus.kmgmamba_demo import create_kmgmamba_demo
        for classes in sorted({2 if mid == 'kmgmamba_source' else 3 for mid in selected if mid.startswith('kmgmamba_')}):
            kmgmamba_data[classes] = create_kmgmamba_demo(root/f'kmgmamba_input_{classes}', patients=10*classes, dimension=1024, classes=classes)
    mgtrans_data = None
    if 'mgtrans_source_repaired' in selected:
        from milplus.mgtrans_demo import create_mgtrans_demo
        mgtrans_data = create_mgtrans_demo(root/'mgtrans_input', patients=36, dimension=1024, classes=3, roots=16)
    zoommil_data = {}
    if any(mid.startswith('zoommil_') for mid in selected):
        from milplus.zoommil_demo import create_zoommil_demo
        for method in sorted(mid for mid in selected if mid.startswith('zoommil_')):
            geometry='crc' if method=='zoommil_crc_source_repaired' else 'bright'
            zoommil_data[method]=create_zoommil_demo(root/('input_'+method),patients=18,dimension=1024,roots=16,geometry=geometry)
    augdiff_data = None
    if any(mid.startswith('augdiff_') for mid in selected):
        import torch
        import yaml

        from milplus.augdiff_assets import export_trained_dae
        from milplus.augdiff_data import AugDiffInputs
        from milplus.augdiff_demo import create_augdiff_demo
        from milplus.augdiff_pretrain import DiffusionTrainer
        from milplus.data import Bundle
        # One actual TRAIN-only DAE per shared input condition, reused across
        # the six compatible classifier/control routes, not six different fits.
        # Native width/levels/loss/1000steps retained; this is an explicitly
        # short software preparation budget, not a historical epoch claim.
        augdiff_data = create_augdiff_demo(root/'augdiff_input', patients=18, dimension=512, instances=2)
        torch.set_num_threads(1)
        inputs = AugDiffInputs(Bundle(augdiff_data), 'augdiff_views.json')
        dae = DiffusionTrainer(inputs, epochs=1, batch_size=14)
        dae.fit(root/'augdiff_dae_preparation'); del dae
        manifest = export_trained_dae(Bundle(augdiff_data), 'augdiff_views.json',
                                      root/'augdiff_dae_preparation', augdiff_data/'dae.json')
        path = augdiff_data/'task.yaml'; task = yaml.safe_load(path.read_text())
        task['encoders']['toy']['augdiff'] = {'manifest': manifest}
        path.write_text(yaml.safe_dump(task, sort_keys=False))
    vlsa_data=None
    if 'vlsa_conch_source_repaired' in selected:
        from milplus.vlsa_demo import create_vlsa_demo
        vlsa_data=create_vlsa_demo(root/'vlsa_survival',patients=20)
    slip_data={}
    for architecture in ['clip_vit_b16','clip_rn50']:
        mid='slip_'+architecture+'_source_repaired'
        if mid in selected:
            from milplus.slip_demo import create_slip_demo
            slip_data[mid]=create_slip_demo(root/('slip_'+architecture),patients=20,architecture=architecture)
    dispro_data=None
    if any(mid.startswith('dispro_') for mid in selected):
        from milplus.dispro_demo import create_dispro_demo
        # Explicit two-pathway SOFTWARE fixture. Full331/native768/default130
        # budgets have their own independent installed-wheel admission gate.
        dispro_data=create_dispro_demo(root/'dispro_survival',patients=20,dimension=1024,pathway_sizes=[2,3])
    amiml_data = None
    if any(mid.startswith('amiml_') for mid in selected):
        from milplus.amiml_demo import create_amiml_demo
        amiml_data = create_amiml_demo(root/'amiml_classification',patients=20,dimension=16,per_cluster=12)
    dtmil_data = None
    if any(mid.startswith('dtmil_') for mid in selected):
        from milplus.dtmil_demo import create_dtmil_demo
        dtmil_data = create_dtmil_demo(root/'dtmil_classification', patients=20, dimension=32)
    pamil_policy_data = None
    if any(mid.startswith('pamil_policy_') for mid in selected):
        from milplus.pamil_policy_demo import create_pamil_policy_demo
        pamil_policy_data = create_pamil_policy_demo(root/'pamil_dynamic_policy', patients=20)
    focus_data = None
    if 'focus_source' in selected:
        from milplus.focus_demo import create_focus_demo
        focus_data = create_focus_demo(root/'focus_classification', patients=20)
    codmil_data = None
    if 'codmil_source_repaired' in selected:
        from milplus.codmil_demo import create_codmil_demo
        codmil_data = create_codmil_demo(root/'codmil_classification', patients=30, dimension=32, classes=3, text_dimension=16)
    ins_data = None
    if 'ins_frozen_source_repaired' in selected:
        from milplus.ins_demo import create_ins_demo
        ins_data = create_ins_demo(root/'ins_classification', patients=20, dimension=32)
    sclwc_paper = sclwc_source = None
    if any(mid.startswith('sclwc_') for mid in selected):
        from milplus.sclwc_demo import create_sclwc_demo
        sclwc_paper = create_sclwc_demo(root/'sclwc_paper_classification', patients=30, classes=3)
        sclwc_source = create_sclwc_demo(root/'sclwc_source_classification', patients=20, classes=2)
    deepgraphsurv_data=None
    if any(mid.startswith('deepgraphsurv_') for mid in selected):
        from milplus.deepgraphsurv_demo import create_deepgraphsurv_demo
        deepgraphsurv_data=create_deepgraphsurv_demo(root/'deepgraphsurv_survival',patients=20)
    triagemil_data=None
    if any(mid.startswith('triagemil_') for mid in selected):
        from milplus.triagemil_demo import create_triagemil_demo
        triagemil_data=create_triagemil_demo(root/'triagemil_survival',patients=20)
    c2mil_data=None
    if any(mid.startswith('c2mil_') for mid in selected):
        from milplus.c2mil_demo import create_c2mil_demo
        c2mil_data=create_c2mil_demo(root/'c2mil_survival',patients=30)
    gatmamba_data=None
    if any(mid.startswith('gatmamba_') for mid in selected):
        from milplus.gatmamba_demo import create_gatmamba_demo
        gatmamba_data=create_gatmamba_demo(root/'gatmamba_survival',patients=20)
    protosurv_data=None
    if any(mid.startswith('protosurv_') for mid in selected):
        from milplus.protosurv_demo import create_protosurv_demo
        protosurv_data=create_protosurv_demo(root/'protosurv_survival',patients=20)
    pathrwkv_data=None
    if any(mid.startswith('pathrwkv_') for mid in selected):
        from milplus.pathrwkv_demo import create_pathrwkv_demo
        pathrwkv_data=create_pathrwkv_demo(root/'pathrwkv_multitask')
    h2mil_data=None
    if any(mid.startswith('h2mil_') for mid in selected):
        from milplus.h2mil_demo import create_h2mil_demo
        h2mil_data=create_h2mil_demo(root/'h2mil_classification')
    hipt_data=None
    if any(mid.startswith('hipt_lgp_') for mid in selected):
        from milplus.hipt_demo import create_hipt_demo
        hipt_data=create_hipt_demo(root/'hipt_classification',os.environ.get(
            'MILPLUS_HIPT_WEIGHTS','/data/milplus_model_weights/hipt_vit4k_xs_teacher_backbone.pt'))
    # A selected-profile run requires only the selected methods' data/deps.
    # With the full matrix selected, every original fixture/argument is kept.
    simil_data=None
    if any(mid.startswith('simil_') for mid in selected):
        from milplus.simil_demo import create_simil_demo
        simil_data=create_simil_demo(root/'simil_classification')
    mamba2d_data=None
    if any(mid.startswith('mamba2dmil_') for mid in selected):
        from milplus.mamba2d_demo import create_mamba2d_demo
        mamba2d_data=create_mamba2d_demo(root/'mamba2d_classification')
    ttis_data=None
    if any(mid.startswith('ttis_') for mid in selected):
        from milplus.ttis_demo import create_ttis_demo
        ttis_data=create_ttis_demo(root/'ttis_classification')
    stable_cls=stable_surv=None
    if any(mid.startswith('stablemil_') for mid in selected):
        from milplus.stable_demo import create_stable_demo
        stable_cls=create_stable_demo(root/'stable_classification',patients=20,classes=2)
        stable_surv=create_stable_demo(root/'stable_survival',patients=20,classes=2,survival=True)
    monai_raw=None
    if any(mid.startswith('monai_') for mid in selected):
        from milplus.raw_demo import create_raw_demo
        monai_raw=create_raw_demo(root/'monai_raw', patients=15, classes=3)
    raw_regions=None
    if 'campanella_rnn' in selected:
        from milplus.raw_region_demo import create_raw_region_demo
        raw_regions=create_raw_region_demo(root/'campanella_raw', initial_weights=os.environ.get('MILPLUS_RESNET34_WEIGHTS', '/data/milplus_model_weights/resnet34-b627a593.pth'))
    dual_reco=None
    if any(mid.startswith('recomil_dual') for mid in selected):
        dual_reco=create_dual_scale_demo(root/'dual_reco', patients=20)
    regression=None
    if 'he2rna' in selected:
        regression=create_demo(root/'regression', patients=20, regression=True)
    survival=None
    if any(spec['id'] in selected and spec['tasks']==['survival'] for spec in profiles()):
        survival=create_demo(root/'survival',patients=20,survival=True,omics=True)
    pamoe_data=None
    if any(mid.startswith('pamoe_') for mid in selected):
        from milplus.pamoe_data import add_synthetic_prototypes
        pamoe_data=add_synthetic_prototypes(create_demo(root/'pamoe',patients=20,survival=True))
    aligned=hierarchical=None
    if 'csmil' in selected:
        aligned=create_demo(root/'aligned',patients=20,multiscale='aligned_centers')
    if 'hagmil_ce' in selected:
        hierarchical=create_demo(root/'hierarchical',patients=20,multiscale='quadtree')
    concepts=dual_text=None
    if 'cate' in selected or any(mid.startswith('vilamil') for mid in selected):
        from milplus.vision_language_demo import create_vision_demo
        if 'cate' in selected:concepts = create_vision_demo(root/'concepts', 'concept_anchors')
        if any(mid.startswith('vilamil') for mid in selected):dual_text = create_vision_demo(root/'dual_text', 'learned_dual_scale_prompts')
    hierarchy_labels=None
    if 'hmil_source_repaired' in selected:
        hierarchy_labels = create_hierarchy_demo(root/'hierarchy_labels')
    region_survival=None
    if any(mid.startswith('advmil_') for mid in selected):
        region_survival = create_region_demo(root/'region_survival', semi=True)
    pamil_data={}
    if any(mid.startswith('pamil_prototype_') for mid in selected):
        from milplus.pamil_demo import create_pamil_demo
        pamil_data = {variant: create_pamil_demo(root/('pamil_data_'+variant), multilabel=variant == 'multilabel')
                      for variant in ['classification', 'multilabel']}
    smmile_data={}
    if selected & {'smmile','smmile_binary','smmile_multilabel'}:
        from milplus.smmile_demo import create_smmile_demo
        smmile_data = {method: create_smmile_demo(root/method, variant=variant)
                       for method, variant in [('smmile_class_data','multiclass'),('smmile_binary_data','binary'),('smmile_multi_data','multilabel')]}
    result=[]
    for spec in profiles():
        mid=spec['id']
        if mid not in selected:continue
        if mid == 'convl_yaml_fp16_source_repaired' and convl_native is not None:
            result.append({'id':mid,'state':'passed','device':'cuda_verified_current_wheel_receipt',
                'synthetic':True,'execution_evidence':convl_native['path'],'execution_evidence_sha256':convl_native['sha256'],
                'all_four_full_native_conditions_verified':True,'cpu_FP32_substitution':False})
            write_json(root/'validation.json',{'profiles':result,'software':software_identity(),
                'scope':'Fresh CPU software matrix plus separately verified actual current-wheel CUDA-only profile; no paper reproduction'})
            continue
        cfg={'epochs':2,'threads':1}
        if mid not in {'mean_linear','clam_sb','clam_mb','dsmil','h2t','panther','frmil','survpath','mcat','porpoise','cbmil','cmta','motcat','motcat_micro','csmil','hagmil_ce','slidegraph'}:
            cfg['model']={'hidden':32}
        if mid.startswith('nciemil_'):
            # Full native1024D/four-head/2K graph and literal SGD. This is a
            # two-epoch SOFTWARE probe; separate100epoch full native gates
            # include actual raw Swin training/extraction and both K settings.
            cfg.update(epochs=2,threads=2,model={},lr=2e-4,weight_decay=1e-5,
                grad_clip=None,patience=2,selection_metric='nciemil_val_harmonic')
        if mid.startswith('taxomil_'):
            # Actual full pretrained >200M dualhead GPT2,1024D/257patches/all
            # four original losses. One SOFTWARE epoch only; separate four
            # complete100max/patience20 native64+remainder budgets required.
            cfg.update(epochs=1,threads=2,model={},lr=1e-4 if 'paper_lr' in mid else 1e-5,
                weight_decay=.01,grad_clip=None,patience=20,selection_metric='taxomil_score')
        if mid.startswith('gmmamba_'):
            # All native512D/16state/10groups/257patches,including the literal
            # first selection at epoch_index6. Short SOFTWARE7epoch scope;
            # independent full100max/epoch>50 budgets remain mandatory.
            cfg.update(epochs=7,threads=2,model={},lr=.0001,weight_decay=.00001,
                       grad_clip=None,patience=20,selection_metric='gmmamba_val_auc_f1')
        if mid == 'muse_released_source':
            # Complete89M/12-layer text/all600 original descriptions,257patches
            # and20 distinct updates per TRAIN visit. One SOFTWARE epoch only;
            # separate native200max/patience20/epoch>80 budgets are mandatory.
            cfg.update(epochs=1,model={},lr=.0001,weight_decay=.00001,
                       grad_clip=None,patience=20,selection_metric='muse_val_error')
        if mid.startswith('libramil_'):
            cfg.update(lr=.0001, weight_decay=.01, grad_clip=None,
                       selection_metric='libramil_val_accuracy', model={})
        if mid.startswith('m4_'):
            cfg.update(epochs=2,model={})  # Full2048/512/five experts/ten endpoints; separate full32epoch native budgets.
        if mid.startswith('convl_'):
            cfg.update(epochs=1,model={})  # Actual full112M network, complete patient bags; software budget only, separate full200-max CUDA gate.
        if mid.startswith('vlsa_'):
            # Native full text and original patient-window32. Small patient
            # fixture only exercises a tail window; independent60/80-patient
            # installed-wheel10epoch gates require full32 plus actual tail.
            cfg.update(epochs=2,model={})
        if mid.startswith('ahdmil_'):
            cfg.update(epochs=5, model={'dmin_epochs':2,'joint_epochs':3,'dmin_selection_start':1})  # Native architecture SOFTWARE check; separate200+200-max budgets.
        if mid.startswith('hdmil_'):
            cfg.update(epochs=4, model={'dmin_epochs':2,'lipn_epochs':2,'dmin_selection_start':1})  # Full two-stage SOFTWARE check; separate200+200 native gate.
        if mid.startswith('center_embeddings_'):
            cfg.update(epochs=2,model={'train_batch_size':1})  # Full originalK32/256RGB/R34/FPN/three-loss software check; separate defaultbatch4/20epoch native budget.
        if mid.startswith('miladapter_'):
            cfg.update(epochs=2, lr=.001, weight_decay=1e-5, model={'k_shots':2})  # Full native512D26source conditions; separate20epoch/k8/full257patch budgets.
        if mid.startswith('conmil_'):
            cfg.update(epochs=2,model={})  # Full512/256/128/cubicKAN and actual native text; software33patch fixture, separate4097patch50epoch native gate.
        if mid.startswith('sparseconvmil_raw_'):
            cfg.update(epochs=2,model={})  # Full compiled sparse network/software64px fixture; separate512px200epoch native gate.
        if mid.startswith('c2c_cluster_'):
            cfg.update(epochs=2,model={})  # Original full pretrained ResNet18/256/64/32 and8x8/three-loss recipe; explicit64px software fixture, distinct native512px25epoch gate.
        if mid.startswith('sammil_'):
            cfg.update(epochs=2,model={})  # Full512 projection/128 attention/all area and pseudo-bag losses; separate200max native gate.
        if mid.startswith('lrmil_'):
            cfg.update(epochs=3,model={'stage_epochs':[1,1,1],'patch_batch_size':4,'inference_batch_size':4})  # Full native encoder/three actual phases; separate declared200+200+200 native gate.
        if mid.startswith('kmgmamba_'):
            cfg.update(epochs=2,model={'num_workers':0})  # Complete96width source graph; independent100max native gate required.
        if mid.startswith('mgtrans_'):
            cfg.update(epochs=2, model={'num_workers': 0})  # All source widths/512anchors and actual MI window; distinct full200max native gate.
        if mid.startswith('zoommil_'):
            cfg.update(epochs=2,model={'num_workers':0})  # full source512/256/100samples; software I/O port; separate native8-reader100epoch gate
        if mid.startswith('augdiff_'):
            cfg.update(epochs=2, model={})  # full source UNet and all classifier widths
        if mid.startswith('dispro_'):
            cfg.update(epochs=6,model={'stage_epochs':[2,2,2]})
        if mid.startswith('slip_'):
            # Native full12-layer text and exact prompts; the separate installed
            # wheel gate retains the complete original10epoch training budget.
            cfg.update(epochs=2,model={})
        if mid.startswith('amiml_'):
            cfg.update(epochs=6,model={'R':11,'batch_size':4})
        if mid.startswith('dtmil_'):
            cfg.update(epochs=2, model={'map_size': 8, 'batch_size': 4})
        if mid.startswith('pamil_policy_'):
            # Small explicit16row groups for short software regression only.
            # Independent complete300epoch gates retain full512group/policy width.
            cfg.update(epochs=3, model={'bag_size':16})
        if mid == 'focus_source':
            # Full native pretrained text,512visual and original8192budget.
            # Two-epoch profile regression is not the independent200max native gate.
            cfg.update(epochs=2, model={})
        if mid == 'codmil_source_repaired':
            # Short regression only; independent native1024/1536 runs retain
            # source200maxepochs and minimum-epoch stopping, full192gate.
            cfg.update(epochs=3, model={})
        if mid == 'ins_frozen_source_repaired':
            # Short exhaustive regression, not the independent150epoch native
            # budget. Retain full source ResNet18, B64, queue8192 and all losses.
            cfg.update(epochs=3, model={'warmup_epochs': 1})
        if mid.startswith('c2aug_'):
            cfg['model'] = {'max_views': 2}
        if mid.startswith('sclwc_'):
            cfg.update(epochs=6, model={})  # Native512/256, all losses, active WSCL after warmup.
        if mid.startswith('deepgraphsurv_'):
            cfg.update(epochs=6,model={'batch_size':4})
        if mid.startswith('triagemil_'):
            cfg['model']={'hidden':16,'heads':2,'hypergraph_layers':2}
        if mid.startswith('c2mil_'):
            cfg.update(epochs=3,model={'hidden':8,'batch_size':8})
        if mid.startswith('gatmamba_'):
            cfg['model']={'uni_hidden':8,'positional_embedding_size':4,'batch_size':4}
        if mid.startswith('protosurv_'):
            cfg['model']={'batch_size':3,'hidden':8,'prototype_dimension':24,'prototype_hidden':16,
                'prototypes_per_class':2,'prototype_iterations':1,'fusion_layers':1}
        if mid.startswith('pathrwkv_'):
            cfg['model'] = {'hidden': 128, 'max_tiles': 7, 'inference_chunk': 5,
                            'precision': '32', 'backend': 'torch'}
        if mid.startswith('agdrmil_'):
            cfg['model'] = {}  # Native512, both complete source branches and loss.
        if mid.startswith('h2mil_'):
            cfg['model']={'hidden':16,'batch_size':4}
        if mid in {'mhim_abmil','ibmil','dgrmil'} or mid.startswith(('mfcmil_', 'sdfp_')):
            cfg['model']['warmup_epochs']=1
        if mid in {'momil', 'momil_survival_extension'}:
            cfg['model'] = {'hidden': 32, 'state_size': 8, 'scan_chunk': 16}
        if mid in {'smmile', 'smmile_binary', 'smmile_multilabel'}:
            cfg['model'] = {'warmup_epochs': 1, 'groups': 2}
        if mid.startswith('pamil_prototype_'):
            cfg.update(epochs=3, model={'hidden': 32, 'attention_dim': 16, 'prototypes': 6,
                                       'warmup_epochs': 1, 'push_epochs': [2], 'last_epochs': 1})
        if mid.startswith('monai_'):
            cfg['model'] = {'tile_size': 32, 'tile_count': 4, 'trans_blocks': 1, 'batch_size': 4, 'eval_chunk': 3}
        if mid.startswith('mammoth_'):
            cfg['model'] = {'hidden': 32, 'attention_dim': 16, 'experts': 3, 'slots': 3,
                            'heads': 4, 'slot_dim': 16, 'aggregation_heads': 4,
                            'ilra_landmarks': 4, 'minimum_epochs': 1}
        if mid.startswith('pmil_shapley_'):
            cfg.update(epochs=4, model={'hidden': 32, 'attention_dim': 16, 'rounds': 2, 'round_epochs': 2,
                'max_pseudo': 2, 'pseudo_step': 1, 'search_rate': 2, 'sample_rate': 2, 'k_sample': 4})
        if mid.startswith('msattention_'):
            cfg['model'] = {}  # Preserve original 128-wide, three-depth architecture.
        if mid.startswith('ttis_'):
            cfg['model'] = {'hidden': 512 if 'rrt' in mid else 32, 'attention_dim': 16}
        if mid.startswith('mamba2dmil_'):
            cfg['model'] = {'hidden': 16, 'state_size': 4}
        if mid.startswith('simil_'):
            cfg['model'] = {}  # Native 384/246, K20, four PF-Mixer blocks.
        if mid.startswith('hipt_lgp_'):
            cfg['model'] = {}  # Actual teacher, native 384/192, complete physical 16x16 regions.
        if mid == 'campanella_rnn':
            cfg.update(epochs=12, model={'cnn_epochs': 10, 'rnn_epochs': 2, 'test_every': 10, 'cnn_batch_size': 4, 'rnn_batch_size': 4})
        if mid.startswith('stablemil_'):
            cfg['model'] = {'hidden': 32, 'heads': 4, 'depth': 2}
        if mid in {'setmil_source', 'setmil_repaired'}:
            cfg['model'] = {'map_size': 12, 'reduction': 7, 'token_dim': 8, 'embed_dim': 8, 'depth': 2, 'heads': 4, 'batch_size': 4}
        if mid in {'gtp_source', 'gtp_repaired'}:
            cfg['model'] = {'hidden': 32, 'clusters': 8, 'depth': 2, 'heads': 4, 'batch_size': 4}
        if mid.startswith('probsa_'):
            cfg['model'] = {'hidden': 32, 'attention_dim': 16, 'train_samples': 8, 'test_samples': 16, 'cycles': 2}
        if mid == 'micromil_frozen':
            cfg['model'] = {'hidden': 16, 'clusters': 4}
        if mid == 'loss_attention_classical':
            cfg['model'] = {'widths': [32, 16, 8]}
        if mid.startswith('recomil_'):
            cfg['model'] = {'hidden': 32, 'prototypes': 8, 'cluster_iterations': 5}
        if mid.startswith('mambamil_'):
            cfg['model'] = {'hidden': 16, 'layers': 2, 'rate': 5, 'scan_chunk': 16}
        if mid == 'msmmil':
            cfg['model'] = {'hidden': 16, 'layers': 2, 'rate': 10, 'scan_chunk': 16}
        if mid.startswith('mfemil_'):
            cfg['model'] = {'decoder_dim': 32, 'decoder_depth': 1, 'decoder_heads': 4}
        if mid in {'mammil', 'mammil_source', 'mammil_patient_union'}:
            cfg['model'] = {'hidden': 32, 'state_size': 8, 'scan_chunk': 16}
        if mid in {'dgmil', 'dgmil_source_extremes'}:
            cfg['model'] = {'clusters': 2}
        if mid.startswith('pamoe_'):
            cfg['model'] = {'hidden': 16, 'batch_size': 4}
        if mid.startswith('pathomics_'):
            cfg['model'] = {'hidden': 16, 'pretrain_epochs': 1, 'accumulation_steps': 3}
        if mid.startswith('mmp_'):
            cfg['model'] = {'hidden': 16, 'prototypes': 3, 'prototype_patches': 30, 'cluster_iterations': 5, 'batch_size': 4}
        if mid == 'pibd_source':
            cfg['model'] = {'hidden': 16, 'batch_size': 4, 'num_patches': 64, 'micro_bag_size': 32, 'sample_num': 3}
        if mid.startswith('advmil_'):
            cfg['model'] = {'hidden': 16, 'discriminator_dim': 8, 'batch_size': 4, 'selection_warmup': 0, 'test_samples': 4}
        if mid == 'hmil_source_repaired':
            cfg['model'] = {'hidden': 16, 'attention_dim': 8, 'batch_size': 4}
        if mid == 'tpmil':
            cfg['model'] = {'hidden': 32, 'attention_dim': 16}
        if mid == 'cdpmil_source':
            cfg['model'] = {'clusters': 3}
        if mid.startswith('rankmix_'):
            cfg['model'] = {'warmup_epochs': 1}
        if mid == 'cate':
            cfg['model'] = {'pim_weight': 1., 'sim_weight': .1}
        if mid.startswith('vilamil'):
            cfg['model'] = {'hidden': 16, 'prototypes': 3}
        if mid == 'he2rna':
            cfg['model'] = {'layers': [32, 32], 'ks': [1, 2, 4, 8], 'supertile_count': 8, 'batch_size': 4}
        if mid == 'healnet':
            cfg['model'] = {'bins': 3, 'attn_dropout': .1, 'ff_dropout': .1}
        if mid == 'camil_context':
            cfg['model'] = {'hidden': 64, 'dim_head': 8, 'landmarks': 16}
        if mid == 'retmil':
            cfg['model']['chunk_len']=16
        if mid in {'mil_dropout_abmil', 'mil_dropout_gated'}:
            cfg['model'] = {}
        if mid == 'sdfp_dsmil':
            cfg['model'].pop('hidden', None)
        if mid.startswith('packmil_'):
            cfg['model'].update(batch_size=4, pack_length=32, min_sequence=2, downsample_ratio=2)
        if mid.startswith('mico'):
            cfg['model'].update(clusters=8, enhancers=2, cluster_slides=3, cluster_iterations=5)
        if mid.startswith('efficientmil_'):
            cfg['model'].update(selected_patches=8)
        if mid == 'scmil':
            cfg['model'].update(components=5, cluster_size=8)
        if mid == 'deepattnmisl':
            cfg['model'] = {'clusters': 3, 'batch_size': 4}
        if mid.startswith('bayesmil_'):
            cfg['model'].update(attention_dim=16, mc_samples=4, weighted_sampling=True)
        if mid == 'psamil':
            cfg['model'].update(batch_size=3, attention_chunk=8)
        if mid == 'rrt':
            cfg['model']['region_num']=2
        if mid in {'psemix_abmil', 'prodiv_abmil'}:
            cfg['model'].update(pseudo_bags=3, phenotypes=2, batch_size=4)
        if mid == 'sgpmil':
            cfg['model'].update(hidden1=16, hidden2=8, attention_dim=4, inducing_points=6, mc_samples=4)
        if mid == 'ptcmil':
            cfg['model'].update(clusters=3)
        if mid == 'cbmil':
            cfg['model'] = {'selected_instances': 16, 'clusters': 3, 'batch_size': 4, 'contrastive_dim': 8}
        if mid == 'motcat_micro':
            cfg['model'] = {'micro_batch_size': 16}
        if mid == 'csmil':
            cfg['model'] = {'clusters': 3}
        if mid == 'hagmil_ce':
            cfg['model'] = {'hidden_dims': [32, 48, 32, 32], 'k_sample': 2, 'train_parents': 4, 'eval_parents': 4}
        data = concepts if mid == 'cate' else dual_text if mid.startswith('vilamil') else regression if mid == 'he2rna' else aligned if mid == 'csmil' else hierarchical if mid == 'hagmil_ce' else survival if spec['tasks']==['survival'] else classification
        if mid.startswith('nciemil_'):
            data=nciemil_data
        if mid.startswith('taxomil_'):
            data = taxomil_data[mid]
        if mid.startswith('gmmamba_'):
            data = gmmamba_data[mid]
        if mid == 'muse_released_source':
            data = muse_data
        if mid.startswith('libramil_'):
            data = libramil_data
        if mid.startswith('m4_'):
            data = m4_data
        if mid.startswith('convl_'):
            data = convl_data
        if mid.startswith('dispro_'):
            data=dispro_data
        if mid.startswith('vlsa_'):
            data=vlsa_data
        if mid.startswith('slip_'):
            data=slip_data[mid]
        if mid.startswith(('hdmil_', 'ahdmil_')):
            data = hdmil_data[3 if mid in {'hdmil_rcc_source','ahdmil_rcc_source_repaired'} else 2]
        if mid.startswith('center_embeddings_'):
            data = center_embeddings_data
        if mid.startswith('miladapter_'):
            data = miladapter_data
        if mid.startswith('conmil_'):
            data = conmil_data[3 if mid.endswith('multiclass_repaired') else 2]
        if mid.startswith('sparseconvmil_raw_'):
            data = sparseconvmil_data[3 if mid.endswith('_c128') else 2]
        if mid.startswith('c2c_cluster_'):
            data = c2c_data[3 if mid.endswith('multiclass_repaired') else 2]
        if mid.startswith('sammil_'):
            data = sammil_data[2 if mid.endswith('source_control') else 3]
        if mid.startswith('lrmil_'):
            data = lrmil_data
        if mid.startswith('kmgmamba_'):
            data = kmgmamba_data[2 if mid == 'kmgmamba_source' else 3]
        if mid.startswith('mgtrans_'):
            data = mgtrans_data
        if mid.startswith('zoommil_'):
            data = zoommil_data[mid]
        if mid.startswith('augdiff_'):
            data = augdiff_data
        if mid.startswith('sclwc_'):
            data = sclwc_source if mid.endswith('source_control') else sclwc_paper
        if mid == 'ins_frozen_source_repaired':
            data = ins_data
        if mid.startswith('amiml_'):
            data = amiml_data
        if mid.startswith('dtmil_'):
            data = dtmil_data
        if mid.startswith('pamil_policy_'):
            data = pamil_policy_data
        if mid == 'focus_source':
            data = focus_data
        if mid == 'codmil_source_repaired':
            data = codmil_data
        if mid.startswith('deepgraphsurv_'):
            data = deepgraphsurv_data
        if mid.startswith('triagemil_'):
            data = triagemil_data
        if mid.startswith('c2mil_'):
            data = c2mil_data
        if mid.startswith('gatmamba_'):
            data = gatmamba_data
        if mid.startswith('protosurv_'):
            data = protosurv_data
        if mid.startswith('pathrwkv_'):
            data = pathrwkv_data
        if mid.startswith('monai_'):
            data = monai_raw
        if mid.startswith('ttis_'):
            data = ttis_data
        if mid.startswith('mamba2dmil_'):
            data = mamba2d_data
        if mid.startswith('simil_'):
            data = simil_data
        if mid.startswith('hipt_lgp_'):
            data = hipt_data
        if mid.startswith('h2mil_'):
            data = h2mil_data
        if mid == 'campanella_rnn':
            data = raw_regions
        if mid.startswith('stablemil_'):
            data = stable_surv if 'survival' in mid else stable_cls
        if mid.startswith('recomil_dual'):
            data = dual_reco
        if mid.startswith('pamoe_'):
            data = pamoe_data
        if mid.startswith('advmil_'):
            data = region_survival
        if mid in {'smmile', 'smmile_binary', 'smmile_multilabel'}:
            key = {'smmile':'smmile_class_data','smmile_binary':'smmile_binary_data','smmile_multilabel':'smmile_multi_data'}[mid]
            data = smmile_data[key]
        if mid.startswith('pamil_prototype_'):
            data = pamil_data['multilabel' if mid.endswith('multilabel') else 'classification']
        try:
            if mid == 'hmil_source_repaired':
                data = hierarchy_labels
            if mid.startswith('attention2majority_'):
                data=attention2majority_data[3 if 'kidney' in mid else 2]
                cfg={'epochs':2,'threads':2,'model':{'discriminator_epochs':2}}
            if mid.startswith('icmil_context_'):
                data=context_data
                cfg={'threads':2}
            if mid=='pmil_prototypical_raw_patient_reference':
                data=pmil_prototypical_data
                cfg={'epochs':1,'threads':2,'model':{'encoder_epochs':1}}
            if mid.startswith('icmil_coupled_'):
                data=coupled_data
                cfg={'epochs':1,'threads':2,'model':{'iterations':1,'distill_epochs':1}}
            if mid.startswith('comel_'):
                data=comel_data
                cfg={'epochs':2,'threads':2}
            if mid.startswith('macnet_'):
                if macnet_native is not None:
                    result.append({'id':mid,'state':'passed','device':'cuda','synthetic':True,'evidence':macnet_native,
                        'scope':'Actual current installed-wheel full100epoch CUDA protocol verified; not a CPU precision/model substitute'})
                    write_json(root/'validation.json',{'profiles':result,'software':software_identity(),'scope':'synthetic software/native verification; not paper reproduction'})
                    continue
                from milplus.macnet_driver import PROFILES as MAC_RECIPES
                data=macnet_data[MAC_RECIPES[mid]['condition']];cfg={'epochs':2,'threads':2}
            if mid.startswith('protomil_'):
                data=protomil_data
                cfg={'sae_epochs':1,'mil_epochs':2,'probe_per_class':5,'threads':2}
            if mid.startswith('kmpmil_'):
                data=kmpmil_data
                cfg={'epochs':2,'threads':2}
            if mid.startswith('qpmilvl_'):
                data=qpmilvl_data
                cfg={'epochs':2,'threads':2}
            if mid.startswith('akdpmp_'):
                from milplus.akdpmp_driver import PROFILES as AKD_RECIPES
                data=akdpmp_data[AKD_RECIPES[mid]['preset']]
                cfg={'epochs':2,'threads':2}
            path=train(data,mid,root/mid,
                       config=cfg,device=args.device)
            if mid.startswith('protomil_'):
                if read_json(path/'status.json')['state']=='awaiting_concept_review':
                    request=read_json(path/'review_request.json');bank=read_json(path/'concepts/concept_bank.json');concept=bank['active_concepts'][0]
                    write_json(request['review_file'],{'schema_version':1,'sae_state_sha256':request['sae_state_sha256'],
                        'concept_bank_sha256':request['concept_bank_sha256'],'reviewer':'synthetic_validation_only',
                        'provenance':'Explicit synthetic software-test decision only; no clinical/pathologist review claimed',
                        'removed_concepts':[concept],'concept_names':{str(concept):'Synthetic test removal, no clinical interpretation'}})
                    train(data,mid,root/mid,config=cfg,device=args.device,resume=True)
            metrics=evaluate(path,device=args.device)
            retained=retain_metadata(root,path) if args.metadata_only else {'weight_retention':'all'}
            result.append({'id':mid,'state':'passed','device':args.device,'synthetic':True,'metrics':metrics,**retained})
        except Exception as e:  # noqa: BLE001 -- independent matrix failures must be recorded, not discarded
            result.append({'id':mid,'state':'failed','device':args.device,'error':repr(e)})
        write_json(root/'validation.json',{'profiles':result,'software': software_identity(),
                   'scope':'synthetic software validation; not paper reproduction'})
    print(json.dumps(result,indent=2))
    if any(r['state']!='passed' for r in result):
        raise SystemExit(1)


if __name__=='__main__':
    main()
