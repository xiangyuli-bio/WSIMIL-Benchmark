"""One independent installed-wheel DisPro130epoch/full331/native-width gate.

Run both1024 and1536 separately. No CLI switch shortens a stage or the pathway
set. Synthetic inputs are labelled permanently; this is not paper-score evidence.
"""
import argparse
import gc
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for key in ['wheel','output','source-sha','lock-sha']:p.add_argument('--'+key,required=True)
    p.add_argument('--dimension',required=True,type=int,choices=[1024,1536])
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--expected-version',default='0.65.0')
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    wheel=Path(a.wheel).resolve();wheel_sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
    installed=out/'wheel_import';installed.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        for n in archive.namelist():
            if Path(n).is_absolute() or '..' in Path(n).parts:raise ValueError('Unsafe wheel member')
        archive.extractall(installed)
    sys.path.insert(0,str(installed))
    import torch
    import milplus
    from milplus.data import Bundle
    from milplus.dispro import PHASES
    from milplus.dispro_demo import create_dispro_demo,public_pathway_sizes
    from milplus.dispro_source import backbone
    from milplus.engine import train,evaluate,units
    from milplus.io import digest_file,read_json,write_json
    from milplus.provenance import software_identity
    assert Path(milplus.__file__).is_relative_to(installed)
    assert milplus.__version__==a.expected_version
    software=software_identity()
    assert software['source_sha256']==a.source_sha and software['sources_lock_sha256']==a.lock_sha
    if a.device=='cuda':assert torch.cuda.is_available()
    started=time.monotonic()
    report={'state':'running','formal_admission':False,'synthetic':True,
        'version':a.expected_version,'software':software,'wheel':str(wheel),'wheel_sha256':wheel_sha,
        'feature_dimension':a.dimension,'device':a.device,'stage_epochs':[30,50,50],
        'scope':'Actual full12layer768BioBERT/native768two-layer331pathway SNN/fusion, original complete30/50/50 source-repaired recipe, all observed patients and missing-modality objectives. Synthetic only, no clinical scores.'}
    if a.device=='cuda':report.update(gpu=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count())
    write_json(out/'validation.json',report)
    try:
        data=create_dispro_demo(out/'inputs',patients=20,dimension=a.dimension,instances=300)
        bundle=Bundle(data);inventory=bundle.check(deep=True)
        sizes=public_pathway_sizes();assert len(sizes)==331 and sum(sizes)==9745
        assert [len(g['features']) for g in bundle.dispro.pathways]==sizes
        train_groups=units(bundle,'train');train_ids={g[0]['global_patient_id'] for g in train_groups}
        allowed=[{g for g in train_ids if bundle.dispro.observed(g,'WSI')},
                 {g for g in train_ids if bundle.dispro.observed(g,'Omics')},train_ids]
        for row in bundle.rows.to_dict('records'):
            if bundle.dispro.observed(row['global_patient_id'],'WSI'):assert len(bundle.load(row)[0])==300
        expected_updates=[len(ids)*epochs for ids,epochs in zip(allowed,[30,50,50])]
        write_json(out/'native_inputs.json',{'feature_dimension':a.dimension,'pathway_sizes':sizes,
            'train_patients_by_stage':[sorted(ids) for ids in allowed],
            'minimum_actual_observed_patches_per_slide':300,'feature_inventory':inventory['feature_inventory_sha256'],
            'expected_stage_updates':expected_updates,'declared_missingness':read_json(data/'dispro_inputs.json')})
        cfg={'epochs':130,'lr':2e-4,'weight_decay':1e-5,'grad_clip':None,'max_instances':None,
             'selection_metric':'dispro_c_index','threads':2,'model':{'stage_epochs':[30,50,50]}}
        run=train(data,'dispro_source_repaired',out/'dispro_source_repaired',device=a.device,config=cfg,seed=2065)
        status=read_json(run/'status.json');protocol=read_json(run/'survival_protocol.json')
        history=read_json(run/'history.json')
        assert status['state']=='completed' and status['completed_epochs']==130 and len(history)==130
        assert [s['name'] for s in protocol['stages']]==list(PHASES)
        assert [s['completed_epochs'] for s in protocol['stages']]==[30,50,50]
        assert [s['optimizer_updates'] for s in protocol['stages']]==expected_updates
        assert protocol['native_inputs']['ordered_pathways']==331 and protocol['native_inputs']['full_pathway_sizes']==sizes
        assert protocol['foundation']['native_hidden']==768 and protocol['foundation']['native_layers']==12
        assert protocol['bound_teachers']==protocol['teacher_best']
        assert all(history[i]['train']['dispro_phase']==phase for phase,bounds in enumerate([(0,30),(30,80),(80,130)]) for i in range(*bounds))
        # Verify actual saved sampler/counters, scheduler horizons and selected
        # teachers, rather than treating130 log lines as proof of training.
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        state=last['model']['_extra_state']
        weights=last['model']
        assert weights['network.wsi_adapter.head.proj.weight'].shape==(768,a.dimension)
        assert weights['network.cls_token'].shape==(1,1,768)
        assert weights['network.holder_path'].shape==(1,255,768)
        assert weights['network.holder_omic'].shape==(1,256,768)
        for i,width in enumerate(sizes):
            prefix=f'network.omics_adapter.sig_networks.{i}.'
            assert weights[prefix+'0.0.weight'].shape==(768,width)
            assert weights[prefix+'1.0.weight'].shape==(768,768)
        for i in range(2):
            assert weights[f'network.omics_adapter.transformer.layers.{i}.self_attn.in_proj_weight'].shape==(2304,768)
        assert not any(n.startswith('network.omics_adapter.transformer.layers.2.') for n in weights)
        assert len(backbone().encoder.layer)==12
        assert sum(p.numel() for p in backbone().parameters())==107719680
        assert state['stage_progress']==[30,50,50] and state['optimizer_steps']==expected_updates
        assert len(state['train_orders'])==130 and state['teacher_states']==[None,None]
        for item in state['train_orders']:
            ids=item['patients'];assert set(ids)==allowed[item['phase']] and len(ids)==len(set(ids))
        assert [s['last_epoch'] for s in state['schedulers']]==[30,50,50]
        assert [s['T_max'] for s in state['schedulers']]==[30,50,50]
        assert len(last['optimizers'])==3 and last['optimizers'][0]['state']==last['optimizers'][1]['state']=={}
        selected=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        assert selected['model']['_extra_state']['phase']==2 and 80<=selected['epoch']<130
        assert selected['model']['_extra_state']['bound_teachers']==protocol['bound_teachers']
        assert 'optimizers' not in selected and 'rng' not in selected
        selected_epoch=selected['epoch']
        del last,selected;gc.collect()
        metrics=evaluate(run,device=a.device);prediction_sha=digest_file(run/'test_predictions.csv')
        assert evaluate(run,device=a.device)==metrics and digest_file(run/'test_predictions.csv')==prediction_sha
        for name,module in list(sys.modules.items()):
            if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).is_relative_to(installed),name
        report.update(state='passed',completed_stage_epochs=[30,50,50],completed_epochs=130,
            native_language_parameters=107719680,native_hidden=768,genomic_pathways=331,
            pathway_sizes=sizes,actual_patches_per_observed_slide=300,
            optimizer_updates=expected_updates,total_optimizer_updates=sum(expected_updates),
            selected_epoch=selected_epoch,selected_checkpoint_sha256=digest_file(run/'best.pt'),
            resumable_checkpoint_sha256=digest_file(run/'last.pt'),run=str(run),test_metrics=metrics,
            prediction_sha256=prediction_sha,actual_selected_checkpoint_reloaded_twice=True,
            elapsed_seconds=time.monotonic()-started,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None)
        write_json(out/'validation.json',report)
        print(json.dumps({k:report[k] for k in ['state','feature_dimension','completed_epochs','optimizer_updates','elapsed_seconds']}),flush=True)
    except BaseException as e:
        report.update(state='failed',error=repr(e),elapsed_seconds=time.monotonic()-started)
        write_json(out/'validation.json',report);raise


if __name__=='__main__':main()
