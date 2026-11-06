"""Independent installed-wheel VLSA native32patient/10epoch training budgets.

The published cached-notebook agreement is a separate required fidelity gate.
This program never converts a failed author-example check into pass evidence.
"""
import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    for key in ['wheel','output','source-sha','lock-sha','weights']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    wheel=Path(args.wheel).resolve();wheel_sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
    installed=root/'wheel_import';installed.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Invalid installed wheel member')
        archive.extractall(installed)
    sys.path.insert(0,str(installed))
    import torch
    import milplus
    from milplus.data import Bundle
    from milplus.engine import evaluate,train
    from milplus.io import digest_file,read_json,write_json
    from milplus.provenance import software_identity
    from milplus.vlsa_demo import create_vlsa_demo
    from milplus.vlsa_source import WEIGHTS_SHA
    assert Path(milplus.__file__).is_relative_to(installed) and milplus.__version__=='0.67.0'
    identity=software_identity()
    assert identity['source_sha256']==args.source_sha and identity['sources_lock_sha256']==args.lock_sha
    if args.device=='cuda':assert torch.cuda.is_available()
    report={'state':'running','synthetic':True,'formal_admission':False,'software':identity,
        'device':args.device,'wheel_sha256':wheel_sha,'native_budgets':[],
        'author_notebook_reference':'Separate required gate; CPU discrepancy unresolved, no reference pass claim'}
    write_json(root/'validation.json',report);started=time.monotonic()
    try:
        for key,patients,queries in [('tcga_brca_0',60,10),('tcga_blca_0',80,12)]:
            data=create_vlsa_demo(root/('inputs_'+key),args.weights,patients=patients,query_key=key)
            bundle=Bundle(data);inventory=bundle.check(deep=True)
            method='vlsa_conch_source_repaired'
            cfg={'threads':2,'epochs':10,'lr':2e-4,'weight_decay':1e-5,'grad_clip':None,
                'max_instances':None,'selection_metric':'vlsa_final_epoch','model':{'update_patients':32}}
            run=train(data,method,root/('run_'+key),device=args.device,config=cfg,seed=2067)
            status=read_json(run/'status.json');protocol=read_json(run/'survival_protocol.json')
            assert status['state']=='completed' and status['completed_epochs']==10
            assert protocol['native_window'] is True and protocol['update_patients']==32
            assert protocol['completed_epochs']==10 and protocol['num_queries']==queries
            train_ids=bundle.vlsa.train_patients
            windows=[32]*(len(train_ids)//32)
            if len(train_ids)%32:windows.append(len(train_ids)%32)
            assert len(train_ids)>32 and windows[0]==32 and len(windows)>1
            assert protocol['optimizer_updates']==10*len(windows)
            last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
            assert last['epoch']==best['epoch']==9 and len(last['history'])==10
            state=last['model']['_extra_state']
            assert state['window_sizes']==[windows]*10 and len(state['train_orders'])==10
            assert all(len(order)==len(train_ids) and set(order)==set(train_ids) for order in state['train_orders'])
            groups=last['optimizers'][0]['param_groups']
            assert len(groups)==2 and [g['weight_decay'] for g in groups]==[0.,1e-5]
            assert all(g['lr']==2e-4 and tuple(g['betas'])==(.9,.999) for g in groups)
            assert last['model']['core.prompt_learner.context_embeds'].shape==(4,768)
            assert last['model']['core.prompt_learner.rank_embeds'].shape==(4,4,768)
            assert last['model']['core.mil_encoder.Q.residual_features'].shape==(queries,512)
            assert last['model']['core.mil_encoder.visual_adapter.weight'].shape==(512,512)
            assert len(last['model']['time_coordinates'])==bundle.vlsa.num_ranks
            for key_name,value in last['model'].items():
                if torch.is_tensor(value):assert torch.equal(value,best['model'][key_name]),key_name
            metrics=evaluate(run,device=args.device);pred_sha=digest_file(run/'test_predictions.csv')
            assert evaluate(run,device=args.device)==metrics and digest_file(run/'test_predictions.csv')==pred_sha
            report['native_budgets'].append({'state':'passed','query_key':key,'queries':queries,
                'patients':patients,'train_patients':len(train_ids),'epochs':10,'selected_epoch':9,
                'window_sizes':windows,'optimizer_updates':protocol['optimizer_updates'],
                'time_ranks':bundle.vlsa.num_ranks,'full_native_text':'768hidden/12layers/12heads/128context',
                'native_visual':512,'weights_sha256':WEIGHTS_SHA,'selected_checkpoint_reloaded_twice':True,
                'feature_inventory':inventory['feature_inventory_sha256'],'run':str(run),
                'best_sha256':digest_file(run/'best.pt'),'last_sha256':digest_file(run/'last.pt'),
                'test_prediction_sha256':pred_sha,'metrics':metrics})
            write_json(root/'validation.json',report)
        for name,module in list(sys.modules.items()):
            if name.startswith('milplus') and getattr(module,'__file__',None):
                assert Path(module.__file__).is_relative_to(installed),name
        report.update(state='passed',elapsed_seconds=time.monotonic()-started,
            scope='Two complete native32-patient+tail/10epoch actual-CONCH-text installed-wheel software budgets only; synthetic bags/outcomes, no clinical or paper-score result')
        write_json(root/'validation.json',report)
        print(json.dumps({'state':'passed','native_budgets':2,'epochs_each':10,'update_patients':32}))
    except BaseException as error:
        report.update(state='failed',error=repr(error),elapsed_seconds=time.monotonic()-started)
        write_json(root/'validation.json',report);raise


if __name__=='__main__':main()
