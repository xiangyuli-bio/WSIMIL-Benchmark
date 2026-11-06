"""Independent installed-wheel original SLIP10epoch budgets for both CLIP branches."""
import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    for key in ['wheel','output','source-sha','lock-sha']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    wheel=Path(args.wheel).resolve();wheel_sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
    installed=root/'wheel_import';installed.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        archive.extractall(installed)
    sys.path.insert(0,str(installed))
    import torch
    import milplus
    from milplus.data import Bundle
    from milplus.engine import evaluate,train
    from milplus.io import digest_file,read_json,write_json
    from milplus.provenance import software_identity
    from milplus.slip_demo import create_slip_demo
    from milplus.slip_source import ASSETS
    assert Path(milplus.__file__).is_relative_to(installed) and milplus.__version__=='0.66.0'
    identity=software_identity()
    assert identity['source_sha256']==args.source_sha and identity['sources_lock_sha256']==args.lock_sha
    if args.device=='cuda':assert torch.cuda.is_available()
    report={'state':'running','formal_admission':False,'synthetic':True,'software':identity,
            'wheel_sha256':wheel_sha,'device':args.device,'profiles':[]}
    write_json(root/'validation.json',report);started=time.monotonic()
    try:
        for architecture,classes,patients in [('clip_vit_b16',2,20),('clip_rn50',3,30)]:
            dimension=ASSETS[architecture][2]
            data=create_slip_demo(root/('inputs_'+architecture),patients=patients,classes=classes,architecture=architecture)
            bundle=Bundle(data);inventory=bundle.check(deep=True)
            method='slip_'+architecture+'_source_repaired'
            cfg={'threads':2,'epochs':10,'lr':.002,'weight_decay':0.,'grad_clip':None,
                 'max_instances':None,'selection_metric':'slip_final_epoch','model':{}}
            run=train(data,method,root/method,device=args.device,config=cfg,seed=2066)
            status=read_json(run/'status.json');protocol=read_json(run/'method_protocol.json')
            assert status['state']=='completed' and status['completed_epochs']==10
            assert protocol['completed_epochs']==10 and protocol['optimizer_updates']==10*len(bundle.records('train'))
            assert protocol['text_hidden']==512 and protocol['text_layers']==12 and protocol['text_heads']==8
            assert protocol['feature_dimension']==dimension and protocol['actual_asset_sha256']==ASSETS[architecture][1]
            last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
            assert last['epoch']==best['epoch']==9 and len(last['history'])==10
            assert len(last['optimizers'])==1 and last['optimizers'][0]['state']=={}
            groups=last['optimizers'][0]['param_groups'];assert len(groups)==1
            assert groups[0]['lr']==.002 and groups[0]['momentum']==groups[0]['weight_decay']==0
            state=last['model']['_extra_state'];allowed={r['slide_id'] for r in bundle.records('train')}
            assert len(state['train_orders'])==10
            assert all(len(o)==len(allowed) and set(o)==allowed for o in state['train_orders'])
            assert best['model']['core.prompt_learner.ctx'].shape==(classes,1,512)
            assert torch.equal(best['model']['core.prompt_learner.ctx'],last['model']['core.prompt_learner.ctx'])
            assert not any(k.startswith(('core.native.','core.text_encoder.')) for k in last['model'])
            metrics=evaluate(run,device=args.device);pred_sha=digest_file(run/'test_predictions.csv')
            assert evaluate(run,device=args.device)==metrics and digest_file(run/'test_predictions.csv')==pred_sha
            report['profiles'].append({'id':method,'state':'passed','classes':classes,'feature_dimension':dimension,
                'text_hidden':512,'text_layers':12,'text_heads':8,'actual_asset_sha256':ASSETS[architecture][1],
                'completed_epochs':10,'selected_epoch':9,'batch_size':1,'optimizer_updates':protocol['optimizer_updates'],
                'train_slides':len(allowed),'train_order_count':len(state['train_orders']),
                'actual_selected_checkpoint_reloaded_twice':True,'feature_inventory':inventory['feature_inventory_sha256'],
                'run':str(run),'best_sha256':digest_file(run/'best.pt'),'last_sha256':digest_file(run/'last.pt'),
                'test_prediction_sha256':pred_sha,'test_metrics':metrics})
            write_json(root/'validation.json',report)
        for name,module in list(sys.modules.items()):
            if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).is_relative_to(installed),name
        report.update(state='passed',elapsed_seconds=time.monotonic()-started,
            scope='Both actual original-size CLIP/text/context/SLIP-loss/source-SGD10epoch installed-wheel budgets; synthetic inputs only, no natural WSI or paper-score reproduction')
        write_json(root/'validation.json',report);print(json.dumps({'state':'passed','profiles':2,'completed_epochs_each':10}))
    except BaseException as error:
        report.update(state='failed',error=repr(error),elapsed_seconds=time.monotonic()-started)
        write_json(root/'validation.json',report);raise


if __name__=='__main__':main()
