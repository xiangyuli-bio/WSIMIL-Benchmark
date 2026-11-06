"""Both native1024/512 AGDR source recipes, full200-epoch budget and source early stop."""
import argparse
import ast
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn

from milplus.agdrmil import AGDRMIL, COMMIT, PRESETS
from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import evaluate, seed_everything, train
from milplus.io import digest_file, read_json, write_json
from milplus.provenance import software_identity
from milplus.upstream import pinned_text, verified_source


def source_selector():
    root,sha=verified_source('agdrmil');assert sha==COMMIT
    nodes=[n for n in ast.parse(pinned_text(root,sha,'utils/utils.py')).body if isinstance(n,ast.ClassDef) and n.name=='Earlystopping']
    assert len(nodes)==1;scope={'torch':torch}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'native_original_agdr_selector.py','exec'),scope)  # noqa: S102 -- inspected fixed original callback
    return scope['Earlystopping'](patience=20,stop_epoch=20,verbose=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    identity=software_identity();report={'state':'running','software':identity,'profiles':[],
        'scope':'Complete actual1024/512 source models, both declared YAML/TCGA source presets, full200-epoch maximum budget and original20patience/minimumepoch early-stop policy. Synthetic frozen features only; no natural WSI/GPU/paper-score claim on CPU.'}
    if args.device=='cuda' and not torch.cuda.is_available():
        report.update(state='not_run',reason='CUDA hardware/runtime unavailable');write_json(root/'validation.json',report);return
    data=create_demo(root/'data',patients=20,dimension=1024);bundle=Bundle(data)
    for profile in PRESETS:
        seed_everything(2026);initial=AGDRMIL(bundle,profile=profile)
        untouched={f'network.{n}':p.detach().clone() for n,p in initial.network.named_parameters() if n.startswith('att.')}
        del initial
        run=train(data,profile,root/profile,device=args.device,config={'threads':1})
        assert not (run/'test_predictions.csv').exists()
        history=read_json(run/'history.json');completed=len(history)
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=False);cfg=read_json(run/'request.json')['config']
        assert cfg['epochs']==200 and cfg['patience']==20 and cfg['lr']==2e-4 and cfg['weight_decay']==1e-5
        assert cfg['grad_clip'] is None and cfg['max_instances'] is None and cfg['selection_metric']=='agdr_val_loss'
        assert 22<=completed<=200 and state['epoch']==completed-1
        updates=completed*len(bundle.records('train'));extra=state['model']['_extra_state']
        assert extra['optimizer_steps']==updates and extra['completed_epochs']==completed
        assert state['model']['network.reduce_dim.0.weight'].shape==(512,1024)
        assert state['model']['network.classifier_1.0.weight'].shape==(2,512)
        for name,value in untouched.items():torch.testing.assert_close(state['model'][name],value,rtol=0,atol=0)
        optimizer=state['optimizers'][0]
        assert optimizer['param_groups'][0]['betas']==(.9,.999) and optimizer['param_groups'][0]['lr']==2e-4
        assert optimizer['param_groups'][0]['weight_decay']==1e-5
        assert all(int(v['step'])==updates for v in optimizer['state'].values())
        callback=source_selector();dummy=nn.Linear(1,1)
        for i,h in enumerate(history):
            callback(h['epoch'],h['validation']['agdr_val_loss'],dummy,ckpt_name=root/(profile+'_source_policy.pt'))
            assert not callback.early_stop or i==completed-1
        assert completed==200 or callback.early_stop
        losses=[h['validation']['agdr_val_loss'] for h in history]
        selected=max(i for i,v in enumerate(losses) if v==min(losses))
        assert torch.load(run/'best.pt',map_location='cpu',weights_only=False)['epoch']==selected
        assert callback.best_score==-min(losses)
        metrics=evaluate(run,device=args.device);sha=digest_file(run/'test_predictions.csv')
        assert metrics==evaluate(run,device=args.device) and sha==digest_file(run/'test_predictions.csv')
        frame=pd.read_csv(run/'test_predictions.csv')
        raw=frame[[f'agdr_full_sigmoid_{i}' for i in range(2)]].to_numpy(dtype=np.float32)
        expected=torch.from_numpy(raw).softmax(1).numpy()
        np.testing.assert_allclose(frame[['p_0','p_1']].to_numpy(),expected,rtol=0,atol=1e-15)
        assert (raw>=0).all() and (raw<=1).all()
        entries=read_json(run/'test_instances/manifest.json')['records']
        assert len(entries)==len(bundle.records('test'))
        for entry in entries:
            with h5py.File(entry['path']) as h:
                retained=h['retained_order'][:];attn=h['attention'][:];grades=h['grades'][:]
                assert h['coords'].shape==(entry['instances'],2) and len(retained)==len(np.unique(retained))
                np.testing.assert_array_equal(h['partition'][:],np.where(grades>=PRESETS[profile][0],2,np.where(grades>=PRESETS[profile][1],1,0)))
                assert (attn[np.setdiff1d(np.arange(len(attn)),retained)]==0).all()
                np.testing.assert_allclose(attn.sum(),1.,atol=1e-6,rtol=0)
        protocol=read_json(run/'method_protocol.json')
        assert protocol['completed_epochs']==completed and protocol['optimizer_steps']==updates
        report['profiles'].append({'id':profile,'state':'passed','device':args.device,'max_epochs':200,
            'completed_epochs':completed,'early_stopped':completed<200,'optimizer_updates':updates,
            'selected_epoch':selected,'native_input_dimension':1024,'native_hidden_dimension':512,
            'unchanged_hard_grader_parameters':len(untouched),'source_selector':'actual callback exact stop/last-tie on every epoch',
            'test_prediction_sha256':sha,'structural_slides':len(entries),'metrics':metrics})
        assert software_identity()==identity;write_json(root/'validation.json',report)
    report['state']='passed';write_json(root/'validation.json',report)
    print('Both full-budget native AGDR source recipes passed:',sum(r['optimizer_updates'] for r in report['profiles']),'actual Adam updates, selected-checkpoint reloading and instance export')


if __name__=='__main__':main()
