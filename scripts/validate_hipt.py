"""Actual pretrained native HIPT, both complete twenty-epoch classification routes."""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from milplus.data import Bundle
from milplus.engine import evaluate, train
from milplus.hipt_demo import create_hipt_demo
from milplus.hipt_source import PROFILES, TENSOR_SHA, tensor_map_hash
from milplus.io import digest_file, read_json, write_json
from milplus.provenance import software_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--weights',default=os.environ.get('MILPLUS_HIPT_WEIGHTS',
        '/data/milplus_model_weights/hipt_vit4k_xs_teacher_backbone.pt'))
    args=parser.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    identity=software_identity()
    report={'state':'running','software':identity,'profiles':[],
        'scope':'Actual canonical public teacher and complete physical 16x16 grids; native 384/192 architecture and twenty actual epochs. Synthetic patch features, no actual WSI/DINO extraction or paper-score validation.'}
    if args.device=='cuda' and not torch.cuda.is_available():
        report.update(state='not_run',reason='CUDA hardware/runtime unavailable')
        write_json(output/'validation.json',report);return
    data=create_hipt_demo(output/'data',args.weights);bundle=Bundle(data)
    steps=20*len(bundle.records('train'))
    for profile,frozen in PROFILES.items():
        run=train(data,profile,output/profile,device=args.device,config={'threads':1})
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        extra=last['model']['_extra_state']
        assert extra['optimizer_steps']==steps and extra['completed_epochs']==20 and last['epoch']==19
        request=read_json(run/'request.json')
        assert request['config']['epochs']==20 and request['config']['selection_metric']=='hipt_val_loss'
        assert torch.load(run/'best.pt',map_location='cpu',weights_only=False)['epoch']==19
        state={k.removeprefix('network.local_vit.'):v for k,v in last['model'].items() if k.startswith('network.local_vit.')}
        assert len(state)==78 and state['pos_embed'].shape==(1,197,192)
        teacher_unchanged=tensor_map_hash(state)==TENSOR_SHA
        assert teacher_unchanged==frozen
        optimizer=last['optimizers'][0]
        assert optimizer['param_groups'][0]['betas']==(.9,.999)
        assert optimizer['param_groups'][0]['weight_decay']==1e-5
        assert optimizer['param_groups'][0]['lr']==2e-4
        assert all(int(s['step'])==steps for s in optimizer['state'].values())
        assert len(read_json(run/'history.json'))==20
        result=evaluate(run,device=args.device);prediction_sha=digest_file(run/'test_predictions.csv')
        assert result==evaluate(run,device=args.device) and prediction_sha==digest_file(run/'test_predictions.csv')
        instances=read_json(run/'test_instances/manifest.json')
        assert len(instances['records'])==len(bundle.records('test'))
        for entry in instances['records']:
            with h5py.File(entry['path']) as h:
                order=h['canonical_region_rows'][:];g=h['region_attention'][:];local=h['local_cls_attention_all_heads'][:].mean(1)
                assert order.shape==(2,256) and local.shape==(2,257)
                np.testing.assert_allclose(h['hierarchical_attention_product'][:][order],g[:,None]*local[:,1:],rtol=0,atol=0)
                np.testing.assert_allclose(g.sum(),1.,rtol=0,atol=1e-6)
                np.testing.assert_allclose(entry['patch_attention_mass']+entry['cls_self_attention_mass'],1.,rtol=0,atol=1e-6)
        protocol=read_json(run/'method_protocol.json')
        assert protocol['completed_epochs']==20 and protocol['optimizer_steps']==steps and protocol['freeze_local_vit']==frozen
        report['profiles'].append({'id':profile,'state':'passed','device':args.device,'epochs':20,
            'input_dimension':384,'local_dimension':192,'region_grid':[16,16],'optimizer_updates':steps,
            'initial_teacher_tensor_sha256':TENSOR_SHA,'pretrained_teacher_unchanged':teacher_unchanged,
            'selected_epoch':19,'metrics':result,'prediction_sha256':prediction_sha,
            'interpretation_slides':len(instances['records'])})
        assert software_identity()==identity
        write_json(output/'validation.json',report)
    report['state']='passed';write_json(output/'validation.json',report)
    print('Both actual pretrained native HIPT routes passed twenty epochs each,',2*steps,'Adam updates')


if __name__=='__main__':main()
