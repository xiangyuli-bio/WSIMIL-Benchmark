"""Complete native-width sixty-epoch H2 source/repaired routes, including actual reload."""
import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from milplus.data import Bundle
from milplus.engine import evaluate, seed_everything, train
from milplus.h2mil import H2MIL
from milplus.h2mil_demo import create_h2mil_demo
from milplus.io import digest_file, read_json, write_json
from milplus.provenance import software_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    identity=software_identity();torch.set_num_threads(1)
    report={'state':'running','software':identity,'profiles':[],
        'scope':'Actual full native1024/256/128 source architecture and sixty-epoch released recipe, both explicit head/loss routes. True three-level synthetic physical grids with independent Gaussian level features. No natural WSI, actual KimiaNet extraction or paper-score claim.'}
    if args.device=='cuda' and not torch.cuda.is_available():
        report.update(state='not_run',reason='CUDA hardware/runtime unavailable')
        write_json(root/'validation.json',report);return
    data=create_h2mil_demo(root/'data');bundle=Bundle(data)
    steps=60*math.ceil(len(bundle.records('train'))/32)
    for profile in ['h2mil_source_compat','h2mil_repaired']:
        seed_everything(2026);initial=H2MIL(bundle,profile=profile)
        unchanged={f'network.{name}':p.detach().clone() for name,p in initial.network.named_parameters() if name.startswith('pool_')}
        del initial
        run=train(data,profile,root/profile,device=args.device,config={'threads':1})
        assert not (run/'test_predictions.csv').exists()
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        cfg=read_json(run/'request.json')['config'];extra=last['model']['_extra_state']
        assert cfg['epochs']==60 and cfg['selection_metric']=='h2_val_auc' and cfg['grad_clip'] is None
        assert extra['optimizer_steps']==steps and extra['completed_epochs']==60 and last['epoch']==59
        assert last['model']['network.conv1.lin_l.weight'].shape==(256,1024)
        assert last['model']['network.lin1.weight'].shape==(128,256)
        assert last['model']['network.lin2.weight'].shape==((4 if profile=='h2mil_source_compat' else 2),128)
        for name,value in unchanged.items():torch.testing.assert_close(last['model'][name],value,rtol=0,atol=0)
        optimizer=last['optimizers'][0]
        assert optimizer['param_groups'][0]['betas']==(.9,.999)
        assert optimizer['param_groups'][0]['lr']==5e-5 and optimizer['param_groups'][0]['weight_decay']==5e-4
        assert all(int(s['step'])==steps for s in optimizer['state'].values())
        history=read_json(run/'history.json');assert len(history)==60
        eligible=[h for h in history if h['validation']['h2_training_auc']>.8 or h['epoch']>50]
        expected=max(h['epoch'] for h in eligible if h['validation']['h2_val_auc']==max(v['validation']['h2_val_auc'] for v in eligible))
        assert torch.load(run/'best.pt',map_location='cpu',weights_only=False)['epoch']==expected
        metrics=evaluate(run,device=args.device);sha=digest_file(run/'test_predictions.csv')
        assert metrics==evaluate(run,device=args.device) and sha==digest_file(run/'test_predictions.csv')
        frame=pd.read_csv(run/'test_predictions.csv')
        if profile=='h2mil_source_compat':
            original=frame[[f'h2_original_p_{i}' for i in range(4)]].to_numpy()
            np.testing.assert_allclose(original.sum(1),1.,rtol=0,atol=1e-6)
            np.testing.assert_array_equal(original[:,1],frame.ranking_score)
            np.testing.assert_array_equal(original.argmax(1),frame.h2_native_prediction)
            np.testing.assert_array_equal(original[:,1],frame.p_1)
        instances=read_json(run/'test_instances/manifest.json')
        assert len(instances['records'])==len(bundle.records('test'))
        for entry in instances['records']:
            with h5py.File(entry['path']) as h:
                assert h['coords'].shape==(16,2) and h['all_node_coords'].shape==(21,2)
                np.testing.assert_array_equal(h['all_nodes_final_cluster'][:],h['stage_1/membership'][:][h['stage_0/membership'][:]])
                np.testing.assert_array_equal(h['fine_final_cluster'][:],h['all_nodes_final_cluster'][:][h['original_node_type'][:]==2])
        protocol=read_json(run/'method_protocol.json')
        assert protocol['completed_epochs']==60 and protocol['optimizer_steps']==steps
        report['profiles'].append({'id':profile,'state':'passed','device':args.device,'epochs':60,
            'input_dimension':1024,'hidden_dimension':256,'head_dimension':128,'optimizer_updates':steps,
            'selected_epoch':expected,'unchanged_source_pool_tensors':len(unchanged),'metrics':metrics,
            'prediction_sha256':sha,'structural_slides':len(instances['records'])})
        assert software_identity()==identity;write_json(root/'validation.json',report)
    report['state']='passed';write_json(root/'validation.json',report)
    print('Both native H2 routes passed sixty epochs each,',2*steps,'actual Adam updates and selected-checkpoint reloading')


if __name__=='__main__':main()
