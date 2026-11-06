"""Full native-width, 39-epoch SI-MIL verification; suitable for extracted-wheel imports."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from milplus.data import Bundle
from milplus.engine import evaluate, train
from milplus.io import digest_file, read_json, write_json
from milplus.provenance import software_identity
from milplus.simil_demo import create_simil_demo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    identity=software_identity()
    report={'state':'running','software':identity,'profiles':[],
        'scope':'Native 384 deep / 246 PathExpert descriptors, K20/four PF-Mixers, complete 39 actual epochs. CPU synthetic only unless CUDA explicitly requested; no natural WSI or paper-score claim.'}
    if args.device=='cuda' and not torch.cuda.is_available():
        report.update(state='not_run',reason='CUDA hardware/runtime unavailable')
        write_json(output/'validation.json',report);return
    data=create_simil_demo(output/'data');bundle=Bundle(data)
    steps=39*len(bundle.records('train'))
    for profile in ['simil_source','simil_full_vjp']:
        run=train(data,profile,output/profile,device=args.device,config={'threads':1})
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        extra=last['model']['_extra_state']
        assert extra['prepared'] and extra['optimizer_steps']==steps and extra['completed_epochs']==39
        assert len(extra['normalization']['edges'])==246
        assert extra['scheduler']['last_epoch']==39 and extra['scheduler']['T_max']==40
        assert all(int(s['step'])==steps for s in last['optimizers'][0]['state'].values())
        assert last['model']['network.b_classifier.projection_head_deep.0.weight'].shape==(384,384)
        assert last['model']['network.b_classifier.classification_head.0.weight'].shape==(1,246)
        assert last['optimizers'][0]['param_groups'][0]['betas']==(.5,.9)
        assert last['optimizers'][0]['param_groups'][0]['weight_decay']==.005
        history=read_json(run/'history.json');scores=[e['validation']['simil_joint'] for e in history]
        expected=max(i for i,s in enumerate(scores) if s==max(scores))
        assert torch.load(run/'best.pt',map_location='cpu',weights_only=False)['epoch']==expected
        result=evaluate(run,device=args.device);prediction_sha=digest_file(run/'test_predictions.csv')
        assert result==evaluate(run,device=args.device)
        assert prediction_sha==digest_file(run/'test_predictions.csv')
        frame=pd.read_csv(run/'test_predictions.csv')
        assert np.isfinite(frame[['p_0','p_1','deep_probability']].to_numpy()).all()
        instances=read_json(run/'test_instances'/'manifest.json')
        assert len(instances['records'])==len(bundle.records('test'))
        report['profiles'].append({'id':profile,'state':'passed','device':args.device,
            'epochs':39,'input_dimension':384,'pathexpert_dimension':246,'top_k':20,'mixer_layers':4,
            'optimizer_updates':steps,'selected_epoch':expected,'metrics':result,
            'prediction_sha256':prediction_sha,'interpretation_slides':len(instances['records'])})
        assert software_identity()==identity
        write_json(output/'validation.json',report)
    report['state']='passed';write_json(output/'validation.json',report)
    print('Two native SI-MIL routes passed, 39 epochs each,',2*steps,'actual optimizer updates')


if __name__=='__main__':
    main()
