"""Full original native input counts, stage budgets, recovery and actual CUDA gate."""
import argparse
import math
from pathlib import Path

import numpy as np

from milplus.attention2majority_demo import create_attention2majority_demo
from milplus.engine import train,evaluate
from milplus.io import digest_file,read_json,write_json
from milplus.provenance import software_identity


def stopping(state,maximum,early):
    best=-math.inf;bad=0
    assert state['epochs']==len(state['history']) and 1<=state['epochs']<=maximum
    for epoch,row in enumerate(state['history']):
        assert row['epoch']==epoch and math.isfinite(row['loss'])
        if row['accuracy']>best:best=row['accuracy'];bad=0
        else:bad+=1
    assert state['best_train_accuracy']==best and state['bad']==bad
    assert state['epochs']==maximum or early and bad>=10


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tensorflow-python',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if not a.device.startswith('cuda:'):raise ValueError('This full native validator requires actual TensorFlow CUDA, not a CPU fallback')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);results=[]
    write_json(out/'validation.json',{'state':'running','conditions':results,'software':software_identity(),'formal_admission':False})
    for profile,classes in [('attention2majority_camelyon_source_repaired',2),
            ('attention2majority_camelyon_paper_reference',2),('attention2majority_kidney_paper_reference',3)]:
        data=out/f'raw_{classes}class'
        if not data.exists():create_attention2majority_demo(data,a.weights,classes,patches=2001)
        cfg={'threads':2,'model':{'tensorflow_python':a.tensorflow_python}}
        run=train(data,profile,out/profile,seed=9012,device=a.device,
            config={**cfg,'model':{**cfg['model'],'stop_discriminator_after':3}})
        assert read_json(run/'status.json')['state']=='interrupted'
        partial=read_json(run/'discriminator/state.json');assert partial['epochs']==3
        partial_disc_hashes=partial['checkpoint_files']
        train(data,profile,run,seed=9012,device=a.device,config=cfg,resume=True,stop_after=3)
        assert read_json(run/'status.json')['state']=='interrupted'
        partial=read_json(run/'aggregation/state.json');assert partial['epochs']==3
        partial_MIL_hashes=partial['checkpoint_files']
        train(data,profile,run,seed=9012,device=a.device,config=cfg,resume=True)
        request=read_json(run/'request.json');disc=read_json(run/'discriminator/state.json');mil=read_json(run/'aggregation/state.json')
        assert request['config']['epochs']==request['config']['discriminator_epochs']==150
        assert request['tensorflow_runtime']['tensorflow_build']['is_cuda_build'] is True
        assert read_json(run/'status.json')['state']=='completed' and not (run/'test_predictions.csv').exists()
        source=profile=='attention2majority_camelyon_source_repaired'
        stopping(disc,150,not source);stopping(mil,150,True)
        assert disc['native_patch_batch']==104 and disc['model_parameters']>8_500_000 and disc['trainable_tensors']==174
        assert all('GPU:0' in d for d in disc['model_variable_devices'])
        assert all('GPU:0' in d for d in mil['model_variable_devices'])
        count=4 if classes==2 else 6;patches=count*(2000 if classes==2 else 200)
        steps=math.ceil(patches/104)
        assert disc['updates']==disc['epochs']*steps and mil['updates']==mil['epochs']*count
        assert all(h['patches']==patches for h in disc['history'])
        ledger=read_json(run/'discriminator/selected_TRAIN_patches.json');assert len(ledger)==patches
        assert all(r['slide_id'].startswith('train_') and 0<=r['patch']<2001 for r in ledger)
        assert len({(r['slide_id'],r['patch']) for r in ledger})==patches
        for row in disc['history']:
            assert digest_file(row['order_path'])==row['order_sha256']
            order=np.load(row['order_path'],allow_pickle=False)
            np.testing.assert_array_equal(np.sort(order),np.arange(patches))
        for row in mil['history']:
            assert len(row['TRAIN_order'])==len(set(row['TRAIN_order']))==count
            assert all(s.startswith('train_') for s in row['TRAIN_order'])
        features=read_json(run/'features/manifest.json')
        assert len(features['records'])==4*classes and features['all_original_patches_scored'] is True
        for slide,record in features['records'].items():
            assert record['original_count']==2001
            with np.load(record['path'],allow_pickle=False) as v:
                assert v['all_features'].shape==(2001,1024) and v['selected_features'].shape[1]==1024
                np.testing.assert_array_equal(v['selected_features'],v['all_features'][v['selected_indices']])
                assert len(v['selected_indices'])==len(set(v['selected_indices'].tolist()))
                if classes==3:assert len(v['selected_indices'])==2000
                else:
                    positive=int((v['all_probabilities']>=.5).sum());negative=2001-positive
                    tail=round(.3*negative)
                    expected=round(.3*positive)+(negative if source and tail==0 else tail)
                    assert len(v['selected_indices'])==expected
        first=evaluate(run,device=a.device);prediction=digest_file(run/'test_predictions.csv')
        assert evaluate(run,device=a.device)==first and digest_file(run/'test_predictions.csv')==prediction
        result={'profile':profile,'state':'passed','device':a.device,'actual_GPU_variables_verified':True,
            'native_RGB_pixels':224 if classes==2 else 128,'original_patches_every_slide':2001,
            'TRAIN_discriminator_patch_count':patches,'actual_native_batch104_steps_each_epoch':steps,
            'discriminator_epochs':disc['epochs'],'discriminator_updates':disc['updates'],
            'MIL_epochs':mil['epochs'],'MIL_updates':mil['updates'],'maximum_epochs_both_stages':150,
            'both_actual_interruption_checkpoint_hashes':[partial_disc_hashes,partial_MIL_hashes],
            'actual_full1024D_task_trained_features':True,'selected_patch_identity_and_queue_verified':True,
            'selected_last_checkpoint_reloaded_twice':True,'prediction_sha256':prediction,'metrics':first,
            'synthetic':True,'real_wsi_validation':'not_run','paper_reproduction':'not_run'}
        results.append(result);write_json(out/'validation.json',{'state':'running','conditions':results,
            'software':software_identity(),'formal_admission':False})
    write_json(out/'validation.json',{'state':'passed','conditions':results,'software':software_identity(),
        'formal_admission':False,'scope':'complete original150max/native stopping, full RGB2001patch/native widths on actual CUDA; synthetic inputs, not paper reproduction'})


if __name__=='__main__':main()
