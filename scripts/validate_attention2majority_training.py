"""Actual full native Keras two-stage pipeline and interruption/reload comparison."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import yaml

from milplus.engine import train,evaluate
from milplus.io import digest_file,read_json,write_json


def fixture(root,weights,classes=2,patches=17):
    root.mkdir();rng=np.random.default_rng(9011+classes);rows=[];splits=[];indices=[];pool=[]
    pixels=224 if classes==2 else 128
    for subset,copies in [('train',2),('val',1),('test',1)]:
        for target in range(classes):
            for index in range(copies):
                slide=f'{subset}_class{target}_patient{index}';p=root/(slide+'.h5')
                with h5py.File(p,'w') as f:
                    f.create_dataset('pixels',data=rng.integers(0,256,(patches,pixels,pixels,3),dtype=np.uint8))
                    f.create_dataset('patch_ids',data=np.asarray([slide+f'_patch{i}' for i in range(patches)],dtype=h5py.string_dtype()))
                rows.append({'slide_id':slide,'patient_id':slide,'global_patient_id':'synthetic:'+slide,'label':f'class{target}'})
                splits.append({'patient_id':slide,'subset':subset});indices.append({'slide_id':slide,'path':p.name})
                if subset=='train':pool.append(slide)
    pd.DataFrame(rows).to_csv(root/'slides.csv',index=False);pd.DataFrame(splits).to_csv(root/'split.csv',index=False)
    pd.DataFrame(indices).to_csv(root/'images.csv',index=False)
    schema={'schema_version':1,'kind':'attention2majority_uint8_rgb_hdf5','mode':'camelyon_binary' if classes==2 else 'kidney_three_grade',
        'image_provenance':'synthetic complete native RGB224/128 software validation','patch_provenance':'all original randomly generated pixels and unique patch IDs',
        'weights':str(Path(weights).resolve())}
    if classes==2:
        schema.update(discriminator_train_pool=pool,discriminator_pool_evidence='pool_review.json')
        write_json(root/'pool_review.json',{'kind':'reviewed_large_tumor_TRAIN_pool','selected_train_slide_ids':pool,
            'heldout_labels_used':False,'review_provenance':'Synthetic TRAIN-only pool for software validation, no clinical tumor annotation claim','synthetic':True})
    config={'schema_version':1,'task_id':f'attention2majority_synthetic_{classes}class','task_type':'classification','synthetic':True,
        'classes':[f'class{i}' for i in range(classes)],'defaults':{'fm':'rgb','split':'fold0'},'splits':{'fold0':'split.csv'},
        'encoders':{'rgb':{'dimension':3,'index':'images.csv','attention2majority_raw':schema}}}
    (root/'task.yaml').write_text(yaml.safe_dump(config));return root


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    results=[]
    for profile,classes in [('attention2majority_camelyon_source_repaired',2),
            ('attention2majority_camelyon_paper_reference',2),('attention2majority_kidney_paper_reference',3)]:
        case=out/profile;case.mkdir();data=fixture(case/'data',args.weights,classes)
        cfg={'epochs':2,'threads':2,'model':{'discriminator_epochs':2}}
        full=train(data,profile,case/'continuous',seed=9012,config=cfg)
        interrupted=train(data,profile,case/'resumed',seed=9012,config={**cfg,'model':{**cfg['model'],'stop_discriminator_after':1}})
        assert read_json(interrupted/'status.json')['state']=='interrupted'
        train(data,profile,interrupted,seed=9012,config=cfg,resume=True,stop_after=1)
        assert read_json(interrupted/'status.json')['state']=='interrupted'
        train(data,profile,interrupted,seed=9012,config=cfg,resume=True)
        assert read_json(interrupted/'status.json')['state']=='completed'
        # Independent native processes export every actual TF model/Adam value
        # from the selected checkpoint; no validation/test weight selection.
        for run in [full,interrupted]:
            first=evaluate(run);prediction=digest_file(run/'test_predictions.csv')
            assert evaluate(run)==first and digest_file(run/'test_predictions.csv')==prediction
        a=read_json(full/'aggregation/state.json');b=read_json(interrupted/'aggregation/state.json')
        assert a['epochs']==b['epochs']==2 and a['updates']==b['updates']==4*classes
        assert a['history']==b['history']
        # Exact native checkpoint arrays are compared by the TF-only exporter.
        import subprocess,os
        python='/root/MIL/validation/milplus_validation_envs/camil_tf/bin/python'
        code='''import sys,tensorflow as tf,numpy as np,json
pa,pb=sys.argv[1:3];a=tf.train.load_checkpoint(json.load(open(pa))['checkpoint_prefix']);b=tf.train.load_checkpoint(json.load(open(pb))['checkpoint_prefix']);ka=a.get_variable_to_shape_map();kb=b.get_variable_to_shape_map();assert ka==kb
for k in ka:
 if k=='_CHECKPOINTABLE_OBJECT_GRAPH':continue
 np.testing.assert_array_equal(a.get_tensor(k),b.get_tensor(k),err_msg=k)
print('every native model/optimizer tensor bit-identical')'''
        subprocess.run([python,'-c',code,str(full/'aggregation/state.json'),str(interrupted/'aggregation/state.json')],check=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','TF_ENABLE_ONEDNN_OPTS':'0'})
        # The TRAIN-adapted feature arrays must also be exactly reproducible.
        x=read_json(full/'features/manifest.json');y=read_json(interrupted/'features/manifest.json')
        for slide in x['records']:
            with np.load(x['records'][slide]['path'],allow_pickle=False) as xa,np.load(y['records'][slide]['path'],allow_pickle=False) as ya:
                assert xa.files==ya.files
                for key in xa.files:np.testing.assert_array_equal(xa[key],ya[key])
        results.append({'profile':profile,'state':'passed','classes':classes,'full_native_1024features':True,
            'all_TF_weights_Adam_and_selected_raw_features_recovered_bit_exact':True,'two_independent_evaluation_reloads':True})
        write_json(out/'validation.json',{'state':'running','conditions':results,'scope':'two-epoch software lifecycle validation; original150epoch native-budget gates remain separate'})
    write_json(out/'validation.json',{'state':'passed','conditions':results,'formal_admission':False,
        'real_wsi_validation':'not_run','paper_reproduction':'not_run','scope':'Complete original native-width Keras CNN/filter/MIL, both stage interruption and exact recovery; software budget2epochs, not paper reproduction'})


if __name__=='__main__':main()
