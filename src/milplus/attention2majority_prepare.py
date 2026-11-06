"""Build a traceable Appendix-I TRAIN pool from actual original ImageNet features."""
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from .attention2majority_data import AttentionInputs
from .attention2majority_driver import environment
from .attention2majority_ica import select_large_tumor_train_pool
from .io import digest_file,read_json,write_json
from .provenance import software_identity


def prepare(data,output,template_slide,normal_reference,component_space,*,fm=None,split=None,
            seed=2026,threads=2,device='cpu',tensorflow_python=None):
    if type(threads) is not int or threads<1 or type(seed) is not int:
        raise ValueError('Declare a positive native thread count and integer seed')
    inputs=AttentionInputs(data,fm,split,require_pool=False)
    if inputs.mode!='camelyon_binary':raise ValueError('Appendix-I ICA belongs to the binary Camelyon variant')
    rows=inputs.rows;train={r['slide_id']:r for r in rows if r['subset']=='train'}
    if (template_slide not in train or normal_reference not in train
            or train[template_slide]['target']!=1 or train[normal_reference]['target']!=0
            or component_space not in {'unmixing_rows','mixing_columns'}):
        raise ValueError('Select an explicit TRAIN-positive template, TRAIN-normal reference and disclosed ICA convention')
    cfg={'threads':threads,'tensorflow_python':tensorflow_python};executable,env=environment(cfg,device)
    probe=subprocess.run([executable,'-c',
        'import json;from milplus.attention2majority_tf import runtime_identity;print(json.dumps(runtime_identity()))'],
        env=env,capture_output=True,text=True,check=True)
    runtime=json.loads(probe.stdout);software=software_identity()
    if runtime['runtime_source_sha256']!=software['source_sha256']:
        raise ValueError('Initial-feature adapter and common data protocol source differ')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    request={'software':software,'tensorflow_runtime':runtime,'device':device,'seed':seed,'config':cfg,
        'template_slide':template_slide,'normal_reference':normal_reference,'component_space':component_space,
        'input_contract':inputs.contract,'formal_admission':False}
    write_json(out/'request.json',request);write_json(out/'input_binding.json',inputs.export())
    cmd=[executable,'-m','milplus.attention2majority_initial_tf','--binding',str(out/'input_binding.json'),
        '--request',str(out/'request.json'),'--output',str(out/'initial_TRAIN_features')]
    with (out/'native.log').open('x') as log:
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        write_json(out/'native_process.json',{'pid':child.pid,'command':cmd})
        code=child.wait()
    if code:raise RuntimeError('Actual initial Keras feature extraction failed; inspect native.log')
    initial=read_json(out/'initial_TRAIN_features/manifest.json')
    if initial['state']!='passed' or set(initial['records'])!=set(train):
        raise ValueError('Actual Appendix-I extraction omitted/added TRAIN slides')
    features={}
    for slide,meta in initial['records'].items():
        if digest_file(meta['path'])!=meta['sha256']:raise ValueError('Initial ImageNet feature bytes changed')
        features[slide]=np.load(meta['path'],allow_pickle=False,mmap_mode='r')
    proof=select_large_tumor_train_pool(features,rows,template_slide,normal_reference,
        seed=seed,component_space=component_space,threshold=.1,positive_limit=12)
    pool=[s for s,r in train.items() if r['target']==0]+proof['selected_positive_slide_ids']
    proof.update(selected_train_slide_ids=pool,review_provenance=
        'Explicit Appendix-I reconstruction from every TRAIN patch using original ImageNet conv4/GAP1024; '
        'normal slide, whitening/component convention and positive cap named, absent author ICA code not historically reproduced',
        initial_feature_manifest=str(out/'initial_TRAIN_features/manifest.json'),
        initial_feature_manifest_sha256=digest_file(out/'initial_TRAIN_features/manifest.json'),
        synthetic=bool(inputs.bundle.config.get('synthetic',False)))
    dataset=out/'dataset';dataset.mkdir();write_json(dataset/'pool_review.json',proof)
    table=pd.DataFrame([{k:r[k] for k in ['slide_id','patient_id','global_patient_id','label']} for r in rows])
    table.to_csv(dataset/'slides.csv',index=False)
    inputs.bundle.rows[['patient_id','subset']].drop_duplicates().to_csv(dataset/'split.csv',index=False)
    pd.DataFrame([{'slide_id':s,'path':r['path']} for s,r in inputs.records.items()]).to_csv(dataset/'images.csv',index=False)
    config=copy.deepcopy(inputs.bundle.config);schema=copy.deepcopy(inputs.schema)
    schema.update(weights=str(inputs.weights),discriminator_train_pool=pool,discriminator_pool_evidence='pool_review.json')
    config.update(slides='slides.csv',defaults={'fm':inputs.bundle.fm,'split':inputs.bundle.split},
        splits={inputs.bundle.split:'split.csv'},encoders={inputs.bundle.fm:{'dimension':3,'index':'images.csv','attention2majority_raw':schema}})
    (dataset/'task.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    inputs.check(deep=True);checked=AttentionInputs(dataset)
    write_json(out/'validation.json',{'state':'passed','dataset':str(dataset),'input_contract':checked.contract,
        'initial_TRAIN_features':len(features),'selected_TRAIN_slides':pool,'heldout_labels_used':False,
        'all_original_raw_paths_retained':True,'formal_admission':False,'real_wsi_validation':'not_run','paper_reproduction':'not_run'})
    return dataset
