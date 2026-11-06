"""Marked synthetic native512 CONCH survival inputs; real language weights."""
from pathlib import Path

import yaml

from .demo import create_demo
from .io import write_json
from .vlsa_source import WEIGHTS_SHA,fixed,weights_path


def create_vlsa_demo(output,weights=None,patients=20,query_key='tcga_brca_0'):
    weights=weights_path(weights)
    if not weights.is_file():raise FileNotFoundError('VLSA demo needs the actual native CONCH checkpoint path')
    source,_=fixed()
    root=create_demo(output,patients=patients,dimension=512,survival=True)
    preprocessing='Synthetic native512 Gaussian patient bags; not extracted clinical CONCH features'
    task=yaml.safe_load((root/'task.yaml').read_text())
    task['encoders']['toy'].update(model='conch_ViT-B-16',weights_sha256=WEIGHTS_SHA,
        feature_preprocessing=preprocessing,vlsa={'manifest':'vlsa_inputs.json'})
    write_json(root/'vlsa_inputs.json',{'schema_version':1,'kind':'vlsa_native_conch_survival',
        'synthetic':True,'checkpoint':str(weights),'rank_prompts':str(source/'tools/survival_prompts.json'),
        'query_prompts':str(source/'tools/survival_text_prototypes.json'),'query_key':query_key,
        'num_time_bins':None,'feature_order_provenance':'Synthetic fixed slide/row identities, true patient concatenation',
        'feature_preprocessing':preprocessing,'prompt_provenance':'Exact ordered author prompt JSON from fixed VLSA commit',
        'weights_provenance':'Actual original CONCH checkpoint, exact SHA256 checked; no invented pretrained tensors'})
    (root/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False))
    return root
