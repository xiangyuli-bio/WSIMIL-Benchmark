"""Synthetic native-width CLIP bags; never represented as extracted WSI features."""
import yaml

from .demo import create_demo
from .io import write_json
from .slip_source import ASSETS


def create_slip_demo(output,patients=20,classes=2,architecture='clip_vit_b16'):
    if architecture not in ASSETS:
        raise ValueError('SLIP demo requires a supported original CLIP architecture')
    _,sha,dimension=ASSETS[architecture]
    root=create_demo(output,patients=patients,classes=classes,dimension=dimension)
    task=yaml.safe_load((root/'task.yaml').read_text())
    preprocessing='Synthetic independent Gaussian vectors at the native CLIP output width; no actual image extraction'
    task['encoders']['toy'].update(model=architecture,weights_sha256=sha,feature_preprocessing=preprocessing,
                                   slip={'manifest':'slip_inputs.json'})
    write_json(root/'slip_inputs.json',{'schema_version':1,'kind':'slip_native_clip_prompts',
        'synthetic':True,'architecture':architecture,'weights_sha256':sha,
        'feature_preprocessing':preprocessing,'classes':task['classes'],
        'templates':['{}'],'slide_classnames':[[f'synthetic class {i}'] for i in range(classes)],
        'tissue_classnames':[['synthetic tissue A'],['synthetic tissue B','synthetic tissue B description']],
        'n_shots':'all','few_shot_provenance':'Synthetic fixed TRAIN subset, all provided TRAIN bags; no TEST selection',
        'prompt_provenance':'Explicit synthetic strings for software checks; no clinical knowledge claims'})
    (root/'task.yaml').write_text(yaml.safe_dump(task,sort_keys=False))
    return root
