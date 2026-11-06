"""Native Keras ImageNet feature extraction for the Appendix-I TRAIN selector."""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from .attention2majority_source import build_discriminator,preprocess
from .attention2majority_tf import check_inputs,pixels,runtime_identity
from .io import digest_file,digest_object,read_json,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--request',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    binding=read_json(a.binding);request=read_json(a.request);out=Path(a.output)
    if binding['mode']!='camelyon_binary' or binding['pool']:
        raise ValueError('Appendix-I initial extraction requires explicitly unselected binary TRAIN bags')
    if tf.__version__!='2.15.1' or runtime_identity()!=request['tensorflow_runtime']:
        raise ValueError('Initial ImageNet feature runtime differs from its locked source/environment')
    device=request['device']
    if device.startswith('cuda:'):
        devices=tf.config.list_physical_devices('GPU');index=int(device[5:])
        if len(devices)<=index:raise ValueError('Requested initial extraction CUDA device unavailable; no CPU fallback')
        tf.config.set_visible_devices(devices[index],'GPU');tf.config.experimental.set_memory_growth(devices[index],True)
    elif device!='cpu':raise ValueError('Explicit cpu or cuda:N is required')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(request['config']['threads'])
    tf.keras.utils.set_random_seed(request['seed']);check_inputs(binding)
    _,encoder=build_discriminator(binding['weights'],224,2,'paper_uint8_caffe')
    out.mkdir(exist_ok=False);records={}
    for row in binding['rows']:
        if row['subset']!='train':continue
        slide=row['slide_id'];count=binding['records'][slide]['count'];parts=[]
        for offset in range(0,count,50):
            image=pixels(binding,[(slide,i) for i in range(offset,min(count,offset+50))])
            parts.append(encoder(preprocess(image,'paper_uint8_caffe'),training=False).numpy())
        features=np.concatenate(parts)
        if features.shape!=(count,1024) or not np.isfinite(features).all():
            raise ValueError('Complete native1024D initial ImageNet features are missing')
        path=out/(digest_object(slide)+'.npy');np.save(path,features,allow_pickle=False)
        records[slide]={'path':str(path),'sha256':digest_file(path),'count':count,
            'patch_ids_sha256':binding['records'][slide]['patch_ids_sha256']}
    manifest={'state':'passed','kind':'initial_ImageNet_not_task_adapted_TRAIN_only',
        'records':records,'optimizer_updates':0,'input_contract':binding['input_contract'],
        'request_sha256':digest_object(request),'heldout_feature_extraction':False,
        'native_backbone':'original Keras ResNet50 conv4_block6_out/GAP1024',
        'synthetic':binding['synthetic'],'paper_reproduction':'not_run'}
    check_inputs(binding);write_json(out/'manifest.json',manifest)


if __name__=='__main__':main()
