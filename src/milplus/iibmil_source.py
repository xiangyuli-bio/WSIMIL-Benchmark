"""Complete released IIB-MIL components; not a runnable method admission.

The pinned public tree provides a full network, confidence/prototype modules
and inference entry, but no training epoch. These verified components must not
be presented as a complete faithful training recipe.
"""
import copy
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import yaml

from .io import digest_file
from .upstream import load_definitions, pinned_text, verified_source


COMMIT = 'a354ec6ea2b9ea2159e6784d54a6afb355c04b50'
CONDITIONS = {'luad_gm':'LUAD-GM','tcga_nsclc':'TCGA-NSCLC','tcga_rcc':'TCGA-RCC'}
MODEL_NAMES = ['_get_clones','_get_activation_fn','TransformerDecoderLayer',
               'TransformerDecoder','IIBMIL_Encoder','IIBMIL_Decoder','IIBMIL']


def checked_source():
    root, sha = verified_source('iibmil')
    if sha != COMMIT:
        raise ValueError('IIB-MIL requires its reviewed original source revision')
    return root,sha


def author_components():
    checked_source()
    nested = load_definitions('iibmil','main/util/misc.py',['NestedTensor'],
                              {'torch':torch,'Optional':Optional,'Tensor':Tensor}).NestedTensor
    network = load_definitions('iibmil','main/models/iibmil.py',MODEL_NAMES,
                               {'copy':copy,'torch':torch,'nn':nn,'F':F,'NestedTensor':nested})
    confidence = load_definitions('iibmil','main/util/loss.py',['PartialLoss'],
                                  {'torch':torch,'nn':nn,'F':F}).PartialLoss
    return network,nested,confidence


def source_config(condition):
    root,sha = checked_source()
    if condition not in CONDITIONS:
        raise ValueError('IIB-MIL needs an explicit released LUAD-GM, NSCLC or RCC condition')
    name='configs/'+CONDITIONS[condition]+'.yaml'
    path=root/name
    if path.read_text()!=pinned_text(root,sha,name):
        raise ValueError('Original IIB-MIL configuration bytes changed')
    config=yaml.safe_load(path.read_text())
    # The original entry merges these values through YACS, which evaluates
    # scientific-notation strings. PyYAML alone leaves e.g. `1e-5` as str;
    # normalize only the declared numeric training fields to the same values.
    for key in ['LR','LR_BACKBONE','LR_LINEAR_PROJ_MULT','WEIGHT_DECAY','CLIP_MAX_NORM','EMA']:
        config['TRAIN'][key]=float(config['TRAIN'][key])
    expected={'luad_gm':(2,32,8,16,2e-4),
              'tcga_nsclc':(2,256,3,16,1e-5),
              'tcga_rcc':(3,256,4,4,1e-5)}[condition]
    model,train=config['MODEL_IIBMIL'],config['TRAIN']
    assert (model['NUM_CLASSES'],model['HIDDEN_DIM'],model['DEPTH'],train['BATCH_SIZE'],train['LR'])==expected
    assert model['NUM_INPUT_CHANNELS']==1280 and config['DATASET']['FEATURE_LEN']==500
    assert train['EPOCHS']==300 and train['WARMUP']==10 and train['EMA']==.99
    return config


def build_original(condition, feature_dimension=1280):
    if type(feature_dimension) is not int or feature_dimension<1:
        raise ValueError('IIB-MIL feature width must be explicitly positive')
    config=source_config(condition);model=config['MODEL_IIBMIL']
    network,_,_=author_components()
    # Native1280 is the author feature width. Any other width is an explicit
    # FM interface port on all original input/residual projections, not a hidden
    # reduction of transformer width, depth, FFNs, decoder or learned queries.
    return network.IIBMIL(num_classes=model['NUM_CLASSES'],patch_dim=feature_dimension,
                           dim=model['HIDDEN_DIM'],depth=model['DEPTH'],num_queries=5)


def component_protocol(condition, feature_dimension=1280):
    config=source_config(condition);root,_=checked_source()
    return {'source':'https://github.com/TencentAILabHealthcare/IIB-MIL','commit':COMMIT,
        'condition':condition,'feature_dimension':feature_dimension,'native_feature_dimension':1280,
        'original_config':config,'source_sha256':{name:digest_file(root/name) for name in
            ['main/models/iibmil.py','main/util/loss.py','main/datasets/wsi_feat_dataset.py','main/engine.py','main/test.py']},
        'scope':'Complete original residual-transformer/query-decoder, prototype and confidence components only',
        'training_epoch_in_released_tree':'absent','runnable_profile_registered':False,
        'fidelity_notes':['Original attention ignores the dataset padding mask; no silent mask correction',
                          'Source candidate labels cover all classes; confidence starts from bag label with epsilon',
                          'Original prototype momentum scales by batch size / number of class-matched bags',
                          'Confidence EMA schedule is hardcoded200 in source, distinct from YAML300epochs',
                          'Training update order and omitted warmup logic require evidence before full admission'],
        'formal_admission':False,'real_wsi_validation':'not_run','paper_reproduction':'not_run'}
