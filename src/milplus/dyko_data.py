"""Actual native visual alignment, source concepts and patient-safe shot plans."""
from pathlib import Path

import pandas as pd

from .io import digest_file, digest_object, read_json, write_json
from .dyko_assets import OriginalTITANAssets
from .dyko_source import source_assets, original_shot_selection

SOURCE_ROLES = {
    'CAMELYON16':['normal','metastasis'],
    'NSCLC':['LUSC','LUAD'],
    'RCC':['KIRC','KICH','KIRP'],
    'UBC':['CC','EC','HGSC','LGSC','MC'],
}
VISUAL_WEIGHT_SHA256 = '09daf1912a016512e084636f9c76eee7c42f5c1f09b0461bb76082a3e0fecfa5'


def resolve(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root/path).resolve()


def identity(bundle):
    return {'task_id':bundle.config['task_id'],'task_type':bundle.task_type,'classes':bundle.classes,
            'dimension':bundle.dim,'fm':bundle.fm,'split':bundle.split,
            'training_unit':bundle.config.get('training_unit','slide'),
            'evaluation_unit':bundle.config.get('evaluation_unit','slide'),
            'rows':bundle.rows.to_dict('records')}


def class_mapping(bundle, cohort, mapping):
    if cohort not in SOURCE_ROLES or (bundle.task_type != 'classification' or bundle.dim != 768
            or len(bundle.classes) != len(SOURCE_ROLES[cohort])
            or bundle.config.get('training_unit','slide') != 'slide'
            or bundle.config.get('evaluation_unit','slide') != 'slide'):
        raise ValueError('DyKo requires native768D aligned bags and explicit released slide-classification cohorts')
    roles = SOURCE_ROLES[cohort]
    if (not isinstance(mapping,dict) or any(not isinstance(v,str) or not v for v in mapping.values())
            or set(mapping) != set(roles)
            or set(mapping.values()) != set(bundle.classes) or len(set(mapping.values())) != len(roles)):
        raise ValueError('Bind every ordered original DyKo prompt class to exactly one benchmark label')
    # Keep the model's original label/prompt order. Only the boundary between
    # that order and the common benchmark label order is translated.
    source_to_common = [bundle.classes.index(mapping[role]) for role in roles]
    common_to_source = [source_to_common.index(i) for i in range(len(roles))]
    return source_to_common,common_to_source


def shot_plan(rows, common_to_source, n_shot, seed):
    if sorted(common_to_source) != list(range(len(common_to_source))):
        raise ValueError('DyKo semantic label mapping must be a complete class permutation')
    if len({r['slide_id'] for r in rows}) != len(rows):
        raise ValueError('Actual DyKo slide identities must be unique')
    patients = {}
    for row in rows:
        previous = patients.setdefault(row['global_patient_id'],row['subset'])
        if previous != row['subset']:
            raise ValueError('DyKo few-shot input leaks a patient across partitions')
    train = [row for row in rows if row['subset'] == 'train']
    if not train or any(row['subset'] not in {'train','val','test'} for row in rows):
        raise ValueError('DyKo needs an explicit TRAIN/VAL/TEST patient partition')
    frame = pd.DataFrame({'train_slide_id':[r['slide_id'] for r in train],
                          'train_label':[common_to_source[int(r['target'])] for r in train]})
    selected = original_shot_selection(frame,n_shot,seed)
    by_slide = {r['slide_id']:r for r in train}
    output = []
    for row in selected.to_dict('records'):
        actual = by_slide[row['train_slide_id']]
        output.append({'slide_id':actual['slide_id'],'global_patient_id':actual['global_patient_id'],
                       'common_target':int(actual['target']),'source_target':int(row['train_label'])})
    return {'algorithm':'actual_author_per_source_class_pandas_sample_min_n_random_state_seed',
            'n_shot':n_shot,'seed':seed,'selected':output,
            'actual_slide_count':len(output),'actual_patient_count':len({r['global_patient_id'] for r in output}),
            'class_slide_counts':{str(i):sum(r['source_target']==i for r in output) for i in range(len(common_to_source))},
            'heldout_selection':False,'source_shot_cache_reused':False}


def _visual_metadata(bundle, path):
    value = read_json(path)
    if (value.get('encoder_id') != 'CONCHv1.5' or value.get('feature_dim') != 768
            or value.get('weights_sha256') != VISUAL_WEIGHT_SHA256
            or value.get('native_feature_space') != 'TITAN_CONCHv15'
            or not isinstance(value.get('extraction_provenance'),str)
            or not value['extraction_provenance'].strip()):
        raise ValueError('DyKo needs actual original CONCHv1.5/TITAN-aligned features and extraction provenance;768D shape alone is insufficient')
    return value


def prepare(bundle, schema, output):
    keys = {'schema_version','cohort','class_map','titan_assets','visual_extraction_manifest','provenance'}
    if (not isinstance(schema,dict) or set(schema) != keys or schema['schema_version'] != 1
            or not isinstance(schema['provenance'],str) or not schema['provenance'].strip()):
        raise ValueError('Declare the full DyKo visual/text/label/source input contract')
    if 'dyko' in bundle.encoder:
        raise ValueError('Prepare from the original unbound feature index, before adding encoder.dyko')
    cohort = schema['cohort'];source_to_common,common_to_source = class_mapping(bundle,cohort,schema['class_map'])
    titan = OriginalTITANAssets(resolve(bundle.root,schema['titan_assets']))
    visual = resolve(bundle.root,schema['visual_extraction_manifest']);visual_value = _visual_metadata(bundle,visual)
    author = source_assets(cohort)
    files = {str(titan.manifest):digest_file(titan.manifest),str(visual):digest_file(visual),
             str(author['concepts']):author['concepts_sha256'],str(author['prompts']):author['prompts_sha256']}
    records = {}
    for row in bundle.rows.to_dict('records'):
        x,unused = bundle.load(row)
        path = resolve(bundle.root,row['path']);stat = path.stat()
        records[row['slide_id']] = {'path':str(path),'sha256':digest_file(path),'bytes':stat.st_size,
            'mtime_ns':stat.st_mtime_ns,'instances':len(x),'feature_dim':768,'subset':row['subset'],
            'global_patient_id':row['global_patient_id'],'common_target':int(row['target']),
            'source_target':common_to_source[int(row['target'])]}
    value = {'schema_version':1,'kind':'dyko_native_visual_concepts_TITAN_and_patient_partitions',
        'identity':identity(bundle),'schema':schema,'cohort':cohort,'class_map':schema['class_map'],
        'source_to_common':source_to_common,'common_to_source':common_to_source,
        'titan_assets':str(titan.manifest),'titan_asset_contract':titan.contract,
        'visual_metadata':visual_value,'concepts':str(author['concepts']),'prompts':str(author['prompts']),
        'concepts_shape':list(author['bank'].shape),'records':records,'files':files,
        'provenance':schema['provenance'],'full_method_validation':False}
    value['contract'] = digest_object(value)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    write_json(output,value)
    return output


class DyKoInputs:
    def __init__(self, bundle):
        config = bundle.encoder.get('dyko')
        if not isinstance(config,dict) or set(config) != {'manifest'}:
            raise ValueError('Bind encoder.dyko.manifest to the complete native input contract')
        self.bundle = bundle;self.path = resolve(bundle.root,config['manifest'])
        value = read_json(self.path);self.contract = value.pop('contract',None)
        if (value.get('schema_version') != 1 or value.get('kind') != 'dyko_native_visual_concepts_TITAN_and_patient_partitions'
                or self.contract != digest_object(value) or value.get('identity') != identity(bundle)):
            raise ValueError('DyKo native inputs, source concepts, labels or patient partitions changed')
        mapping = class_mapping(bundle,value['cohort'],value['class_map'])
        if (value['source_to_common'],value['common_to_source']) != mapping:
            raise ValueError('DyKo original prompt-to-class semantics changed')
        self.value = value;self.records = value['records'];self.cohort = value['cohort']
        self.source_to_common,self.common_to_source = mapping
        self.titan = OriginalTITANAssets(value['titan_assets'])
        if self.titan.contract != value['titan_asset_contract']:
            raise ValueError('DyKo pretrained TITAN initialization identity changed')
        source = source_assets(self.cohort)
        if (value['concepts_shape'] != list(source['bank'].shape)
                or value['concepts'] != str(source['concepts']) or value['prompts'] != str(source['prompts'])):
            raise ValueError('DyKo original full ordered concept or prompt source changed')
        self.files = [self.path,*[Path(p) for p in value['files']]]
        self.verify(deep=True)

    def verify(self, deep=False):
        if read_json(self.path) != {**self.value,'contract':self.contract} or self.value['identity'] != identity(self.bundle):
            raise ValueError('DyKo method/input binding changed')
        self.titan.verify(deep=deep)
        for path,sha in self.value['files'].items():
            if digest_file(path) != sha:
                raise ValueError('DyKo native source concepts/prompts/alignment metadata changed')
        if set(self.records) != set(self.bundle.rows.slide_id):
            raise ValueError('Actual DyKo WSI population changed')
        for row in self.bundle.rows.to_dict('records'):
            r = self.records[row['slide_id']];path = resolve(self.bundle.root,row['path']);stat = path.stat()
            if (str(path) != r['path'] or r['subset'] != row['subset'] or r['global_patient_id'] != row['global_patient_id']
                    or r['common_target'] != int(row['target']) or r['source_target'] != self.common_to_source[int(row['target'])]
                    or stat.st_size != r['bytes'] or stat.st_mtime_ns != r['mtime_ns']
                    or (deep and digest_file(path) != r['sha256'])):
                raise ValueError('Actual DyKo feature bytes, patient membership or semantic label order changed')

    def plan(self, n_shot, seed):
        self.verify()
        return shot_plan(self.bundle.rows.to_dict('records'),self.common_to_source,n_shot,seed)

    def source_target(self, row):
        return self.common_to_source[int(row['target'])]

    def common_probabilities(self, tensor):
        # common_to_source maps each common output column to its native column.
        return tensor[...,self.common_to_source]
