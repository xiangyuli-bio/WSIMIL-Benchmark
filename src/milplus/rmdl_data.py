"""Patient-bound full-slide and TRAIN-only pixel-supervision contract for RMDL."""
import copy
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .io import digest_file, digest_object
from .rmdl_pixels import WORKING_MPP, materialized
from .rmdl_source import IMAGENET_MD5, IMAGENET_SHA256


def resolve(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root/path).resolve()


def _identity(bundle):
    return {'task_id': bundle.config['task_id'], 'task_type': bundle.task_type,
            'classes': bundle.classes, 'fm': bundle.fm, 'split': bundle.split,
            'dimension': bundle.dim, 'rows': bundle.rows.to_dict('records')}


def _file(path):
    stat = path.stat()
    if not path.is_file():
        raise ValueError('RMDL requires materialized original files')
    return {'path': str(path), 'sha256': digest_file(path),
            'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _inspect(bundle, row, declared, files):
    sid = row['slide_id']
    required = {'slide_id','kind','working_mpp','annotation','provenance'}
    if (not isinstance(declared, dict) or set(declared) != required or declared['slide_id'] != sid
            or not isinstance(declared['provenance'], str) or not declared['provenance'].strip()
            or type(declared['working_mpp']) not in {int,float} or declared['working_mpp'] != WORKING_MPP):
        raise ValueError('Declare actual original-scale RMDL slide geometry, provenance and TRAIN pixel annotation path')
    path = resolve(bundle.root, row['path']);file = _file(path);files[str(path)] = file
    record = {**copy.deepcopy(declared), **file, 'global_patient_id': row['global_patient_id'], 'subset': row['subset']}
    if declared['kind'] == 'working_rgb_hdf5':
        with h5py.File(path, 'r') as handle:
            rgb = materialized(handle, 'rgb')
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 299:
                raise ValueError('RMDL requires complete spatial uint8 RGB slides, not frozen features or bags of unrelated patches')
            if float(handle.attrs.get('working_mpp', -1)) != WORKING_MPP:
                raise ValueError('Materialized RMDL spatial pixels must carry the exact declared physical scale')
            height, width = map(int, rgb.shape[:2])
            if row['subset'] != 'train' and 'annotation' in handle:
                raise ValueError('Remove held-out pixel annotations from the RMDL experiment input; only TRAIN localization supervision is admitted')
        record.update(working_size=[width,height], native_mpp=[WORKING_MPP,WORKING_MPP])
    elif declared['kind'] == 'openslide':
        try:
            import openslide
        except ImportError as error:
            raise ImportError('RMDL original WSI inventory needs openslide-python and its native library') from error
        with openslide.OpenSlide(str(path)) as slide:
            try:
                mx = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
                my = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
            except (KeyError,ValueError) as error:
                raise ValueError('Actual WSI physical MPP metadata is required; magnification names alone do not define the native field of view') from error
            if not math.isfinite(mx) or not math.isfinite(my) or min(mx,my) <= 0:
                raise ValueError('Invalid native WSI physical scale')
            width,height = slide.dimensions
        record.update(native_mpp=[mx,my],working_size=[int(width*mx/WORKING_MPP),int(height*my/WORKING_MPP)])
        if min(record['working_size']) < 299:
            raise ValueError('Actual native WSI cannot supply complete299pixels at the required physical scale')
    else:
        raise ValueError('RMDL accepts complete OpenSlide WSIs or losslessly materialized spatial RGB at the working scale')
    mask = declared['annotation']
    if row['subset'] != 'train':
        if mask is not None:
            raise ValueError('Validation/test pixel annotation paths cannot enter localization fitting or hard-negative mining')
        record['annotation'] = None
    else:
        if not isinstance(mask, str) or not mask.strip():
            raise ValueError('Every TRAIN slide needs actual pixel labels0normal/1dysplasia/2cancer/255ignore, not inferred slide labels')
        mask = resolve(bundle.root, mask);files[str(mask)] = _file(mask)
        with h5py.File(mask, 'r') as handle:
            pixels = materialized(handle, 'annotation')
            if pixels.dtype != np.uint8 or pixels.shape != tuple(reversed(record['working_size'])):
                raise ValueError('Actual localization annotation must align pixel-for-pixel with the working-scale WSI')
            if float(handle.attrs.get('working_mpp', -1)) != WORKING_MPP:
                raise ValueError('Localization annotation physical scale differs from its slide')
            values = set()
            for offset in range(0, pixels.shape[0], 1024):
                values.update(map(int, np.unique(pixels[offset:offset+1024])))
                if not values <= {0,1,2,255}:
                    raise ValueError('Unknown localization mask class; unlabeled pixels must be255 rather than guessed negatives')
        record.update(annotation=str(mask),annotation_classes_present=sorted(values))
    return record


class RMDLInputs:
    def __init__(self, bundle):
        schema = bundle.encoder.get('rmdl_raw')
        keys = {'schema_version','kind','records','imagenet_weights','image_provenance','annotation_provenance'}
        if (not isinstance(schema,dict) or set(schema) != keys or schema['schema_version'] != 1
                or schema['kind'] != 'rmdl_original_scale_WSI_and_TRAIN_pixel_masks'
                or bundle.task_type != 'classification' or bundle.classes != ['normal','dysplasia','cancer']
                or bundle.dim != 1024 or bundle.config.get('training_unit','slide') != 'slide'
                or bundle.config.get('evaluation_unit','slide') != 'slide'
                or any(not isinstance(schema[k],str) or not schema[k].strip() for k in keys-{'schema_version'})):
            raise ValueError('RMDL native three-grade complete method needs original-scale slides, explicit pixel supervision and full1024D FConv; no generic FM feature substitution')
        self.bundle = bundle
        index = resolve(bundle.root, schema['records'])
        self.files = [index]
        self.file_records = {str(index):_file(index)}
        declared = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
        if (any(not isinstance(r,dict) or not isinstance(r.get('slide_id'),str) for r in declared)
                or len({r['slide_id'] for r in declared}) != len(declared)
                or {r['slide_id'] for r in declared} != set(bundle.rows.slide_id)):
            raise ValueError('Every RMDL slide needs exactly one actual raw image/annotation record')
        declared = {r['slide_id']:r for r in declared}
        self.records = {row['slide_id']:_inspect(bundle,row,declared[row['slide_id']],self.file_records)
                        for row in bundle.rows.to_dict('records')}
        present=set().union(*(set(r.get('annotation_classes_present',[])) for r in self.records.values()))
        if not {0,1,2} <= present:
            raise ValueError('RMDL TRAIN pixel annotations do not contain all three localization classes')
        hashes = {}
        for record in self.records.values():
            previous = hashes.setdefault(record['sha256'], record['global_patient_id'])
            if previous != record['global_patient_id']:
                raise ValueError('Identical raw WSI bytes are assigned to different patients')
        self.weights = resolve(bundle.root, schema['imagenet_weights'])
        digest = hashlib.md5()
        with self.weights.open('rb') as stream:
            for block in iter(lambda:stream.read(8*2**20), b''):
                digest.update(block)
        if digest.hexdigest() != IMAGENET_MD5 or digest_file(self.weights) != IMAGENET_SHA256:
            raise ValueError('RMDL requires the verified complete official Keras InceptionResNetV2 no-top ImageNet initialization')
        self.file_records[str(self.weights)] = _file(self.weights)
        self.files.extend(Path(p) for p in self.file_records if Path(p) not in self.files)
        common = [bundle.root/'task.yaml',resolve(bundle.root,bundle.config.get('slides','slides.csv')),
                  resolve(bundle.root,bundle.config['splits'][bundle.split]),resolve(bundle.root,bundle.encoder['index'])]
        for path in common:
            self.file_records[str(path)] = _file(path)
        self.value = {'schema_version':1,'kind':schema['kind'],'schema':copy.deepcopy(schema),
                      'identity':_identity(bundle),'records':self.records,'files':self.file_records,
                      'weights':str(self.weights),'synthetic':bool(bundle.config.get('synthetic',False)),
                      'annotation_scope':'TRAIN_only_pixel_labels','paper_score_reproduction':False}
        self.contract = digest_object(self.value)

    def check(self, deep=False):
        if self.value['identity'] != _identity(self.bundle):
            raise ValueError('RMDL task/labels/partitions/patient identities changed')
        check_binding(self.export_unchecked(), deep=deep)

    def export_unchecked(self):
        return {**self.value,'input_contract':self.contract}

    def export(self):
        self.check(deep=True)
        return self.export_unchecked()

    def inventory(self, row, deep=False):
        self.check(deep=deep)
        return self.records[row['slide_id']]


def check_binding(binding, deep=False):
    value = {k:v for k,v in binding.items() if k != 'input_contract'}
    if binding.get('input_contract') != digest_object(value):
        raise ValueError('RMDL complete raw input binding was changed')
    for name, record in binding['files'].items():
        stat = Path(name).stat()
        if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns'] or (deep and digest_file(name) != record['sha256']):
            raise ValueError('Actual RMDL WSI/annotation/ImageNet/task bytes changed: '+name)
