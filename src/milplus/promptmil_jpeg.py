"""Original200-tile JPEG storage and original OpenCV fallback decoding.

This is a second explicit native input container, not an HDF5 shape bypass,
runtime resizer, alternate foundation model or reconstructed patch sequence.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .data import resolve
from .io import digest_file, digest_object, read_json
from .promptmil_source import COMMIT, checked_source
from .upstream import load_definitions

STORAGE = 'source_merged_jpeg_rgb224'
KIND = 'promptmil_original_200_tile_merged_jpeg_bag'


def runtime_identity():
    import cv2
    return {'opencv_version': cv2.__version__,
            'opencv_build_sha256': digest_object(cv2.getBuildInformation())}


def check_schema(schema):
    actual = runtime_identity()
    if (schema.get('storage') != STORAGE or schema.get('merge_batch') != 200
            or schema.get('jpeg_decoder') != 'source_cv2_fallback'
            or schema.get('source_commit') != COMMIT
            or any(schema.get(key) != value for key, value in actual.items())
            or not isinstance(schema.get('jpeg_protocol_sha256'), str)
            or len(schema['jpeg_protocol_sha256']) != 64):
        raise ValueError('Prompt-MIL merged JPEG requires actual pinned OpenCV/source/200-tile protocol')


def reader(schema):
    import cv2
    check_schema(schema)
    checked_source()
    # Explicit original optional-dependency fallback, never a replacement PIL
    # decoder or a claim of recovering the author's undisclosed jpeg4py build.
    return load_definitions('promptmil', 'dataset/merge_patch_wsi_dataset.py',
        ['read_rgb_img'], {'cv2': cv2, 'use_jpeg4py': False, 'jpeg': None}).read_rgb_img


def describe(path, schema):
    check_schema(schema)
    path = Path(path)
    if path.suffix.lower() != '.json':
        raise ValueError('Prompt-MIL merged JPEG requires an explicit original ordered bag index')
    value = read_json(path)
    ids = value.get('patch_ids')
    if (value.get('kind') != KIND or value.get('schema_version') != 1
            or value.get('protocol_sha256') != schema['jpeg_protocol_sha256']
            or not isinstance(ids, list) or not ids
            or len(ids) != len(set(ids))
            or any(not isinstance(item, str) or not item.strip() for item in ids)):
        raise ValueError('Prompt-MIL JPEG original protocol or unique patch identities changed')
    chunks = value.get('chunks')
    if not isinstance(chunks, list) or len(chunks) != (len(ids) + 199) // 200:
        raise ValueError('Prompt-MIL JPEG original200-tile chunk coverage changed')
    records = []
    paths = set()
    for number, chunk in enumerate(chunks):
        count = min(200, len(ids) - number * 200)
        filename = resolve(path.parent, chunk['path']).resolve()
        if (chunk.get('offset') != number * 200 or chunk.get('count') != count
                or filename in paths or filename.suffix.lower() not in {'.jpg', '.jpeg'}):
            raise ValueError('Prompt-MIL JPEG chunks must retain exact original order/count, without reuse')
        paths.add(filename)
        if digest_file(filename) != chunk['sha256']:
            raise ValueError('Prompt-MIL original merged JPEG bytes changed')
        with Image.open(filename) as image:
            if image.format != 'JPEG' or image.mode != 'RGB' or image.size != (224, count * 224):
                raise ValueError('Prompt-MIL original merged JPEG geometry/color changed')
        records.append({'path': str(filename), 'sha256': chunk['sha256'],
                        'offset': number * 200, 'count': count})
    return {'shape': [len(ids), 224, 224, 3], 'ordered_patch_ids_sha256': digest_object(ids),
            'jpeg_chunks': records, 'jpeg_protocol_sha256': schema['jpeg_protocol_sha256']}


def pixels(path, schema):
    record = describe(path, schema)
    read = reader(schema)
    tiles = []
    for chunk in record['jpeg_chunks']:
        image = read(chunk['path'])
        if image.shape != (chunk['count'] * 224, 224, 3) or image.dtype != np.uint8:
            raise ValueError('Actual original OpenCV JPEG reader returned wrong native pixels')
        tiles.append(image.reshape(chunk['count'], 224, 224, 3))
    return np.concatenate(tiles, axis=0)
