"""Pack actual author-style x_y image folders, with source order and no resize."""
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from .io import digest_file, digest_object, write_json


def pack(table, output):
    table = Path(table).resolve(); rows = pd.read_csv(table, dtype=str)
    if (set(rows.columns) != {'slide_id', 'path'} or rows.empty or rows.isna().any().any()
            or rows.slide_id.duplicated().any() or any(not str(v).strip() for v in rows.to_numpy().flat)):
        raise ValueError('SparseConvMIL packing needs unique slide_id,path rows pointing to actual tile folders')
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    records, receipts = [], []
    for row in rows.to_dict('records'):
        folder = Path(row['path']); folder = folder if folder.is_absolute() else table.parent/folder
        folder = folder.resolve()
        if not folder.is_dir():
            raise ValueError('SparseConvMIL source tile folder does not exist')
        # The source uses os.listdir order, not lexical sorting. Persist this
        # actual order and original pixel SHA instead of silently changing the
        # input sequence before Python random.choices selects tile indices.
        paths = [folder/name for name in os.listdir(folder) if name.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not paths:
            raise ValueError('SparseConvMIL original tile folder is empty')
        coords = []
        for path in paths:
            try:
                pair = list(map(int, path.name.split('.')[0].split('_')))
            except ValueError as error:
                raise ValueError('SparseConvMIL original tile name must encode integer x_y coordinates') from error
            if len(pair) != 2 or min(pair) < 0:
                raise ValueError('SparseConvMIL original x_y coordinates must be nonnegative pairs')
            coords.append(pair)
        if len({tuple(pair) for pair in coords}) != len(coords):
            raise ValueError('SparseConvMIL original tile names repeat a coordinate')
        target = root/(digest_object({'slide_id': row['slide_id']})[:24]+'.h5')
        originals, shape = [], None
        with h5py.File(target, 'w') as archive:
            for index, path in enumerate(paths):
                with path.open('rb') as stream, Image.open(stream) as image:
                    rgb = np.asarray(image.convert('RGB'))
                if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.shape[0] != rgb.shape[1] or rgb.shape[0] < 32:
                    raise ValueError('SparseConvMIL requires full square RGB tiles; packing never resizes them')
                if shape is None:
                    shape = rgb.shape
                    pixels = archive.create_dataset('pixels', (len(paths), *shape), dtype=np.uint8,
                        chunks=(1, *shape), compression='gzip', compression_opts=1)
                if rgb.shape != shape:
                    raise ValueError('SparseConvMIL tiles in one bag have inconsistent original sizes')
                pixels[index] = rgb
                originals.append({'path': str(path), 'sha256': digest_file(path)})
            archive['coords'] = np.asarray(coords, dtype=np.int64)
            archive['patch_ids'] = np.asarray([str(p) for p in paths], dtype=h5py.string_dtype())
        records.append({'slide_id': row['slide_id'], 'path': str(target)})
        receipts.append({'slide_id': row['slide_id'], 'archive': str(target), 'sha256': digest_file(target),
                         'images': originals, 'shape': [len(paths), *shape]})
    index = root/'images.csv'; pd.DataFrame(records).to_csv(index, index=False)
    write_json(root/'packing.json', {'schema_version': 1, 'input_table': str(table),
        'input_sha256': digest_file(table), 'records': receipts,
        'coordinate_units': 'Retained source filenames; user must declare real level0 pixels or tile-grid indices in schema',
        'pixel_conversion': 'Exact original PIL convert RGB; no resize/normalization/stain modification',
        'patch_order': 'Actual original os.listdir order, persisted before sampling'})
    return index
