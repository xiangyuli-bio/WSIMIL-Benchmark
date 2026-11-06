"""Pack original author-format path/wsi CSV tiles without changing their pixels."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .io import digest_file, digest_object, write_json


def pack(table, output):
    try:
        import cv2
    except ImportError as error:
        raise ValueError('C2C original-tile packing requires the optional opencv-python-headless4.10.0.84 input tool') from error
    table, output = Path(table).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve existing image packs; choose a new output directory')
    rows = pd.read_csv(table, dtype=str, keep_default_na=False)
    if (not {'wsi', 'path'} <= set(rows) or rows[['wsi', 'path']].eq('').any().any()
            or rows[['wsi', 'path']].duplicated().any()):
        raise ValueError('Original C2C table must declare each wsi/path tile exactly once')
    output.mkdir(parents=True); index, records = [], {}
    for slide, images in rows.groupby('wsi', sort=True):
        paths = [Path(value) if Path(value).is_absolute() else table.parent/value for value in images.path]
        path = output/(digest_object(slide)+'.h5')
        source_records, dimensions = [], None
        with h5py.File(path, 'x') as archive:
            pixels = None
            for number, source in enumerate(paths):
                value = cv2.imread(str(source))
                if value is None:
                    raise ValueError(f'C2C original tile cannot be decoded: {source}')
                value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
                if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
                    raise ValueError('C2C original source decoder must return actual uint8 RGB')
                if dimensions is None:
                    dimensions = value.shape
                    pixels = archive.create_dataset('pixels', (len(paths), *dimensions), dtype=np.uint8,
                        chunks=(1, *dimensions), compression='gzip', compression_opts=1)
                if value.shape != dimensions:
                    raise ValueError('Original C2C training stacks one WSI of equal-size tiles; no silent resizing is allowed')
                pixels[number] = value
                source_records.append({'path': str(source.resolve()), 'sha256': digest_file(source)})
            archive['patch_ids'] = np.asarray([str(source.resolve()) for source in paths], dtype=h5py.string_dtype())
        index.append({'slide_id': slide, 'path': str(path)})
        records[slide] = {'archive': str(path), 'sha256': digest_file(path),
                          'shape': [len(paths), *dimensions], 'source_tiles': source_records}
    pd.DataFrame(index).to_csv(output/'images.csv', index=False)
    write_json(output/'pack_receipt.json', {'state': 'packed', 'original_table': str(table),
        'original_table_sha256': digest_file(table), 'opencv_version': cv2.__version__,
        'decoder': 'literal author cv2.imread(default flags), cvtColor(BGR2RGB); no resize, crop or normalization',
        'within_slide_order': 'original CSV row order', 'slides': records,
        'task_labels_and_splits': 'Not inferred; use separately audited patient/label/split tables'})
    return output/'images.csv'
