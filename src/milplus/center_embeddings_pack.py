"""Read exact original level-agnostic WSI centers into reproducible256RGB caches."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from .data import load_array
from .io import digest_file
from .upstream import load_definitions


def norm_coord(center, level, mult=1):
    source = load_definitions('center_embeddings', 'research_mil/loaders/datasets.py', ['MILdataset'],
        {'Dataset': Dataset, 'torch': torch, 'np': np})
    context = SimpleNamespace(level=level, size=int(np.round(256*mult)))
    return source.MILdataset.norm_coord(context, center)


def pack(table, output, level=2, mult=1):
    if type(level) is not int or level < 0 or type(mult) not in {int, float} or not np.isfinite(mult) or mult < 1:
        raise ValueError('Center Embeddings source WSI level and size multiplier must be valid')
    import openslide
    table = Path(table).expanduser().resolve()
    frame = pd.read_csv(table, dtype=str, keep_default_na=False)
    required = ['slide_id', 'wsi', 'coordinates']
    if (not set(required) <= set(frame) or frame[required].eq('').any().any()
            or frame.slide_id.duplicated().any()
            or any('/' in name or name in {'.', '..'} for name in frame.slide_id)):
        raise ValueError('Center Embeddings pack table needs unique safe slide_id, real WSI and coordinate paths')
    output = Path(output).expanduser().resolve(); output.mkdir(parents=True, exist_ok=False)
    records = []
    source_norm = load_definitions('center_embeddings', 'research_mil/loaders/datasets.py', ['MILdataset'],
        {'Dataset': Dataset, 'torch': torch, 'np': np}).MILdataset.norm_coord
    context = SimpleNamespace(level=level, size=int(np.round(256*mult)))
    for row in frame.to_dict('records'):
        wsi = Path(row['wsi']).expanduser(); coords_path = Path(row['coordinates']).expanduser()
        wsi = wsi if wsi.is_absolute() else table.parent/wsi
        coords_path = coords_path if coords_path.is_absolute() else table.parent/coords_path
        coords = load_array(coords_path, key='coords')
        if (coords.ndim != 2 or coords.shape[1] != 2 or not len(coords)
                or not np.issubdtype(coords.dtype, np.integer) or (coords < 0).any()
                or len(np.unique(coords, axis=0)) != len(coords)):
            raise ValueError('Center Embeddings needs actual unique level0 center coordinates, not a fabricated grid')
        provenance = {'wsi': str(wsi.resolve()), 'wsi_sha256': digest_file(wsi),
            'coordinates': str(coords_path.resolve()), 'coordinate_file_sha256': digest_file(coords_path),
            'reader': 'Original source MILdataset.norm_coord + OpenSlide.read_region(...).convert(RGB)',
            'read_level': level, 'mult': mult, 'read_size': context.size,
            'resize': 'Original Image.NEAREST to256 only when mult!=1'}
        path = output/(row['slide_id']+'.h5')
        with openslide.OpenSlide(str(wsi)) as source, h5py.File(path, 'x') as archive:
            if level >= source.level_count:
                raise ValueError('Requested original extraction level does not exist in the actual WSI')
            pixels = archive.create_dataset('pixels', (len(coords), 256, 256, 3), dtype=np.uint8,
                chunks=(1, 256, 256, 3), compression='gzip', compression_opts=1)
            for i, center in enumerate(coords):
                image = source.read_region(source_norm(context, center), level, (context.size, context.size)).convert('RGB')
                if mult != 1:
                    image = image.resize((256, 256), Image.Resampling.NEAREST)
                pixels[i] = np.asarray(image, dtype=np.uint8)
            archive['coords'] = coords
            archive['patch_ids'] = np.asarray([f"{row['slide_id']}/center_{int(x)}_{int(y)}" for x, y in coords], dtype=h5py.string_dtype())
            archive.attrs['extraction_provenance'] = json.dumps(provenance, sort_keys=True)
        records.append({'slide_id': row['slide_id'], 'path': str(path)})
    index = output/'images.csv'; pd.DataFrame(records).to_csv(index, index=False)
    return index
