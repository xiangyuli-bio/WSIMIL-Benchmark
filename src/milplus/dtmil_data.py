"""Original DT-MIL physical feature canvas and fully bound optional TRAIN views."""
import math
from functools import lru_cache

import h5py
import numpy as np
import torch

from .data import resolve
from .io import digest_file, digest_object
from .setmil import physical_indices
from .smmile_data import array_identity, load_spatial_inputs, positive_int
from .spatial_attention import spatial_metadata
from .upstream import load_definitions, load_single


@lru_cache(maxsize=1)
def augmentation():
    aug = load_single('dtmil', 'util/numpy_img_aug.py')
    transform = load_definitions('dtmil', 'datasets/wsi_feat_dataset.py', ['TrFeatureMapAug', 'ValFeatureMapAug'],
                                 {'np': np, 'tensor_aug': aug})
    return aug, transform


def encode_map(features, coordinates, stride, size, repaired, train=False, views=None, max_bytes=32*1024**3):
    positive_int(size, 'map_size'); positive_int(max_bytes, 'max_dense_bytes')
    if (features.ndim != 2 or min(features.shape) < 1 or not features.is_floating_point()
            or not torch.isfinite(features).all() or coordinates is None or len(coordinates) != len(features)):
        raise ValueError('DT-MIL needs finite full features and aligned physical coordinates')
    if isinstance(stride, bool) or not isinstance(stride, (int, float)) or not math.isfinite(stride) or stride <= 0:
        raise ValueError('DT-MIL patch stride must be finite positive')
    points = coordinates.detach().cpu().numpy()
    if (points < 0).any() or (points > 2**53-1).any():
        raise ValueError('DT-MIL physical coordinates must be nonnegative and exactly representable')
    grid = physical_indices(coordinates, stride)
    h, w = map(int, grid.max(0)+1); dimension = features.shape[1]
    if h*w*(dimension+1)*8+size*size*(dimension*32+16) > max_bytes:
        raise ValueError('DT-MIL physical canvas exceeds max_dense_bytes before allocation')
    values = features.detach().cpu().numpy()
    chosen_views = None
    if views is not None:
        views = np.asarray(views)
        if (views.ndim != 3 or views.shape[0] != len(values) or views.shape[2] != dimension
                or views.shape[1] < 1 or not np.issubdtype(views.dtype, np.floating) or not np.isfinite(views).all()):
            raise ValueError('DT-MIL TRAIN views require finite aligned [N,V,D]')
        chosen_views = [int(np.random.choice(v.shape[0], 1).item()) for v in views]
        values = np.stack([v[j] for v,j in zip(views, chosen_views)])
    canvas = np.zeros((h,w,dimension), dtype=np.float64)
    identities = np.full((h,w,1), -1, dtype=np.int64)
    canvas[grid[:,0],grid[:,1]] = values
    identities[grid[:,0],grid[:,1],0] = np.arange(len(grid))
    aug, transform = augmentation()
    if h > size:
        canvas = aug.center_crop(canvas, (size,w)); identities = aug.center_crop(identities, (size,w))
    if w > size:
        current_h = canvas.shape[0] if repaired else h
        canvas = aug.center_crop(canvas, (current_h,size)); identities = aug.center_crop(identities, (current_h,size))
    retained = identities[identities>=0].reshape(-1)
    if not len(retained):
        raise ValueError('DT-MIL declared physical crop retains no registered patch rows')
    metadata = {'source_canvas_shape':[h,w], 'cropped_shape':list(canvas.shape[:2]),
        'original_rows':len(features), 'retained_rows':len(retained),
        'retained_ids_sha256':array_identity(retained)['sha256'],
        'coordinate_sha256':array_identity(coordinates.detach().cpu().numpy())['sha256'],
        'views_sha256':digest_object(chosen_views), 'augmented':bool(train)}
    mask = np.zeros(canvas.shape[:2])  # Source treats interior empty cells as unmasked zero features.
    canvas, mask = aug.pad_image_and_mask(canvas, mask, size, image_fill_value=0, mask_fill_value=1)
    canvas, mask = (transform.TrFeatureMapAug() if train else transform.ValFeatureMapAug())(canvas,mask)
    tensor = torch.from_numpy(canvas.copy()).float().permute(2,0,1)
    mask = torch.from_numpy(mask.copy()).bool()
    if tensor.shape != (dimension,size,size) or mask.shape != (size,size) or mask.all() or not torch.isfinite(tensor).all():
        raise ValueError('DT-MIL original transformations returned invalid/fully padded physical maps')
    metadata['transformed_map_sha256'] = array_identity(tensor.numpy())['sha256']
    metadata['transformed_mask_sha256'] = array_identity(mask.numpy())['sha256']
    return tensor,mask,metadata


class SpatialInputs:
    def __init__(self, bundle, profile, map_size, max_bytes):
        self.bundle, self.profile, self.map_size, self.max_bytes = bundle,profile,map_size,max_bytes
        meta = spatial_metadata(bundle)
        if not isinstance(meta['provenance'],str) or not meta['provenance'].strip():
            raise ValueError('DT-MIL actual physical coordinate provenance must be nonempty')
        self.coordinate_unit = meta['unit']
        self.stride = meta.get('patch_stride',meta['patch_size'])
        if isinstance(self.stride,bool) or not isinstance(self.stride,(int,float)) or not math.isfinite(self.stride) or self.stride<=0:
            raise ValueError('DT-MIL requires positive declared physical patch_stride')
        self.cached = bundle.encoder.get('dtmil_cached_views')
        if self.cached is not None and (not isinstance(self.cached,dict) or set(self.cached)!={'dataset','provenance'}
                or any(not isinstance(v,str) or not v.strip() for v in self.cached.values())):
            raise ValueError('DT-MIL cached views require dataset and provenance in the same hashed HDF5')
        identities = []; content_hashes = {}
        for row in bundle.rows.to_dict('records'):
            feature = resolve(bundle.root,row['path'])
            coordinates = resolve(bundle.root,row.get('coords_path') or row['path'])
            for path in [feature,coordinates]:
                if path not in content_hashes:
                    content_hashes[path] = digest_file(path)
            identities.append({'slide_id':row['slide_id'],'feature_sha256':content_hashes[feature],
                'coordinate_sha256':content_hashes[coordinates]})
        self.contract = digest_object({'geometry':meta,'cached_views':self.cached,'rows':identities,
            'classes':bundle.classes,'dimension':bundle.dim,'profile':profile,'map_size':map_size})

    def row(self,row,train=False):
        features,coords = load_spatial_inputs(self.bundle,row)
        if self.coordinate_unit in {'level0_pixels','tile_grid'} and not torch.equal(coords,coords.round()):
            raise ValueError('DT-MIL pixel/tile-grid coordinates must be actual integer positions')
        views = self.read_views(row,len(features)) if train and self.cached is not None else None
        return encode_map(features,coords,self.stride,self.map_size,self.profile=='dtmil_repaired_reference_train',
                          train,views,self.max_bytes)

    def read_views(self,row,count):
        path=resolve(self.bundle.root,row['path'])
        if path.suffix not in {'.h5','.hdf5'}:
            raise ValueError('DT-MIL TRAIN views must share the hashed HDF5 file')
        with h5py.File(path,'r') as handle:
            if self.cached['dataset'] not in handle or not isinstance(handle[self.cached['dataset']],h5py.Dataset):
                raise ValueError('DT-MIL declared TRAIN views dataset missing or not a dataset')
            array=handle[self.cached['dataset']]
            if (array.ndim!=3 or array.shape[0]!=count or array.shape[2]!=self.bundle.dim
                    or array.shape[1]<1 or not np.issubdtype(array.dtype,np.floating)):
                raise ValueError('DT-MIL TRAIN views require floating aligned [N,V,D]')
            if array.size*array.dtype.itemsize>self.max_bytes:
                raise ValueError('DT-MIL cached views exceed max_dense_bytes')
            value=array[:]
        if not np.isfinite(value).all():
            raise ValueError('DT-MIL cached TRAIN views must be finite')
        return value
