"""Original-scale RMDL pixels, annotation sampling and spatial instance selection.

The paper does not release its sampler, hard-negative trainer or NMS code.
Choices made here are explicit reference choices, never attributed to that code.
No WSI label is converted into a localization label.
"""
from pathlib import Path
import math

import h5py
import numpy as np
from PIL import Image, ImageOps

WORKING_MPP = 0.5034
PATCH = 299
TRAIN_CROP = 470
BLOCK = 1899
STRIDE = 32
BLOCK_STRIDE = 1632
PER_CLASS = 100


def materialized(file, name):
    link = file.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        raise ValueError('RMDL pixel and mask datasets must be materialized hard links')
    dataset = file[name]
    if not isinstance(dataset, h5py.Dataset) or dataset.is_virtual or dataset.external:
        raise ValueError('RMDL raw pixels and masks cannot use unbound external/virtual storage')
    return dataset


class Slide:
    """Read spatial regions at the declared physical scale, without loading a WSI."""
    def __init__(self, record):
        self.record = record
        self.width, self.height = record['working_size']
        self.handle = None
        if record['kind'] == 'working_rgb_hdf5':
            self.handle = h5py.File(record['path'], 'r')
            self.rgb = materialized(self.handle, 'rgb')
        elif record['kind'] == 'openslide':
            try:
                import openslide
            except ImportError as error:
                raise ImportError('Raw WSI RMDL needs openslide-python and its native library in the isolated TF environment') from error
            self.handle = openslide.OpenSlide(record['path'])
        else:
            raise ValueError('Use an actual OpenSlide WSI or spatial working-scale RGB HDF5')

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def read(self, x, y, size):
        if any(type(v) is not int for v in [x, y, size]) or size < 1:
            raise ValueError('Working-scale RMDL region coordinates must be integer pixels')
        # The full FCN uses complete1899 blocks; only its edge context may be
        # padded. Selected299 patches and training470 crops remain fully valid.
        output = np.full((size, size, 3), 255, dtype=np.uint8)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + size), min(self.height, y + size)
        if x0 >= x1 or y0 >= y1:
            return output
        if self.record['kind'] == 'working_rgb_hdf5':
            region = self.rgb[y0:y1, x0:x1]
        else:
            native_x, native_y = self.record['native_mpp']
            downsample = WORKING_MPP / min(native_x, native_y)
            level = self.handle.get_best_level_for_downsample(downsample)
            ds = self.handle.level_downsamples[level]
            extent = (math.ceil((x1-x0)*WORKING_MPP/(native_x*ds)),
                      math.ceil((y1-y0)*WORKING_MPP/(native_y*ds)))
            origin = (round(x0*WORKING_MPP/native_x), round(y0*WORKING_MPP/native_y))
            rgba = self.handle.read_region(origin, level, extent)
            background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            region = np.asarray(background.convert('RGB').resize((x1-x0, y1-y0), Image.Resampling.LANCZOS))
        output[y0-y:y1-y, x0-x:x1-x] = region
        return output


def annotation(record, xs, ys):
    """Read only explicitly TRAIN-bound pixel annotations, never task labels."""
    if record.get('subset') != 'train' or not record.get('annotation'):
        raise ValueError('RMDL localization supervision and mining are restricted to TRAIN pixel annotations')
    xs, ys = np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.int64)
    if xs.shape != ys.shape or np.any(xs < 0) or np.any(xs >= record['working_size'][0]) or np.any(ys < 0) or np.any(ys >= record['working_size'][1]):
        raise ValueError('Annotation query lies outside its actual working-scale slide')
    with h5py.File(record['annotation'], 'r') as file:
        mask = materialized(file, 'annotation')
        # h5py does not implement arbitrary paired two-dimensional indexing.
        values = np.asarray([mask[int(y), int(x)] for x, y in zip(xs.flat, ys.flat)], dtype=np.uint8)
    return values.reshape(xs.shape)


def training_candidates(records, seed, normals_per_slide=1024):
    """224-stride abnormal centers plus explicitly random annotated normals.

    Center-label assignment, a finite normal pool and balanced replacement
    sampling are necessary reference conventions absent from the released code.
    """
    if type(normals_per_slide) is not int or normals_per_slide < 1:
        raise ValueError('Declare the positive normal proposal budget')
    rng = np.random.RandomState(seed)
    ledger = []
    for sid in sorted(records):
        record = records[sid]
        if record['subset'] != 'train':
            continue
        width, height = record['working_size']
        if min(width, height) < TRAIN_CROP:
            raise ValueError('Native RMDL localization needs real470x470 training crops')
        centers_x = np.arange(235, width-234, 224, dtype=np.int64)
        centers_y = np.arange(235, height-234, 224, dtype=np.int64)
        xx, yy = np.meshgrid(centers_x, centers_y)
        labels = annotation(record, xx, yy)
        for x, y, label in zip(xx.flat, yy.flat, labels.flat):
            if label in {1, 2}:
                ledger.append({'slide_id': sid, 'center': [int(x), int(y)], 'target': int(label), 'origin': 'annotated_abnormal_grid224'})
        selected = set()
        # Finite and reproducible rejection sampling; never manufacture normal
        # labels when annotations contain no normal tissue.
        for unused in range(100):
            if len(selected) >= normals_per_slide:
                break
            count = max(256, 2*(normals_per_slide-len(selected)))
            xs = rng.randint(235, width-234, count)
            ys = rng.randint(235, height-234, count)
            labels = annotation(record, xs, ys)
            for x, y, target in zip(xs, ys, labels):
                if target == 0:
                    selected.add((int(x), int(y)))
                    if len(selected) == normals_per_slide:
                        break
        for x, y in sorted(selected, key=lambda p: (p[1], p[0])):
            ledger.append({'slide_id': sid, 'center': [x, y], 'target': 0, 'origin': 'annotated_random_normal'})
    if {row['target'] for row in ledger} != {0, 1, 2}:
        raise ValueError('Actual TRAIN pixel annotations must supply normal, dysplasia and cancer; slide labels are not a substitute')
    return ledger


def balanced_batch(ledger, seed, step, batch=80):
    if type(batch) is not int or batch < 3:
        raise ValueError('Localization batch must represent all three actual patch classes')
    rng = np.random.RandomState(np.random.SeedSequence([seed, step, 8101]).generate_state(1)[0])
    indices = []
    # Native80 is indivisible by3: rotate the27/27/26 remainder over3updates.
    order = [(step+i) % 3 for i in range(3)]
    for rank, target in enumerate(order):
        pool = np.asarray([i for i, row in enumerate(ledger) if row['target'] == target])
        if not len(pool):
            raise ValueError('Localization sampling cannot fill an unannotated class')
        count = batch//3 + int(rank < batch % 3)
        indices.extend(rng.choice(pool, size=count, replace=True).tolist())
    return rng.permutation(indices).astype(np.int64)


def augmented_crop(slide, center, rng):
    x, y = center
    if x < 235 or y < 235 or x+235 > slide.width or y+235 > slide.height:
        raise ValueError('RMDL augmentation needs actual complete470pixels, without synthetic edge padding')
    image = Image.fromarray(slide.read(int(x)-235, int(y)-235, TRAIN_CROP))
    scaled = int(round(TRAIN_CROP * rng.uniform(.9, 1.1)))
    image = image.resize((scaled, scaled), Image.Resampling.BICUBIC)
    image = image.rotate(float(rng.uniform(0., 360.)), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255,255,255))
    if rng.rand() < .5:
        image = ImageOps.mirror(image)
    if rng.rand() < .5:
        image = ImageOps.flip(image)
    offset = (scaled-PATCH)//2
    return np.asarray(image.crop((offset, offset, offset+PATCH, offset+PATCH)), dtype=np.uint8)


def grid_shape(width, height):
    if type(width) is not int or type(height) is not int or min(width, height) < PATCH:
        raise ValueError('Actual RMDL WSI must contain a full299pixel instance')
    return ((height-PATCH)//STRIDE+1, (width-PATCH)//STRIDE+1)


def select_instances(probabilities):
    """Three independent spatial NMS passes, each retaining100 actual299 patches.

    Overlap is intersection / equal299-square area, with ties ordered row-major.
    This explicit area-ratio convention is not an unpublished author algorithm.
    The same physical coordinate may legitimately occur in different classes.
    """
    p = np.asarray(probabilities)
    if p.ndim != 3 or p.shape[-1] != 3 or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(-1), 1., atol=2e-5, rtol=0):
        raise ValueError('RMDL selection requires actual three-class FCN probability maps')
    yy, xx = np.indices(p.shape[:2]);x = xx.ravel()*STRIDE;y = yy.ravel()*STRIDE
    rows = []
    for target in range(3):
        scores = p[..., target].ravel()
        order = np.lexsort((np.arange(len(scores)), -scores))
        kept = []
        while len(order) and len(kept) < PER_CLASS:
            index = int(order[0]);kept.append(index);order = order[1:]
            intersection = np.maximum(0, PATCH-np.abs(x[order]-x[index]))*np.maximum(0, PATCH-np.abs(y[order]-y[index]))
            order = order[intersection.astype(np.float64)/(PATCH*PATCH) <= .68]
        if len(kept) != PER_CLASS:
            raise ValueError('Actual slide has fewer than100 spatially admissible patches per class; no duplication or padding is allowed')
        rows.extend({'selection_class': target, 'grid_index': index, 'xy': [int(x[index]), int(y[index])],
                     'score': float(scores[index])} for index in kept)
    return rows


def preprocess(images):
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError('Original InceptionResNet preprocessing needs actual uint8 RGB images')
    return images.astype(np.float32)/np.float32(127.5)-np.float32(1.)
