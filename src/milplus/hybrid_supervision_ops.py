"""Paper hybrid mechanisms and distinct released challenge postprocessing."""
import random

import numpy as np
import torch
from torch.nn import functional as F


def refine_labels(probability, image_label, *, quantized_source=False):
    if (not torch.is_tensor(probability) or probability.ndim < 2
            or not probability.is_floating_point() or not torch.isfinite(probability).all()
            or probability.min() < 0 or probability.max() > 1
            or type(image_label) is not int or image_label not in {0, 1}):
        raise ValueError('Hybrid refinement needs actual finite pixel probabilities and binary image labels')
    if image_label == 0:
        return torch.zeros_like(probability)
    if quantized_source:
        # Source PNG pseudo masks are truncated uint8 before dataloader scaling.
        value = torch.floor(probability*255).to(torch.float32)*4
        value = value.clamp(max=255)
        value = torch.where(value < 255*.01, 0, value)
        return value.floor().div(255).to(probability.dtype)
    # Paper Algorithm1: rescale, clip, then suppress values below.01.
    value = (probability*4).clamp(max=1)
    return torch.where(value < .01, 0, value)


def top_patch_indices(pixel_probability, k=16):
    if (type(k) is not int or k < 1 or pixel_probability.ndim != 3 or len(pixel_probability) < k or not torch.isfinite(pixel_probability).all()
            or pixel_probability.min() < 0 or pixel_probability.max() > 1):
        raise ValueError('Hybrid Top-K needs at least16 actual scored image patches; no padding or FM proxy')
    scores = pixel_probability.flatten(1).amax(1)
    # Stable ties follow the bound actual patch order.
    return torch.argsort(scores, descending=True, stable=True)[:k]


def image_objective(patch_logits, target):
    if (patch_logits.shape != (16, 1) or target.shape != (1,) or target.dtype != torch.long
            or int(target.item()) not in {0, 1} or not torch.isfinite(patch_logits).all()):
        raise ValueError('Hybrid paper classifier requires all16 actual DLA34 patch logits and one image label')
    probability = patch_logits.sigmoid().mean(0)
    return F.binary_cross_entropy(probability, target.to(probability.dtype)), probability


def paper_pool_draw(fine, positive, negative):
    if not fine or not positive or not negative:
        raise ValueError('Full hybrid EM training needs genuine fine, positive-pseudo and hard-negative pools')
    # Exact published R=2:1:7 categorical weights. Pool contents are TRAIN-only
    # and remain separately bound to their annotation/split provenance.
    draw = random.randrange(10)
    kind = 'fine' if draw < 2 else 'positive' if draw == 2 else 'negative'
    return kind, random.choice({'fine': fine, 'positive': positive, 'negative': negative}[kind])


def released_pool_draw(available):
    # Literal released dataloader randint is inclusive; edge cases40/50 belong
    # to fine labels. These differ from the paper2:1:7 and are not conflated.
    if not available.get('fine'):
        raise ValueError('Released hybrid initialization needs actual fine masks')
    draw = random.randint(0, 100)
    kind = ('negative' if draw < 40 and available.get('negative') else
            'positive' if 40 < draw < 50 and available.get('positive') else 'fine')
    return kind, random.choice(available[kind])


def released_heatmap(probabilities, coordinates):
    import cv2
    if (probabilities.dtype != np.uint8 or probabilities.ndim != 3
            or probabilities.shape[1:] != (1024, 1024)
            or coordinates.dtype != np.int64 or coordinates.shape != (len(probabilities), 2)
            or not len(probabilities) or (coordinates < 0).any()):
        raise ValueError('Released challenge heatmap requires ordered uint8 full1024 masks and actual level0coordinates')
    xmin, ymin = coordinates.min(0)
    width, height = (coordinates.max(0)-coordinates.min(0)+2048).tolist()
    heatmap = np.zeros((height//32, width//32), dtype=np.uint8)
    for mask, (x, y) in zip(probabilities, coordinates):
        x, y = int(x-xmin), int(y-ymin)
        resized = cv2.resize(mask, (32, 32))
        view = heatmap[y//32:(y+1024)//32, x//32:(x+1024)//32]
        if view.shape != resized.shape:
            raise ValueError('Released heatmap coordinate frame cannot be silently truncated')
        view[:] = resized
    return heatmap
