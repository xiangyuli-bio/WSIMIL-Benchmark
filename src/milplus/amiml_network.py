"""Pinned AMIML, with explicitly named input/output repairs and ragged BN.

The released default produces eight probabilities, incompatible with its binary
weighted CE. The paper says attention returns two-dimensional instance vectors.
Projecting V from eight to C implements that statement; the missing binary head
is a declared reconstruction, not a recovered unpublished implementation.
"""
import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions, verified_source

COMMIT = '7c038261d1a52a4eb20d4b4f444e730c36bf5cd2'
PAPER = 'amiml_paper_reference'
SOURCE = 'amiml_source_repaired'


@lru_cache(maxsize=1)
def author_class():
    _, sha = verified_source('amiml')
    if sha != COMMIT:
        raise ValueError('AMIML source is not the reviewed fixed version')
    return load_definitions('amiml', 'AMIML/model.py', ['AMIML'],
        {'torch': torch, 'nn': nn, 'F': F, 'math': math}, [
            ('super(HE2RNA, self).__init__()', 'super().__init__()'),
            ('in_channels=256, out_channels=128', 'in_channels=input_dim, out_channels=128')]).AMIML


def make_network(dimension=256, classes=2, R=100, alpha=.1):
    if (type(dimension) is not int or dimension < 1 or type(classes) is not int or classes < 2
            or type(R) is not int or R < 1 or isinstance(alpha, bool)
            or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0):
        raise ValueError('AMIML needs positive dimensions/R, C>=2 and finite nonnegative alpha')
    result = author_class()(input_dim=dimension, R=R, alpha=alpha, d=8)
    # Deliberately keep the released constructor's RNG consumption and all
    # unused BN modules. Only the incompatible V output width is reconstructed.
    result.value = nn.Linear(8, classes)
    return result


def attention_from_encoded(network, encoded):
    """Exact source ranking/decoder/residual/QK dropout/attention/V dropout order."""
    score = network.conv5(encoded)
    high, hi = score.topk(network.minmax, dim=2, largest=True, sorted=True)
    low, lo = score.topk(network.minmax, dim=2, largest=False, sorted=True)
    selected_ids = torch.cat([hi, lo], dim=2)
    selected = encoded.gather(2, selected_ids.expand(-1, 8, -1))
    scalars = torch.cat([high, low], dim=2)
    residual = (network.alpha*network.conv6(scalars)+selected).transpose(1, 2)
    q, k, v = network.query(residual), network.key(residual), network.value(residual)
    weights = (network.dropout(q@k.transpose(1, 2))/math.sqrt(8)).softmax(-1)
    logits = network.dropout(weights@v).mean(1)
    return {'logits': logits, 'probabilities': logits.softmax(-1),
            'selected_ids': selected_ids.squeeze(1), 'attention': weights,
            'residual': residual, 'score': score}


def forward_bags(network, bags):
    if (not isinstance(bags, (tuple, list)) or not bags
            or any(x.ndim != 2 or x.shape[1] != network.input_dim
                or len(x) < network.minmax or not x.is_floating_point()
                or not torch.isfinite(x).all() for x in bags)):
        raise ValueError('AMIML needs finite unpadded [N,D] bags, N>=R; no invented repeated/padded patches')
    lengths = [len(x) for x in bags]
    if len(set(lengths)) == 1:
        x = torch.stack(bags).transpose(1, 2)
        for conv, bn in [(network.conv1, network.bn1), (network.conv2, network.bn2),
                         (network.conv3, network.bn3)]:
            x = network.relu(bn(conv(x)))
        return attention_from_encoded(network, network.conv4(x))
    # BN over all real bag/patch cells, exactly the source axes. Zero padding
    # would contaminate means/variance. Conv1d(1x1) and rank selection preserve
    # each bag boundary. This extension is explicit because the loader is absent.
    x = torch.cat(bags).T[None]
    for conv, bn in [(network.conv1, network.bn1), (network.conv2, network.bn2),
                     (network.conv3, network.bn3)]:
        x = network.relu(bn(conv(x)))
    encoded = network.conv4(x)
    # Apply attention together after independent per-bag min/max selections;
    # processing bags separately would change dropout RNG and batch semantics.
    pieces = encoded.split(lengths, dim=2)
    selected, scalars, ids, scores = [], [], [], []
    for part in pieces:
        score = network.conv5(part)
        high, hi = score.topk(network.minmax, dim=2, largest=True, sorted=True)
        low, lo = score.topk(network.minmax, dim=2, largest=False, sorted=True)
        indices = torch.cat([hi, lo], dim=2)
        selected.append(part.gather(2, indices.expand(-1, 8, -1)))
        scalars.append(torch.cat([high, low], dim=2)); ids.append(indices); scores.append(score)
    residual = (network.alpha*network.conv6(torch.cat(scalars))+torch.cat(selected)).transpose(1, 2)
    q, k, v = network.query(residual), network.key(residual), network.value(residual)
    weights = (network.dropout(q@k.transpose(1, 2))/math.sqrt(8)).softmax(-1)
    logits = network.dropout(weights@v).mean(1)
    return {'logits': logits, 'probabilities': logits.softmax(-1),
        'selected_ids': torch.cat(ids).squeeze(1), 'attention': weights,
        'residual': residual, 'score': scores}
