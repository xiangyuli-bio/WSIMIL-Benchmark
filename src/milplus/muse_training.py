"""Pinned original MUSE20-view training and original five-field collation."""
import numpy as np
import torch

from .upstream import load_definitions


def author_training():
    utility=load_definitions('muse','utils/utils.py',['calculate_error'],{'torch':torch})
    return load_definitions('muse','utils/core_utils.py',['Accuracy_Logger','train_loop'],
        {'torch':torch,'np':np,'calculate_error':utility.calculate_error},
        [('device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")',
          'device=next(model.parameters()).device')])


def author_collate():
    return load_definitions('muse','utils/utils.py',['collate_tranformer'],{'torch':torch}).collate_tranformer
