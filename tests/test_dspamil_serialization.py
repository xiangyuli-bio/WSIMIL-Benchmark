"""Serialization utility fidelity, not a substitute for full-model checkpoint IO."""
from io import BytesIO

import numpy as np
import torch

from milplus.dspamil_runtime import checkpoint_view


def encode(value):
    stream = BytesIO()
    torch.save(checkpoint_view(value), stream)
    return stream.getvalue()


def test_restored_history_string_aliases_do_not_change_bytes_or_tensors_and_metadata():
    model = torch.nn.Linear(3, 2)
    state = model.state_dict()
    first = 'synthetic_long_slide_identifier_for_serialization_fidelity'
    other = first.encode().decode()
    assert first == other and first is not other
    original = {'model': state, 'history': [[first], [first]],
        'numpy_state': np.arange(7, dtype=np.uint32), 'tuple': (.1, .9), 'none': None}
    resumed = {**original, 'history': [[first], [other]]}
    view = checkpoint_view(original)
    assert view['model']['weight'] is state['weight']
    assert view['numpy_state'] is original['numpy_state']
    assert view['model']._metadata == state._metadata
    assert type(view['model']) is type(state) and type(view['tuple']) is tuple
    encoded = encode(original)
    assert encoded == encode(resumed)
    loaded = torch.load(BytesIO(encoded), weights_only=False)
    assert loaded['model']._metadata == state._metadata
    for key, value in state.items():
        assert loaded['model'][key].dtype == value.dtype and torch.equal(loaded['model'][key], value)
    assert np.array_equal(loaded['numpy_state'], original['numpy_state'])
    assert loaded['history'] == original['history'] and loaded['tuple'] == original['tuple']
    assert encode(loaded) == encoded
