"""Original CATE binary mechanism with explicit concepts and a complete weighted objective."""
import math

import torch
from einops import repeat
from torch import nn
from torch.nn import functional as F

from .stochastic import inference_rng
from .upstream import load_definitions


def load_cate(anchors, dimension, stable=True):
    info = load_definitions('cate', 'src/infonce.py', ['InfoNCE', 'info_nce', 'transpose', 'normalize'],
                            {'torch': torch, 'nn': nn, 'F': F})
    patches = [('self.feats = torch.load(concept_path)',
                'for i, value in enumerate(anchors):\n            self.register_buffer(f"concept_{i}", value.clone())'),
               ('self.feats = [feat.to("cuda") for feat in self.feats]', ''), ('.to("cuda")', '')]
    if stable:
        patches += [('x_c.std(dim=(0, 1))', 'x_c.std(dim=(0, 1)).clamp_min(1e-8)')]
    base = load_definitions('cate', 'src/cate.py', ['CATE'],
                    {'torch': torch, 'nn': nn, 'InfoNCE': info.InfoNCE, 'repeat': repeat, 'anchors': anchors}, patches).CATE

    class BufferedCATE(base):
        @property
        def feats(self):
            return [getattr(self, f'concept_{i}') for i in range(len(anchors))]

    return load_definitions('cate', 'src/catemil.py', ['DAttention', 'initialize_weights'],
        {'torch': torch, 'nn': nn, 'F': F, 'CATE': BufferedCATE, 'concept_dim': dimension},
        [('CATE(input_dim=512, ib_dim=512, interv_dim=512, concept_path="./prompt_feats/brca_concepts.pt")',
          'CATE(input_dim=concept_dim, ib_dim=concept_dim, interv_dim=concept_dim, concept_path=None)'),
         ('F.softmax(x)', 'F.softmax(x, dim=-1)')])


class ConceptCATE(nn.Module):
    def __init__(self, dim, classes, text, *, pim_weight, sim_weight, dropout=.25, eval_seed=0):
        super().__init__()
        if text is None or text.kind != 'concept_anchors' or classes != 2:
            raise ValueError('Native CATE needs two classes and class-specific/agnostic concept anchors')
        for value in (pim_weight, sim_weight):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('CATE requires explicit positive PIM and SIM loss weights')
        if dropout not in (0, .25):
            raise ValueError('Author CATE dropout is either disabled or .25')
        self.dim, self.pim_weight, self.sim_weight, self.eval_seed = dim, pim_weight, sim_weight, eval_seed
        self.network = load_cate(text.anchors, dim).DAttention(input_dim=2*dim, n_classes=classes, dropout=dropout)
        self.text_metadata = text.metadata

    def _forward(self, x, label):
        logits, _, _, losses = self.network(x, label)
        result = {'logits': logits, **losses}
        if label is not None:
            result['regularization_loss'] = self.pim_weight*losses['infonce_loss'] + self.sim_weight*losses['kl_loss']
        return result

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or x.shape[1] != self.dim or len(x) < 10 or not torch.isfinite(x).all():
            raise ValueError('CATE requires finite [N,D] features and at least 10 instances')
        if label is not None and (label.numel() != 1 or label.item() not in (0, 1)):
            raise ValueError('CATE source supports exactly binary class labels')
        if self.training:
            return self._forward(x, label)
        with inference_rng(x.device, self.eval_seed):
            # Labels are never supplied for inference; InfoNCE is a training-only target.
            return self._forward(x, None)

    def protocol(self):
        return {'source': 'HKU-MedAI/CATE; binary CATE-MIL/DAttention',
                'objective': 'CE + pim_weight*InfoNCE + sim_weight*KL',
                'pim_weight': self.pim_weight, 'sim_weight': self.sim_weight,
                'recipe_scope': 'source core plus PULS trainer; weights are explicit, not asserted paper defaults',
                'inputs': 'matching VLM image/text space; ordered class concepts and optional agnostic negatives',
                'top_k': 10, 'temperature': .1, 'cfi': 'raw dot, scale 56.3477, global sample std, detached before SNN',
                'compatibility': 'CPU/device buffers; correct concatenated width; std floor 1e-8',
                'evaluation': 'one reproducible stochastic draw; isolated RNG; labels excluded',
                'eval_seed': self.eval_seed, 'text': self.text_metadata}
