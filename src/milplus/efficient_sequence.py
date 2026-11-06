"""Pinned EfficientMIL recurrent variants with APS and dual-stream BCE training."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .stochastic import inference_rng
from .upstream import load_definitions

EFFICIENT_METHODS = {'efficientmil_gru', 'efficientmil_lstm'}


def load_efficient(kind):
    if kind not in {'gru', 'lstm'}:
        raise ValueError('Use the explicit GRU or LSTM author variant; Mamba requires its verified custom backend')
    common = load_definitions('efficientmil', 'efficientmil_common.py',
        ['FCLayer', 'IntelligentPatchSelector', 'BaseBClassifier', 'MILNet'],
        {'torch': torch, 'nn': nn, 'F': F, 'Tensor': torch.Tensor, 'Tuple': tuple})
    module = load_definitions('efficientmil', f'efficientmil_{kind}.py',
        [kind.upper()+'Block', 'BClassifier'],
        {'torch': torch, 'nn': nn, 'F': F, 'Tensor': torch.Tensor, 'copy': copy,
         'BaseBClassifier': common.BaseBClassifier})
    initialize = load_definitions('efficientmil', 'common_utils.py', ['apply_sparse_init'],
                                  {'nn': nn}).apply_sparse_init
    return common, module, initialize


class EfficientSequenceMIL(nn.Module):
    def __init__(self, dim, classes, profile='efficientmil_gru', hidden=768, layers=2,
                 bidirectional=True, dropout=.1, selected_patches=64, selection='aps',
                 l2_weight=1e-4, min_lr=5e-6, inference_seed=7919, max_similarity_bytes=536870912):
        super().__init__()
        if profile not in EFFICIENT_METHODS or min(dim, classes, hidden, layers, selected_patches) < 1:
            raise ValueError('Invalid EfficientMIL profile or positive dimensions')
        if classes < 2 or not 0 <= dropout < 1 or selection not in {'aps', 'top-k', 'random-k'}:
            raise ValueError('EfficientMIL requires exclusive classes >= 2 and an explicit selection strategy')
        if l2_weight < 0 or min_lr < 0 or max_similarity_bytes < 1:
            raise ValueError('Invalid EfficientMIL regularization, learning rate or memory budget')
        kind = profile.rsplit('_', 1)[1]
        common, module, initialize = load_efficient(kind)
        bag = module.BClassifier(input_size=dim, output_class=classes, dropout=dropout,
            bidirectional=bidirectional, big_lambda=selected_patches, selection_strategy=selection,
            use_intelligent_selector=True, **{kind+'_hidden_size': hidden, kind+'_num_layers': layers})
        self.network = common.MILNet(common.FCLayer(dim, classes), bag)
        # The author's train.py uses this initializer, not its alternative factory initializer.
        self.network.apply(initialize)
        self.profile, self.selection, self.l2_weight = profile, selection, l2_weight
        self.min_lr, self.inference_seed = min_lr, inference_seed
        self.max_similarity_bytes, self.total_epochs = max_similarity_bytes, 50
        self.scheduler, self.scheduler_state = None, None

    def configure_training(self, config):
        self.total_epochs = config['epochs']
        if self.min_lr > config['lr']:
            raise ValueError('EfficientMIL min_lr must not exceed base learning rate')

    def make_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(.5, .9), weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.total_epochs, self.min_lr)
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
        return optimizer

    def after_validation(self, metrics):
        self.scheduler.step()

    def get_extra_state(self):
        return {'scheduler': self.scheduler.state_dict() if self.scheduler is not None else self.scheduler_state}

    def set_extra_state(self, state):
        self.scheduler_state = state['scheduler']
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

    def forward(self, x, coords=None, label=None):
        if x.ndim != 2 or len(x) == 0:
            raise ValueError('EfficientMIL needs a nonempty [N,D] feature bag')
        if self.selection == 'aps':
            if len(x) < 2:
                raise ValueError('Author APS diversity requires at least two patches')
            if len(x)**2*x.element_size() > self.max_similarity_bytes:
                raise ValueError('Author APS similarity matrix exceeds max_similarity_bytes')
        if not self.training and self.selection == 'random-k':
            with inference_rng(x.device, self.inference_seed):
                instance, logits, attention, embedding = self.network(x)
        else:
            instance, logits, attention, embedding = self.network(x)
        result = {'logits': logits, 'attention': attention.T, 'embedding': embedding,
                  'max_logits': instance.max(0, keepdim=True).values}
        if label is not None:
            target = F.one_hot(label, logits.shape[-1]).to(logits)
            bag_loss = F.binary_cross_entropy_with_logits(logits, target)
            instance_loss = F.binary_cross_entropy_with_logits(result['max_logits'], target)
            l2 = sum(p.square().sum() for p in self.network.parameters() if p.requires_grad)
            result.update(total_loss=.5*(bag_loss+instance_loss)+self.l2_weight*l2,
                          instance_loss=instance_loss, regularization_loss=self.l2_weight*l2)
        return result

    def protocol(self):
        return {'profile': self.profile, 'selector': self.selection,
                'selected_patches': self.network.b_classifier.big_lambda,
                'objective': 'half bag BCE + half max-instance BCE + explicit L2',
                'l2_weight': self.l2_weight, 'optimizer': 'Adam betas=(0.5,0.9) plus configured weight decay',
                'scheduler': 'epoch cosine', 'min_lr': self.min_lr,
                'initialization': 'author train.py orthogonal Linear weights and default recurrent initialization',
                'prediction': 'bag logits softmax for exclusive classes, as author multiclass evaluation',
                'max_similarity_bytes': self.max_similarity_bytes,
                'memory_scope': 'author APS N*N similarity matrix; not a linear-memory implementation or total RAM cap',
                'random_k_inference': 'isolated fixed RNG',
                'mamba_variant': 'not substituted; custom BiMamba backend remains pending validation'}
