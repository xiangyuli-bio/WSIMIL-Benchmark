"""PTCMIL classification: author prompt clustering, token merger and within-epoch context."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions, load_single


def load_ptcmil():
    from nystrom_attention import NystromAttention
    scope = {'torch': torch, 'nn': nn, 'F': F}
    merger = load_definitions('ptcmil', 'classification/models/tcformer/tcformer_utils.py', ['merge_tokens'], scope)
    scope['merge_tokens'] = merger.merge_tokens
    blocks = load_definitions('ptcmil', 'classification/models/tcformer/tcformer_layers.py', ['CTM_w_clustering'], scope)
    scope.update(CTM_w_clustering=blocks.CTM_w_clustering, NystromAttention=NystromAttention,
                 NystromAttention_custom=load_single('ptcmil', 'classification/models/nystrom_attention_custom.py').NystromAttention_custom)
    return load_definitions('ptcmil', 'classification/models/model_vit_pmt_clu.py',
        ['TransLayer', 'TransLayer_custom', 'ViT_PMT_CLU'], scope, replacements=(
            ('self.identity = torch.eye(self.pmt_token.size(1), self.pmt_token.size(1)).cuda()',
             'self.register_buffer("identity", torch.eye(self.pmt_token.size(1), self.pmt_token.size(1)))'),
            ('pmt_tokens = self.pmt_token.cuda()', 'pmt_tokens = self.pmt_token'),
            ('cls_tokens = self.cls_token.cuda()', 'cls_tokens = self.cls_token'),
            ('torch.tensor(idx_cluster_new).cuda()', 'torch.tensor(idx_cluster_new, device=h.device)')))


class PTCMIL(nn.Module):
    def __init__(self, dim, classes, hidden=384, clusters=5, layers=2, ptc_weight=.1, momentum=.1,
                 moving_prompts=True):
        super().__init__()
        if hidden < 2 or clusters < 2 or clusters > hidden or layers < 2:
            raise ValueError('PTCMIL requires hidden >= clusters >= 2 and at least two layers')
        if not 0 < momentum <= 1 or ptc_weight <= 0:
            raise ValueError('PTCMIL requires 0 < momentum <= 1 and active PTC loss')
        args = SimpleNamespace(input_dim=dim, emb_dim=hidden, cluster_number=clusters,
                               n_layers=layers, moving_pmt=moving_prompts, momentum=momentum, merge_token=True)
        self.network = load_ptcmil().ViT_PMT_CLU(classes, args=args)
        self.ptc_weight, self.moving_prompts = ptc_weight, moving_prompts
        self.register_buffer('prompt_context', torch.zeros(1, clusters, hidden))
        self.register_buffer('context_ready', torch.tensor(False))
        self.pending_context = None

    def start_epoch(self, epoch):
        # Author train_loop initializes result_dict=None at every epoch.
        self.context_ready.fill_(False)
        self.prompt_context.zero_()
        self.pending_context = None

    def forward(self, x, coords=None, label=None):
        context = None
        if self.training and self.moving_prompts and self.context_ready:
            context = {'pmt_tokens': self.prompt_context}
        logits, probability, _, _, extra = self.network(x, result_dict=context)
        result = {'logits': logits, 'probabilities': probability}
        if self.training:
            result['regularization_loss'] = self.ptc_weight * extra['PTC_loss']
            self.pending_context = extra['pmt_tokens'].detach().clone()
        return result

    @torch.no_grad()
    def after_step(self):
        if self.moving_prompts and self.pending_context is not None:
            self.prompt_context.copy_(self.pending_context)
            self.context_ready.fill_(True)
        self.pending_context = None
