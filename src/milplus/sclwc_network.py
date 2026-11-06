"""Paper-based CDA/PNM/WSCL reference, distinct from the literal source control.

The pinned source's gated attention/PNM gate and initializer are used privately.
Paper Eq6 is interpreted as its stated two cross-entropy objectives. Queues hold
actual TRAIN bags, with live current-bag anchors; no fabricated Gaussian bags.
Bank capacities and five-epoch warmup are explicit source-informed choices.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from milplus.io import digest_object
from milplus.upstream import load_definitions, verified_source

COMMIT = 'd8d0ff44e887cadeeaff199cd771af9ee23e5855'


def original_layers():
    _, sha = verified_source('sclwc')
    if sha != COMMIT:
        raise ValueError('Wrong SCL-WC source')
    layers = load_definitions('sclwc', 'models/public_models/SCL.py',
                              ['Att_Head', 'Attn_Net_Gated'], {'torch': torch, 'nn': nn})
    initialize = load_definitions('sclwc', 'utils/utils.py', ['initialize_weights'], {'nn': nn}).initialize_weights
    return layers, initialize


class SCLPaperReference(nn.Module):
    def __init__(self, dimension, classes=2, hidden=512, attention_dim=256,
                 k=8, temperature=.07, negative_class=0, dropout=True,
                 positive_capacity=2048, negative_capacity=16384, hard_negative_capacity=2048,
                 warmup_epochs=5):
        super().__init__()
        if (any(type(v) is not int or v < 1 for v in [dimension, classes, hidden, attention_dim, k,
                 positive_capacity, negative_capacity, hard_negative_capacity]) or classes < 2 or
                type(negative_class) is not int or negative_class not in range(classes) or
                type(warmup_epochs) is not int or warmup_epochs < 0):
            raise ValueError('Invalid paper SCL dimensions, classes, bank sizes or warmup')
        if any(v % k for v in [positive_capacity, negative_capacity, hard_negative_capacity]):
            raise ValueError('Full representative bags must fit the declared k-divisible banks')
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 < temperature < 1:
            raise ValueError('SCL temperature must be positive')
        author, initialize = original_layers()
        stem = [nn.Linear(dimension, hidden), nn.ReLU()]
        if dropout:
            stem.append(nn.Dropout(.25))
        self.projector = nn.Sequential(*stem)
        self.attention = author.Attn_Net_Gated(hidden, attention_dim, dropout=dropout, n_classes=classes)
        self.bag_classifiers = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(classes)])
        self.instance_classifiers = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(classes)])
        self.pnm_gate = author.Att_Head(hidden, attention_dim)
        self.pnm_classifier = nn.Linear(hidden, classes)
        initialize(self)
        self.dimension, self.classes, self.hidden = dimension, classes, hidden
        self.k, self.temperature, self.negative_class, self.warmup_epochs = k, float(temperature), negative_class, warmup_epochs
        self.capacities = {'negative': negative_capacity//k, 'hard_negative': hard_negative_capacity//k,
                           **{'positive_'+str(c): positive_capacity//k for c in range(classes) if c != negative_class}}
        self.banks = {name: [] for name in self.capacities}
        self.contract = digest_object({'source': COMMIT, 'dimension': dimension, 'classes': classes,
            'hidden': hidden, 'attention_dim': attention_dim, 'k': k, 'temperature': temperature,
            'negative_class': negative_class, 'dropout': dropout, 'capacity_bags': self.capacities,
            'warmup_epochs': warmup_epochs, 'paper_eq6_interpretation': 'sum of two stated CE objectives',
            'contrast_normalization': 'Eq7 sum across live anchors; mean across positive keys'})

    def contrastive(self, anchors, label, slide_id):
        records = [item for bank in self.banks.values() for item in bank if item['slide_id'] != slide_id]
        positive = [item for item in records if item['label'] == label]
        negative = [item for item in records if item['label'] != label]
        if not positive or not negative:
            return anchors.sum()*0, {'active': False, 'positive_bags': len(positive), 'negative_bags': len(negative)}
        keys = torch.cat([item['features'] for item in records], 0).to(device=anchors.device, dtype=anchors.dtype)
        target = torch.tensor([item['label'] == label for item in records], device=anchors.device).repeat_interleave(self.k)
        # Memory keys are historical detached features. Current anchor features
        # remain live, so this term genuinely refines the task projector.
        logits = F.normalize(anchors, dim=1)@F.normalize(keys, dim=1).T/self.temperature
        log_probability = logits-logits.logsumexp(1, keepdim=True)
        loss = -log_probability[:, target].mean(1).sum()
        return loss, {'active': True, 'positive_bags': len(positive), 'negative_bags': len(negative),
                      'positive_keys': int(target.sum()), 'total_keys': len(keys)}

    def forward(self, x, label=None, slide_id=None, epoch=0):
        if x.ndim != 2 or x.shape[1] != self.dimension or len(x) < 2*self.k or not torch.isfinite(x).all():
            raise ValueError('Paper SCL needs complete full-dimensional bags with at least 2k actual patches')
        h = self.projector(x)
        attention_raw, _ = self.attention(h)
        attention = attention_raw.T.softmax(1)
        pooled = attention@h
        logits = torch.stack([head(pooled[i]) for i, head in enumerate(self.bag_classifiers)]).reshape(1, self.classes)
        result = {'logits': logits, 'features': h, 'attention': attention, 'attention_raw': attention_raw.T,
                  'pooled': pooled}
        if label is None:
            return result
        y = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        if y not in range(self.classes) or not isinstance(slide_id, str) or not slide_id:
            raise ValueError('SCL TRAIN bags need an actual class and stable slide identity')
        top = attention[y].topk(self.k).indices
        bottom = (-attention[y]).topk(self.k).indices
        instances = torch.cat([h[top], h[bottom]], 0)
        pseudo = torch.cat([torch.ones(self.k), torch.zeros(self.k)]).long().to(x.device)
        target = torch.tensor([y], device=x.device)
        mil = F.cross_entropy(logits, target)
        instance = F.cross_entropy(self.instance_classifiers[y](instances), pseudo)
        gate = self.pnm_gate(h)
        positive_feature, negative_feature = (gate*h).mean(0), ((1-gate)*h).mean(0)
        positive_logits = self.pnm_classifier(positive_feature)[None]
        negative_logits = self.pnm_classifier(negative_feature)[None]
        positive_loss = F.cross_entropy(positive_logits, target)
        negative_loss = F.cross_entropy(negative_logits, torch.tensor([self.negative_class], device=x.device))
        if epoch >= self.warmup_epochs:
            contrast, info = self.contrastive(h[top], y, slide_id)
        else:
            contrast, info = h.sum()*0, {'active': False, 'warmup': True}
        terms = {'mil': mil, 'instance': instance, 'pnm_positive': positive_loss,
                 'pnm_negative': negative_loss, 'wscl': contrast}
        result.update(loss=.8*mil+.2*instance+.2*(positive_loss+negative_loss)+.01*contrast,
            terms=terms, pnm_gate=gate, pnm_positive=positive_feature, pnm_negative=negative_feature,
            selected_top=top, selected_bottom=bottom, contrastive_info=info, label=y, slide_id=slide_id)
        return result

    @torch.no_grad()
    def enqueue(self, result):
        label, slide, h = result['label'], result['slide_id'], result['features'].detach()
        top, bottom = result['selected_top'], result['selected_bottom']
        entries = [('hard_negative', h[top]), ('negative', h[bottom])] if label == self.negative_class else [('positive_'+str(label), h[top])]
        for name, features in entries:
            record = {'slide_id': slide, 'label': label, 'features': features.detach().cpu().clone()}
            self.banks[name].append(record)
            if len(self.banks[name]) > self.capacities[name]:
                del self.banks[name][0]

    def get_extra_state(self):
        return {'contract': self.contract, 'banks': copy.deepcopy(self.banks),
                'scope': 'Actual historical TRAIN representative bags only; no synthetic random queue entries'}

    def set_extra_state(self, state):
        if state['contract'] != self.contract or set(state['banks']) != set(self.capacities):
            raise ValueError('SCL bank/feature/recipe contract differs')
        for name, entries in state['banks'].items():
            if len(entries) > self.capacities[name]:
                raise ValueError('SCL bank capacity exceeded')
            for entry in entries:
                x = entry['features']
                expected = self.negative_class if name in ['negative', 'hard_negative'] else int(name.split('_')[-1])
                if (entry['label'] != expected or not entry['slide_id'] or
                        not isinstance(x, torch.Tensor) or x.shape != (self.k, self.hidden) or
                        x.requires_grad or not torch.isfinite(x).all()):
                    raise ValueError('Invalid real SCL historical feature bag')
        self.banks = copy.deepcopy(state['banks'])
        for entries in self.banks.values():
            for entry in entries:
                entry['features'] = entry['features'].cpu()
