"""Literal2018DeepGraphSurv AGCN equations, independently checked against TF1.

Full native graph/attention/Cox/BN and TF1Adam, with explicit frozen-FM input width.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TFPopulationBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))
        self.register_buffer('moving_mean', torch.zeros(channels))
        self.register_buffer('moving_var', torch.zeros(channels))

    def forward(self, x, training, decay):
        if training:
            mean = x.mean((0, 1))
            var = (x-mean).square().mean((0, 1))
            with torch.no_grad():
                coefficient = x.new_tensor(1)-x.new_tensor(decay)
                self.moving_mean.sub_((self.moving_mean-mean.detach())*coefficient)
                self.moving_var.sub_((self.moving_var-var.detach())*coefficient)
        else:
            mean, var = self.moving_mean, self.moving_var
        # TF1 population variance and EMA initialization, not torch.BatchNorm's
        # unbiased running-variance update or Keras's initial moving variance1.
        return (x-mean)*torch.rsqrt(var+x.new_tensor(1e-3))*self.gamma+self.beta


class SGCLL(nn.Module):
    def __init__(self, dimension, channels, relu=True, dropout=True, batch_norm=True, K=3, epsilon=1e-5):
        super().__init__()
        if K < 2:
            raise ValueError('Original SGC_LL unconditionally builds T1; K must be at least2')
        self.weight = nn.Parameter(torch.empty(dimension*K, channels))
        self.metric = nn.Parameter(torch.empty(dimension, dimension))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.alpha = nn.Parameter(torch.full((1,), .01))
        nn.init.xavier_uniform_(self.weight);nn.init.xavier_uniform_(self.metric)
        self.bn = TFPopulationBN(channels) if batch_norm else None
        self.K, self.epsilon, self.relu, self.dropout = K, epsilon, relu, dropout

    def forward(self, x, intrinsic, sizes, training, bn_decay, dropout_mask=None):
        maximum = x.shape[1]
        rows, residuals = [], []
        for b, n in enumerate(sizes.tolist()):
            node = x[b, :n]
            transformed = node@self.metric
            gram = transformed@transformed.T
            diagonal = torch.diag(gram)
            # This is the literal published plus-Gram formula. It is not
            # replaced by squared Euclidean distance or a symmetric repair.
            raw = gram+diagonal[None, :]+diagonal[:, None]
            # TF l2_normalize applies epsilon to the squared norm, before sqrt.
            normalized = raw*torch.rsqrt(raw.square().sum(0, keepdim=True).clamp_min(self.epsilon))
            gaussian = self.alpha*torch.exp(-normalized)
            residual = torch.diag(gaussian.sum(0))-gaussian
            laplacian = intrinsic[b, :n, :n]+residual
            basis = [node, laplacian@node]
            for _ in range(2, self.K):
                basis.append(2*(laplacian@basis[-1])-basis[-2])
            # Exact TensorFlow [K,N,F] -> [N,F,K] flattening order.
            features = torch.stack(basis, -1).reshape(n, -1)
            convolved = features@self.weight+self.bias
            if self.relu:convolved = F.relu(convolved)
            rows.append(F.pad(convolved, (0, 0, 0, maximum-n)))
            residuals.append(F.pad(residual, (0, maximum-n, 0, maximum-n)))
        result = torch.stack(rows)
        if self.bn is not None:result = self.bn(result, training, bn_decay)
        if self.dropout:
            if dropout_mask is None:
                # The author applies dropout even at inference. Retain it;
                # independent comparisons inject actual original TF masks.
                result = F.dropout(result, .2, training=True)
            else:
                if dropout_mask.ndim == 4:dropout_mask = dropout_mask.squeeze(2)
                if dropout_mask.shape != result.shape:
                    raise ValueError('Original dropout mask shape differs')
                result = result*result.new_tensor(1.25)*dropout_mask.to(result.dtype)
        return result, torch.stack(residuals)


class DenseTF(nn.Module):
    def __init__(self, dimension, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dimension, channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.tanh(x@self.weight+self.bias)


class NativeAGCN(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.attention1 = SGCLL(dimension, 32, relu=False, dropout=False, batch_norm=False)
        self.attention2 = SGCLL(32, 1, relu=False, dropout=False, batch_norm=False)
        self.backbone1 = SGCLL(dimension, 256)
        self.backbone2 = SGCLL(256, 256)
        self.backbone3 = SGCLL(256, 128)
        self.fc1, self.fc2 = DenseTF(128, 64), DenseTF(64, 1)

    def forward(self, x, intrinsic, sizes, bn_decay=.5, dropout_masks=None):
        attention, _ = self.attention1(x, intrinsic, sizes, self.training, bn_decay)
        attention, _ = self.attention2(attention, intrinsic, sizes, self.training, bn_decay)
        attention = torch.sigmoid(attention/1e3).softmax(1).transpose(1, 2)
        outputs, residuals = x, []
        for i, layer in enumerate([self.backbone1, self.backbone2, self.backbone3]):
            # Every layer uses the original intrinsic Laplacian, not the
            # previous layer's learned Laplacian.
            outputs, residual = layer(outputs, intrinsic, sizes, self.training, bn_decay,
                                       None if dropout_masks is None else dropout_masks[i])
            residuals.append(residual)
        pooled = torch.stack([(attention[b, :, :n]@outputs[b, :n]).flatten()
                              for b, n in enumerate(sizes.tolist())])
        return {'output': self.fc2(self.fc1(pooled)), 'graph_feature': pooled,
                'attention': attention, 'res_graph': residuals[-1], 'residuals': residuals}

    def source_parameter_map(self):
        mapping = {}
        for ours, source in [('attention1', 'attention_net/sgc_ll_1'), ('attention2', 'attention_net/sgc_ll_2'),
                             ('backbone1', 'backbone_net/sgc_ll_1'), ('backbone2', 'backbone_net/sgc_ll_2'),
                             ('backbone3', 'backbone_net/sgc_ll_3')]:
            for key, native in [('weight', 'weights'), ('metric', 'metric'), ('bias', 'biases'), ('alpha', 'alpha')]:
                mapping[ours+'.'+key] = source+'/'+native
            if ours.startswith('backbone'):
                bn = source+'/'+source.rsplit('/', 1)[-1]
                mapping[ours+'.bn.beta'] = bn+'/beta'
                mapping[ours+'.bn.gamma'] = bn+'/gamma'
        for layer in ['fc1', 'fc2']:
            mapping[layer+'.weight'] = 'backbone_net/'+layer+'/weights'
            mapping[layer+'.bias'] = 'backbone_net/'+layer+'/biases'
        return mapping


def source_cox(risks, events):
    risks = risks.reshape(-1)
    return -((risks-torch.exp(risks).cumsum(0).log())*events).sum()


class TF1Adam(torch.optim.Optimizer):
    """TF1 dense Adam epsilon position, global float32 beta powers and updates."""
    def __init__(self, params, lr=.0002):
        super().__init__(params, {'lr': lr, 'beta1': .9, 'beta2': .999, 'eps': 1e-8,
            'beta1_power': torch.tensor(.9, dtype=torch.float32),
            'beta2_power': torch.tensor(.999, dtype=torch.float32), 'updates': 0})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():loss = closure()
        for group in self.param_groups:
            one = group['beta1_power'].new_tensor(1.)
            alpha = (group['beta1_power'].new_tensor(group['lr'])*
                     (one-group['beta2_power']).sqrt()/(one-group['beta1_power']))
            for p in group['params']:
                if p.grad is None:continue
                if p.grad.is_sparse:raise ValueError('Original model has dense gradients')
                state = self.state[p]
                if not state:state.update(m=torch.zeros_like(p), v=torch.zeros_like(p))
                b1, b2 = p.new_tensor(group['beta1']), p.new_tensor(group['beta2'])
                state['m'].add_((p.grad-state['m'])*(1-b1))
                state['v'].add_((p.grad.square()-state['v'])*(1-b2))
                p.sub_(alpha.to(p.device)*state['m']/(state['v'].sqrt()+p.new_tensor(group['eps'])))
            group['beta1_power'].mul_(group['beta1'])
            group['beta2_power'].mul_(group['beta2'])
            group['updates'] += 1
        return loss
