"""Complete native PAMIL policy/SFFR/TCM and explicit paper-reference assembly.

The public repository has no train loop. Author kernels remain private, and
source sampling/routing is a separate control for the reconstructed recipe.
"""
import math
import random
from types import SimpleNamespace

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions, verified_source

COMMIT = 'd62960af2c115f87d570a0fbf8852049745b4e94'
PAPER = 'pamil_policy_paper_reference'
SOURCE = 'pamil_policy_released_sampling_reference_training'


def native_classes(fused_history):
    _, sha = verified_source('pamil_policy')
    if sha != COMMIT:
        raise ValueError('PAMIL dynamic-policy source differs from reviewed commit')
    scope = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'math': math, 'random': random, 'rearrange': rearrange}
    helper = load_definitions('pamil_policy', 'utilmodule/utils.py',
        ['cat_msg2cluster_group', 'cosine_scheduler'], scope)
    base = load_definitions('pamil_policy', 'models/basemodel.py',
        ['Attention', 'AttenLayer', 'GroupsAttenLayer', 'basedBasicLayer', 'BasedMILTransformer'],
        {**scope, 'cat_msg2cluster_group': helper.cat_msg2cluster_group})
    fusion = load_definitions('pamil_policy', 'models/SFFR.py',
        ['Cross_Attention', 'FusionHistoryFeatures'], scope,
        replacements=[('torch.stack(memory.msg_states[:], dim=1)', 'torch.stack(memory.merge_msg_states[:], dim=1)')]
        if fused_history else [])
    head = load_definitions('pamil_policy', 'models/classmodel.py',
        ['Attention', 'AttenLayer', 'GroupsMSGAttenLayer', 'HighBasicLayer', 'ClassMultiMILTransformer'], scope)
    # CPU/GPU device ports, registered Gaussian scale and scalar-action/single-
    # step shape fixes. No replacement actor, GRU, distribution or objective.
    patches = [
        ('self.action_var = torch.full((action_size,), action_std).cuda()',
         "self.register_buffer('action_var', torch.full((action_size,), action_std))"),
        ('torch.zeros(1, state_ini.size(0), self.hidden_state_dim).cuda()',
         'state_ini.new_zeros((1, state_ini.size(0), self.hidden_state_dim))'),
        ('torch.zeros(1, batch_size, state.size(2)).cuda()', 'state.new_zeros((1, batch_size, state.size(2)))'),
        ('torch.squeeze(action.view(seq_l * batch_size, -1))', 'action.view(seq_l * batch_size, -1)'),
        ('rewards.std() + 1e-5', '(rewards.std() if rewards.numel()>1 else rewards.new_zeros(())) + 1e-5'),
        ('.cuda()', ''),
    ]
    policy = load_definitions('pamil_policy', 'models/DPSF.py', ['Memory', 'Cat_Attention', 'ActorCritic', 'PPO'],
        scope, replacements=patches)
    environment = load_definitions('pamil_policy', 'utilmodule/environment.py', ['expand_data'], scope,
        replacements=[('torch.cat(random_coords_segments, dim=1)',
                       'torch.cat(random_coords_segments, dim=1) if random_coords_segments else update_coords[:,:0]'),
                      ('torch.cat(random_data_segments, dim=1)',
                       'torch.cat(random_data_segments, dim=1) if random_data_segments else update_datas[:,:0]')])
    grouping = load_definitions('pamil_policy', 'utilmodule/core.py', ['interpolate_probs', 'grouping'], scope,
        replacements=[('idx = torch.tensor(idx)', 'idx = torch.as_tensor(idx, device=update_features.device)'),
            ('mask = torch.ones(update_features.size(1), dtype=torch.bool)',
             'mask = torch.ones(update_features.size(1), dtype=torch.bool, device=update_features.device)')])
    return SimpleNamespace(base=base, fusion=fusion, head=head, policy=policy,
                           environment=environment, grouping=grouping, helper=helper)


def cosine_distance_order(remaining, state):
    # Metric is an explicit paper-reference choice; the supplement does not
    # publish a norm. No query to held-out labels enters sampling.
    return torch.argsort(F.cosine_similarity(remaining, state.reshape(1, -1), dim=-1), descending=True, stable=True)


def paper_selection(ids, features, previous_ids, action, state, size, scheme, threshold):
    count = min(size, len(ids))
    if scheme == 'liis':
        positions = previous_ids.detach().cpu().numpy()
        scores = action.detach().cpu().numpy().reshape(-1)[:len(positions)]
        order = np.argsort(positions, kind='stable')
        values = np.interp(ids.detach().cpu().numpy(), positions[order], scores[order])
        chosen = np.argsort(-values, kind='stable')[:count]
        return torch.as_tensor(chosen, dtype=torch.long, device=ids.device)
    if scheme not in {'gmss', 'ghss'}:
        raise ValueError('Unknown PAMIL paper sampling scheme')
    if float(action.reshape(-1)[0]) <= threshold:
        return torch.as_tensor(np.random.choice(len(ids), count, replace=False), device=ids.device)
    order = cosine_distance_order(features[ids].detach(), state.detach())
    if scheme == 'gmss':
        return order[:count]
    # Explicit far-rank2M convention; when the pool is too small, take the
    # farthest still-unselected entries. Odd/final groups preserve unique IDs.
    near = (count+1)//2
    chosen = order[:near]
    far = order[2*size:2*size+count-near]
    if len(far) < count-near:
        used = set(chosen.tolist()) | set(far.tolist())
        extra = [int(x) for x in order.flip(0) if int(x) not in used]
        far = torch.cat([far, order.new_tensor(extra[:count-near-len(far)])])
    return torch.cat([chosen, far])


class PAMILPolicyReference(nn.Module):
    def __init__(self, dimension, classes, profile=PAPER, bag_size=512, scheme='liis',
                 input_projection='source_dimensions', policy_lr=1e-5, policy_gamma=.1,
                 policy_epochs=3, action_std=.5, eps_clip=.2, layers=2):
        super().__init__()
        if (profile not in {PAPER, SOURCE} or scheme not in {'liis', 'gmss', 'ghss'}
                or any(type(x) is not int or x < 1 for x in [dimension, classes, bag_size, policy_epochs, layers])
                or classes < 2 or input_projection not in {'source_dimensions', 'learned_full_dim_extension'}
                or (input_projection == 'source_dimensions' and dimension not in {512, 1024})
                or profile == SOURCE and scheme != 'liis'):
            raise ValueError('PAMIL needs explicit profile/scheme and full native512/1024 or declared full-D projection')
        for value in [policy_lr, policy_gamma, action_std, eps_clip]:
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError('Invalid finite PAMIL policy settings')
        if policy_lr <= 0 or not 0 <= policy_gamma <= 1 or action_std <= 0 or not 0 < eps_clip < 1:
            raise ValueError('Invalid PAMIL policy learning rate/discount/Gaussian scale/clip')
        self.dimension, self.classes, self.profile = dimension, classes, profile
        self.bag_size, self.scheme = bag_size, scheme
        self.project_input = input_projection == 'learned_full_dim_extension' or dimension == 1024
        self.definitions = module = native_classes(profile == PAPER)
        args = SimpleNamespace(in_chans=dimension, embed_dim=512, n_classes=classes,
            num_msg=1, ape=True, ape_class=False, num_layers=layers)
        self.base = module.base.BasedMILTransformer(args)
        self.fusion = module.fusion.FusionHistoryFeatures(512, 512)
        self.head = module.head.ClassMultiMILTransformer(args)
        self.ppo = module.policy.PPO(512, 512, 512, False, action_std=action_std, lr=policy_lr,
            gamma=policy_gamma, K_epochs=policy_epochs, eps_clip=eps_clip,
            action_size=bag_size if scheme == 'liis' else 1)
        self.policy, self.policy_old = self.ppo.policy, self.ppo.policy_old

    def initial_memory(self, device):
        memory = self.definitions.policy.Memory()
        if self.profile == PAPER:
            width = self.bag_size if self.scheme == 'liis' else 1
            memory.actions.append(torch.zeros(1, width, device=device))
            memory.logprobs.append(torch.zeros(1, device=device))
        return memory

    def source_rollout(self, features, memory):
        # Numeric IDs are only a row-identity ledger; never fake physical coords.
        ids = torch.arange(len(features), device=features.device).reshape(1, -1, 1)
        ledger, pool, steps = self.definitions.environment.expand_data(ids, features[None],
            action_size=self.bag_size, total_steps=1)
        grouping = self.definitions.grouping.grouping(self.bag_size)
        predictors, selected = [], []
        for step in range(steps):
            scores, memory = grouping.rlselectindex_grouping(self.ppo, memory, ledger, restart=step == 0)
            # Preserve actual source interpolation/weighted replacement. A
            # clipped all-zero score vector is the declared uniform limit.
            if not bool(scores.sum() > 0):
                scores = torch.ones_like(scores)
            group, ledger, pool, memory = grouping.action_make_subbags(self.ppo, memory,
                scores, ledger, pool, action_size=self.bag_size, restart=step == 0, delete_begin=True)
            selected.append(memory.coords_actions[-1].reshape(-1).detach().cpu().tolist())
            _, state, memory, predictor = self.base(self.fusion, group, memory)
            predictors.append(predictor)
            with torch.no_grad():
                self.ppo.select_action(state, memory, restart_batch=step == 0, training=self.training)
        return memory, predictors, selected

    def paper_rollout(self, features, memory, threshold):
        remaining = torch.arange(len(features), device=features.device)
        previous, predictors, selected = None, [], []
        step = 0
        while len(remaining):
            if step == 0:
                chosen = torch.as_tensor(np.random.choice(len(remaining), min(self.bag_size, len(remaining)),
                    replace=False), dtype=torch.long, device=features.device)
            else:
                chosen = paper_selection(remaining, features, previous, memory.actions[-1],
                    memory.merge_msg_states[-1], self.bag_size, self.scheme, threshold)
            ids = remaining[chosen]
            selected.append(ids.detach().cpu().tolist())
            mask = torch.ones(len(remaining), dtype=torch.bool, device=features.device)
            mask[chosen] = False
            remaining = remaining[mask]
            _, state, memory, predictor = self.base(self.fusion, features[ids][None], memory)
            predictors.append(predictor)
            with torch.no_grad():
                self.ppo.select_action(state, memory, restart_batch=step == 0, training=self.training)
            previous, step = ids, step+1
        return memory, predictors, selected

    def forward(self, features, label=None, sia_weight=.5, stl_weight=0., reward=1., threshold=0.,
                update_policy=False):
        if (features.ndim != 2 or not len(features) or features.shape[1] != self.dimension
                or not features.is_floating_point() or not torch.isfinite(features).all()):
            raise ValueError('PAMIL requires the complete finite ordered [N,D] input bag')
        if update_policy and (label is None or not self.training):
            raise ValueError('PAMIL policy optimization is restricted to actual TRAIN bag labels')
        if label is not None:
            label = torch.as_tensor(label, device=features.device).reshape(-1)
            if label.shape != (1,) or label.dtype != torch.int64 or not 0 <= int(label[0]) < self.classes:
                raise ValueError('PAMIL needs one actual slide class ID')
        embedded = self.base.fc1(features) if self.project_input else features
        embedded = embedded+self.base.absolute_pos_embed[0]
        memory = self.initial_memory(features.device)
        if self.profile == SOURCE:
            memory, predictors, selected = self.source_rollout(embedded, memory)
        else:
            memory, predictors, selected = self.paper_rollout(embedded, memory, threshold)
        head_memory = (memory if self.profile == SOURCE else
            SimpleNamespace(msg_states=memory.merge_msg_states, cls_states=memory.cls_states))
        result, _ = self.head(head_memory)
        result = {'logits': result['logits'], 'probabilities': result['Y_prob'],
            'prediction': result['Y_hat'], 'selected_row_ids': selected, 'steps': len(selected), 'loss': None}
        if label is not None:
            wsl = F.cross_entropy(result['logits'], label)
            stl = torch.stack([F.cross_entropy(x['logits'], label) for x in memory.results_dict]).mean()
            sia = result['logits'].sum()*0
            for step in range(1, len(predictors)):
                current, previous = [memory.merge_msg_states[k].reshape(1, 512) for k in [step, step-1]]
                sia = sia-.5*(F.cosine_similarity(predictors[step], previous.detach()).mean()+
                    F.cosine_similarity(predictors[step-1], current.detach()).mean())/len(predictors)
            result.update(loss=wsl+stl_weight*stl+sia_weight*sia, wsl=wsl, stl=stl, sia=sia)
            tokens = torch.cat([x.reshape(1, 512) for x in memory.merge_msg_states])
            cls = memory.cls_states[-1].reshape(1, 512)
            rewards = float(int(result['prediction'][0]) == int(label[0]))*reward-F.cosine_similarity(cls, tokens)
            result['rewards'] = rewards.detach()
            if update_policy:
                for value in rewards.detach():
                    memory.rewards.append(value.reshape(1))
                self.ppo.update(memory)
        if result['loss'] is not None and not torch.isfinite(result['loss']):
            raise ValueError('PAMIL nonfinite complete WSL/STL/SIA objective')
        return result
