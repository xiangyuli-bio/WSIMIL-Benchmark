"""Full actual native graph with independent VAL20 selection/stopping control.

Synthetic architecture checks do not replace the unavailable pretrained asset.
"""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from milplus.feather_source import architecture_classes, original_configuration
from milplus.feather_training import fit, new_progress, reference_recipe, setup


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('reversed_validation', [False, True])
def test_full_uni_source_graph_val20_matches_literal_epoch_selection_and_early_stop(reversed_validation):
    torch.set_num_threads(2)
    torch.manual_seed(703)
    model = architecture_classes().ABMIL(**original_configuration('uni', 2))
    generator = torch.Generator().manual_seed(19)
    rows = [{'slide_id': str(i), 'target': i % 2} for i in range(6)]
    val = [{'slide_id': 'v' + str(i), 'target': (i % 2) ^ int(reversed_validation)} for i in range(4)]
    features = {}
    for i, row in enumerate(rows + val):
        x = torch.randn(257 + i, 1024, generator=generator) * .1
        # Different complete synthetic bags share a simple class signal.
        signal = i % 2
        x[:, :64] += 2. if signal else -2.
        features[row['slide_id']] = x
    load = lambda row: features[row['slide_id']]
    initial = deepcopy(model.state_dict())
    initial_rng = torch.get_rng_state()
    recipe = reference_recipe('finetune_val20')
    opt, sched, scaler, sample_generator = setup(model, device='cpu', seed=21, recipe=recipe)
    state = new_progress()
    best = []
    def checkpoint(progress, improved):
        if improved:
            best.append((progress['epochs'], deepcopy(model.state_dict())))
    assert fit(model, opt, sched, scaler, sample_generator, rows, val, load, state,
        device='cpu', classes=2, recipe=recipe, on_checkpoint=checkpoint) == 'completed'
    final = deepcopy({'model': model.state_dict(), 'optimizer': opt.state_dict(),
        'scheduler': sched.state_dict(), 'rng': torch.get_rng_state(), 'sampler': sample_generator.get_state()})
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(initial_rng)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5,
        betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=0.)
    sample_generator = torch.Generator().manual_seed(21)
    weights = torch.tensor([1 / 3] * 6, dtype=torch.float64)
    incumbent, bad, expected_best = None, 0, []
    for epoch in range(1, 21):
        draw = torch.multinomial(weights, 6, replacement=True, generator=sample_generator).tolist()
        losses = []
        for i in draw:
            row = rows[i]
            model.train()
            x = F.dropout(features[row['slide_id']], p=.1, training=True)
            opt.zero_grad(set_to_none=True)
            output, unused = model(x.unsqueeze(0))
            loss = F.cross_entropy(output['logits'], torch.tensor([row['target']]))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            total = 0.
            for row in val:
                output, unused = model(features[row['slide_id']].unsqueeze(0))
                total += float(F.cross_entropy(output['logits'].float(), torch.tensor([row['target']])))
        mean = total / len(val)
        improved = incumbent is None or mean < incumbent
        if improved:
            incumbent, bad = mean, 0
            expected_best.append((epoch, deepcopy(model.state_dict())))
        else:
            bad += 1
        old_lr = opt.param_groups[0]['lr']
        sched.step()
        assert state['history'][epoch - 1] == {'epoch': epoch, 'lr_used': old_lr,
            'next_lr': opt.param_groups[0]['lr'], 'train_loss': sum(losses) / 6,
            'train_draws': [rows[i]['slide_id'] for i in draw], 'validation_loss': mean,
            'improved': improved, 'bad_epochs': bad}
        if epoch >= 10 and bad >= 5:
            break
    assert epoch == state['epochs']
    assert 10 <= epoch <= 20
    assert state['best_epoch'] == expected_best[-1][0]
    exact(best, expected_best)
    exact(final, {'model': model.state_dict(), 'optimizer': opt.state_dict(),
        'scheduler': sched.state_dict(), 'rng': torch.get_rng_state(), 'sampler': sample_generator.get_state()})
    if reversed_validation:
        assert epoch == 10
    else:
        assert epoch == 20
