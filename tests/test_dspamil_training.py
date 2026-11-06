"""Fifty-epoch control on complete dual heads; not full-PAT/asset validation.

The production PAT is never replaced in a public run. This isolated control
supplies pattern tensors directly to test the full-width teacher/student heads,
all losses, held-out selection and exact optimizer-boundary recovery.
"""
from copy import deepcopy

import pytest
import torch

from milplus.dspamil_equations import EquationChoices
from milplus.dspamil_reference import DualConceptHeads, HeadReference
from milplus.dspamil_training import (
    TrainingChoices, fit, grade_predictions, new_progress, reference_recipe, setup, validation)


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for first, second in zip(a, b):
            exact(first, second)
    else:
        assert a == b


def case():
    torch.set_num_threads(2)
    torch.manual_seed(43)
    equations = EquationChoices('instances_reference', 'student_to_teacher_eq8', True, 'l2_norm')
    heads = HeadReference(1024, 1024, 1024, 'gelu', .1, 'Full-width synthetic component control, not original head evidence')
    concepts = torch.randn(3, 4, 1024) * .01

    class Component(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = DualConceptHeads(concepts, equations=equations, heads=heads,
                concept_contract='explicit_synthetic_unpretrained_component')

        def forward(self, features, *, true_grade_concept=None):
            patterns = features.mean(1).unsqueeze(1).repeat(1, 3, 1)
            return self.heads(features, patterns, true_grade_concept=true_grade_concept)

    model = Component()
    features = {subset + str(i): torch.randn(257, 1024) * .01
                for subset in ['train', 'val'] for i in range(6)}
    grade_text = torch.randn(6, 1024) * .01
    train = [{'slide_id': 'train' + str(i), 'subset': 'train', 'target': i} for i in range(6)]
    val = [{'slide_id': 'val' + str(i), 'subset': 'val', 'target': i} for i in range(6)]

    class Inputs:
        supervision_calls = []

        def load_features(self, row):
            return features[row['slide_id']]

        def training_supervision(self, row):
            assert row['subset'] == 'train'
            self.supervision_calls.append(row['slide_id'])
            grade = row['target']
            return torch.tensor(grade), torch.tensor([grade % 2, (grade // 2) % 2, (grade // 4) % 2]), grade_text[grade]

    recipe = reference_recipe(TrainingChoices(50, 50, 'half_up_expected_grade_reference',
        'Explicit controller reference: all50epochs, no historical paper result asserted'))
    return model, train, val, Inputs(), recipe, equations


def snapshot(model, optimizer, generator, progress):
    return deepcopy({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'generator': generator.get_state(), 'rng': torch.get_rng_state(), 'progress': progress,
        'gradients': {name: None if value.grad is None else value.grad.detach().clone()
                      for name, value in model.named_parameters()}})


def test_full_width_dual_heads_all50epochs300updates_and_update7_recovery_are_exact():
    model, train, val, inputs, recipe, equations = case()
    original, initial_rng = deepcopy(model.state_dict()), torch.get_rng_state()
    optimizer, generator = setup(model, seed=31, recipe=recipe)
    progress = new_progress()
    checkpoints = []
    result = fit(model, optimizer, generator, train, val, inputs, progress, device='cpu',
        recipe=recipe, equations=equations, on_checkpoint=lambda p, best: checkpoints.append((p['epochs'], best)))
    final = snapshot(model, optimizer, generator, progress)
    assert result == 'completed' and progress['epochs'] == 50
    assert progress['train_batches'] == progress['optimizer_updates'] == 300
    assert len(checkpoints) == len(progress['history']) == 50
    assert len(inputs.supervision_calls) == 300
    assert all(sorted(row['train_order']) == sorted(r['slide_id'] for r in train) for row in progress['history'])
    assert all(row['validation']['student_only'] and not row['validation']['true_grade_concepts_consumed']
               for row in progress['history'])
    assert set(progress['history'][0]['train_loss_terms']) == {'teacher_mse', 'student_mse',
        'gating_kl', 'output_kl', 'text_alignment', 'pattern_ce_mean_over3', 'total'}
    # Every trainable dual-head tensor received Adam first/second moments.
    assert len(optimizer.state) == len(list(model.parameters()))
    model.load_state_dict(original)
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(initial_rng)
    optimizer, generator = setup(model, seed=31, recipe=recipe)
    progress = new_progress()
    saved = []
    def checkpoint(state, best):
        if state['train_batches'] == 7:
            saved.append(snapshot(model, optimizer, generator, state))
    assert fit(model, optimizer, generator, train, val, inputs, progress, device='cpu',
        recipe=recipe, equations=equations, on_checkpoint=checkpoint, stop_after_batch=7) == 'interrupted'
    assert saved[0]['progress']['epochs'] == 1 and saved[0]['progress']['cursor'] == 1
    model, train, val, inputs, recipe, equations = case()
    optimizer, generator = setup(model, seed=31, recipe=recipe)
    value = saved[0]
    model.load_state_dict(value['model'])
    optimizer.load_state_dict(value['optimizer'])
    generator.set_state(value['generator'])
    for name, p in model.named_parameters():
        p.grad = None if value['gradients'][name] is None else value['gradients'][name].clone()
    torch.set_rng_state(value['rng'])
    assert fit(model, optimizer, generator, train, val, inputs, value['progress'], device='cpu',
        recipe=recipe, equations=equations, on_checkpoint=lambda *unused: None) == 'completed'
    exact(snapshot(model, optimizer, generator, value['progress']), final)


def test_ordinal_selection_rule_is_explicit_and_test_rows_never_enter_validation():
    logits = torch.zeros(1, 6)
    assert grade_predictions(logits, 'half_up_expected_grade_reference').tolist() == [3]
    assert grade_predictions(logits, 'argmax_logits_reference').tolist() == [0]
    with pytest.raises(ValueError, match='decision'):
        grade_predictions(logits, 'automatic')
    with pytest.raises(ValueError, match='validation slides only'):
        validation(None, [{'subset': 'test'}], None, 'cpu', decision='half_up_expected_grade_reference')


@pytest.mark.parametrize('key,value', [('epochs', 2), ('lr', .001), ('weight_decay', .0),
    ('scheduler_reference', 'cosine'), ('batch_size_reference', 8)])
def test_paper_budget_and_declared_full_recipe_cannot_be_silently_reduced(key, value):
    recipe = reference_recipe(TrainingChoices(10, 10, 'half_up_expected_grade_reference', 'Explicit reference'))
    recipe[key] = value
    with pytest.raises(ValueError, match='full50epoch'):
        setup(torch.nn.Linear(2, 2), seed=31, recipe=recipe)
