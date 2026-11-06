"""Known MICCAI equations, actual native dimensions, explicit ambiguous choices."""
import itertools

import pytest
import torch

from milplus.dspamil_equations import (
    EquationChoices, concept_aggregation, distillation_losses, expected_grade)


def choices(axis='instances_reference', direction='student_to_teacher_eq8', detach=True, norm='l2_norm'):
    return EquationChoices(axis, direction, detach, norm)


@pytest.mark.parametrize('axis', ['instances_reference', 'concepts_last_axis_reference'])
def test_complete3by4_concept_aggregation_retains_last_of1025_native_instances(axis):
    features = torch.zeros(2, 1025, 1024)
    features[:, -1] = torch.arange(1024) % 3 - 1
    concepts = torch.zeros(3, 4, 1024)
    output, weights, per_concept = concept_aggregation(features, concepts, softmax_axis=axis)
    assert output.shape == (2, 3, 1024) and per_concept.shape == (2, 3, 4, 1024)
    assert weights.shape == (2, 3, 1025, 4)
    scalar = .25 if axis == 'concepts_last_axis_reference' else 1 / 1025
    assert torch.equal(weights, torch.full_like(weights, scalar))
    # The sole nonzero patch is past position1024. A cap/padded replacement
    # would lose it, independently of the chosen aggregation normalization.
    assert torch.count_nonzero(output) > 0
    for k in range(3):
        assert torch.equal(per_concept[:, k, 0], features[:, -1] * scalar)
        assert torch.equal(per_concept[:, k, 0], per_concept[:, k, 3])
    features[:, -1] = 0
    assert torch.count_nonzero(concept_aggregation(features, concepts, softmax_axis=axis)[0]) == 0


def outputs(batch=2, seed=31):
    generator = torch.Generator().manual_seed(seed)
    shapes = {'teacher_logits': (batch, 6), 'student_logits': (batch, 6),
        'teacher_gate_logits': (batch, 3), 'student_gate_logits': (batch, 3),
        'pseudo_text': (batch, 1024), 'pattern_logits': (batch, 3, 2)}
    return {k: torch.randn(shape, generator=generator).requires_grad_() for k, shape in shapes.items()}


@pytest.mark.parametrize('direction,detach,norm', list(itertools.product(
    ['student_to_teacher_eq8', 'teacher_to_student_prose'], [True, False],
    ['l2_norm', 'squared_l2_reference'])))
def test_every_declared_KL_detach_and_text_convention_has_correct_gradient_edges(direction, detach, norm):
    values = outputs()
    losses = distillation_losses(values, torch.tensor([1, 5]), torch.tensor([[1, 1, 0], [0, 1, 1]]),
        torch.zeros(2, 1024), choices=choices(direction=direction, detach=detach, norm=norm))
    assert set(losses) == {'teacher_mse', 'student_mse', 'gating_kl', 'output_kl',
                           'text_alignment', 'pattern_ce_mean_over3', 'total'}
    losses['total'].backward()
    for name in ['teacher_logits', 'student_logits', 'student_gate_logits', 'pseudo_text', 'pattern_logits']:
        assert values[name].grad is not None and torch.isfinite(values[name].grad).all(), name
        assert torch.count_nonzero(values[name].grad), name
    if detach:
        assert values['teacher_gate_logits'].grad is None
    else:
        assert values['teacher_gate_logits'].grad is not None
        assert torch.count_nonzero(values['teacher_gate_logits'].grad)


def test_closed_form_all_terms_preserve_L2_norm_and_one_third_auxiliary_weight():
    values = {key: torch.zeros_like(value) for key, value in outputs(batch=1).items()}
    true = torch.zeros(1, 1024)
    true[0, 0] = 2
    losses = distillation_losses(values, torch.tensor([3]), torch.tensor([[1, 0, 0]]), true, choices=choices())
    assert losses['teacher_mse'].item() == losses['student_mse'].item() == .25
    assert losses['gating_kl'].item() == losses['output_kl'].item() == 0
    assert losses['text_alignment'].item() == 2
    assert torch.equal(losses['pattern_ce_mean_over3'], torch.log(torch.tensor(2.)))
    assert torch.equal(losses['total'], torch.tensor(.5) + torch.tensor(.2) + .01 * torch.log(torch.tensor(2.)))
    squared = distillation_losses(values, torch.tensor([3]), torch.tensor([[1, 0, 0]]), true,
                                 choices=choices(norm='squared_l2_reference'))
    assert squared['text_alignment'].item() == 4
    assert torch.equal(expected_grade(values['student_logits']), torch.tensor([2.5]))


def test_measured_pattern_labels_and_native_alignment_are_required_instead_of_guessed_GG_rules():
    values = outputs()
    with pytest.raises(ValueError, match='measured'):
        distillation_losses(values, torch.tensor([4, 5]), torch.tensor([4, 5]), torch.zeros(2, 1024), choices=choices())
    with pytest.raises(ValueError, match='aligned'):
        concept_aggregation(torch.randn(1, 257, 1024), torch.randn(3, 4, 512), softmax_axis='instances_reference')
    with pytest.raises(ValueError, match='Declare'):
        choices(direction='unspecified').validate()
