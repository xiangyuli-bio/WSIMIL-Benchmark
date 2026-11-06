"""Complete native-width dual heads and strict TRAIN-only teacher boundaries."""
from copy import deepcopy

import pytest
import torch

from milplus.dspamil_equations import EquationChoices, distillation_losses
from milplus.dspamil_reference import DualConceptHeads, HeadReference, _paper_post_softmax_attention
from milplus.dspamil_source import author_definitions


def full_heads():
    torch.set_num_threads(2)
    torch.manual_seed(84)
    equations = EquationChoices('instances_reference', 'student_to_teacher_eq8', True, 'l2_norm')
    reference = HeadReference(1024, 1024, 1024, 'gelu', .1,
        'Explicit native-width mechanism software check; author head recipe unavailable')
    return DualConceptHeads(torch.randn(3, 4, 1024), equations=equations, heads=reference,
                            concept_contract='unpretrained_component_synthetic_concepts_not_original_assets')


def test_all_native_dual_teacher_student_and_auxiliary_modules_receive_objective_gradients():
    model = full_heads()
    x = torch.randn(2, 257, 1024)
    patterns = torch.randn(2, 3, 1024, requires_grad=True)
    text = torch.randn(2, 1024)
    output = model(x, patterns, true_grade_concept=text)
    assert output['dual_patterns'].shape == (2, 3, 2048)
    assert output['pseudo_text'].shape == (2, 1024)
    assert output['pattern_logits'].shape == (2, 3, 2)
    assert output['teacher_logits'].shape == output['student_logits'].shape == (2, 6)
    losses = distillation_losses(output, torch.tensor([2, 5]), torch.tensor([[1, 1, 0], [0, 1, 1]]),
                                 text, choices=model.equations)
    losses['total'].backward()
    assert model.pattern_concepts.grad is None and not model.pattern_concepts.requires_grad
    assert patterns.grad is not None and torch.count_nonzero(patterns.grad)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert all(any(torch.count_nonzero(p.grad) for p in child.parameters())
        for child in [model.teacher_gate, model.student_gate, model.teacher_classifier,
            model.student_classifier, model.pseudo_text_generator, *model.pattern_classifiers])


def test_student_inference_never_reads_teacher_modules_or_true_grade_concepts():
    model = full_heads().eval()
    def forbidden(*args):
        raise AssertionError('Teacher was called during held-out inference')
    first = model.teacher_gate.register_forward_pre_hook(forbidden)
    second = model.teacher_classifier.register_forward_pre_hook(forbidden)
    x, patterns = torch.randn(2, 257, 1024), torch.randn(2, 3, 1024)
    with torch.no_grad():
        before = model(x, patterns)
        for module in [model.teacher_gate, model.teacher_classifier]:
            for parameter in module.parameters():
                parameter.add_(10.)
        after = model(x, patterns)
    assert not any(key.startswith('teacher') for key in before)
    assert torch.equal(before['student_logits'], after['student_logits'])
    assert torch.equal(before['student_gate_weights'], after['student_gate_weights'])
    with pytest.raises(ValueError, match='must not consume'):
        model(x, patterns, true_grade_concept=torch.randn(2, 1024))
    first.remove()
    second.remove()
    model.train()
    with pytest.raises(ValueError, match='actual TRAIN'):
        model(x, patterns)


def test_complete_head_checkpoint_rejects_changed_concepts_or_equation_conventions():
    model = full_heads()
    state = deepcopy(model.state_dict())
    restored = full_heads()
    restored.load_state_dict(state)
    restored.equations = EquationChoices('concepts_last_axis_reference', 'student_to_teacher_eq8', True, 'l2_norm')
    with pytest.raises(ValueError, match='equation conventions'):
        restored.load_state_dict(state)
    different = full_heads()
    different.concept_contract = 'different_unverified_text_source'
    with pytest.raises(ValueError, match='concept identities'):
        different.load_state_dict(state)


def test_post_softmax_paper_mask_is_explicit_and_does_not_renormalize_or_modify_native_shapes():
    from types import MethodType
    torch.manual_seed(2)
    source = author_definitions().Attention(dim=1024, num_heads=8, attn_drop=0., proj_drop=0.)
    paper = author_definitions().Attention(dim=1024, num_heads=8, attn_drop=0., proj_drop=0.)
    paper.load_state_dict(source.state_dict())
    paper.forward = MethodType(_paper_post_softmax_attention, paper)
    value = torch.zeros(1, 260, 1024)
    _, a = source(value)
    _, b = paper(value)
    assert torch.equal(b, a * source.gen_mask(a))
    assert torch.count_nonzero(a[:, :, 0, 1:3]) == 16
    assert torch.count_nonzero(b[:, :, 0, 1:3]) == 0
    assert bool((b[:, :, :3].sum(-1) < 1).all())
