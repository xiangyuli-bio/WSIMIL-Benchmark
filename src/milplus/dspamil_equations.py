"""MICCAI2025 DSPA equations with unresolved conventions named explicitly.

This is a reconstruction component, not the missing author trainer or DSPA-U.
The paper gives incompatible gating-KL orders in its prose and Eq8 and leaves
the softmax axis and ordinal-logit interface unspecified. No convention is
silently selected. Original concept texts/alignment and pattern labels remain
required inputs to any future complete method admission.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

DIMENSION = 1024
PATTERNS = (3, 4, 5)
CONCEPTS_PER_PATTERN = 4
GRADE_CLASSES = 6


@dataclass(frozen=True)
class EquationChoices:
    concept_softmax_axis: str
    gating_kl_direction: str
    detach_teacher_distillation: bool
    text_alignment: str

    def validate(self):
        if self.concept_softmax_axis not in {'instances_reference', 'concepts_last_axis_reference'}:
            raise ValueError('Declare the unspecified DSPA concept-weight softmax axis')
        if self.gating_kl_direction not in {'student_to_teacher_eq8', 'teacher_to_student_prose'}:
            raise ValueError('Declare which of the two published gating-KL orders is used')
        if type(self.detach_teacher_distillation) is not bool:
            raise ValueError('Explicitly declare the unpublished teacher-detach convention')
        if self.text_alignment not in {'l2_norm', 'squared_l2_reference'}:
            raise ValueError('Declare the text-alignment norm convention')
        return self


def _finite(name, value):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError('DSPA needs finite floating-point ' + name)


def concept_aggregation(features, concepts, *, softmax_axis):
    """Full Eq4 aggregation; retain all patches and all3x4 aligned concepts."""
    _finite('native UNI features', features)
    _finite('aligned expert concepts', concepts)
    if (features.ndim != 3 or features.shape[-1] != DIMENSION or features.shape[1] < 1
            or concepts.shape != (3, CONCEPTS_PER_PATTERN, DIMENSION)
            or concepts.device != features.device or concepts.dtype != features.dtype):
        raise ValueError('DSPA Eq4 needs full[B,N,1024] features and aligned[3,4,1024] concepts')
    if softmax_axis not in {'instances_reference', 'concepts_last_axis_reference'}:
        raise ValueError('Declare the DSPA concept-weight softmax axis')
    scores = torch.einsum('bnd,kmd->bknm', features, concepts)
    weights = scores.softmax(dim=-2 if softmax_axis == 'instances_reference' else -1)
    representations = torch.einsum('bknm,bnd->bkmd', weights, features)
    return representations.mean(dim=2), weights, representations


def expected_grade(logits):
    _finite('six-grade logits', logits)
    if logits.ndim != 2 or logits.shape[-1] != GRADE_CLASSES:
        raise ValueError('The explicit six-logit ordinal reference needs[B,6] outputs')
    probabilities = logits.float().softmax(dim=-1)
    return (probabilities * torch.arange(6, device=logits.device, dtype=probabilities.dtype)).sum(dim=-1)


def _kl(p_logits, q_logits):
    p = p_logits.float().log_softmax(dim=-1)
    q = q_logits.float().log_softmax(dim=-1)
    return (p.exp() * (p - q)).sum(dim=-1).mean()


def distillation_losses(outputs, grades, pattern_targets, true_grade_concepts, *, choices):
    """Every Eq8 term, using an explicit six-logit expectation ordinal port.

Grade Group alone cannot establish all pattern-presence targets. Require the
three actual target columns; never invent a GG4/GG5-to-pattern mapping here.
Teacher detach, when requested, affects only distillation and not its own MSE.
"""
    if not isinstance(choices, EquationChoices):
        raise ValueError('Declare the complete DSPA equation conventions')
    choices.validate()
    required = {'teacher_logits', 'student_logits', 'teacher_gate_logits',
                'student_gate_logits', 'pseudo_text', 'pattern_logits'}
    if not required <= outputs.keys():
        raise ValueError('DSPA needs the complete teacher/student/gates/text/pattern outputs')
    batch = len(outputs['student_logits'])
    shapes = {'teacher_logits': (batch, 6), 'student_logits': (batch, 6),
        'teacher_gate_logits': (batch, 3), 'student_gate_logits': (batch, 3),
        'pseudo_text': (batch, DIMENSION), 'pattern_logits': (batch, 3, 2)}
    for name, shape in shapes.items():
        _finite(name, outputs[name])
        if outputs[name].shape != shape:
            raise ValueError('DSPA native output shape changed: ' + name)
    _finite('frozen true grade-concept embeddings', true_grade_concepts)
    if (true_grade_concepts.shape != (batch, DIMENSION) or true_grade_concepts.requires_grad
            or grades.shape != (batch,) or grades.dtype != torch.long
            or pattern_targets.shape != (batch, 3) or pattern_targets.dtype != torch.long
            or batch < 1 or (grades < 0).any() or (grades > 5).any()
            or (pattern_targets < 0).any() or (pattern_targets > 1).any()):
        raise ValueError('Supply measured six-grade and all3binary pattern targets and frozen native concept embeddings')
    device = outputs['student_logits'].device
    if any(value.device != device for value in [grades, pattern_targets, true_grade_concepts,
                                               *[outputs[name] for name in shapes]]):
        raise ValueError('DSPA objective tensors must share the actual model device')
    teacher_grade = expected_grade(outputs['teacher_logits'])
    student_grade = expected_grade(outputs['student_logits'])
    teacher_logits = outputs['teacher_logits']
    teacher_gate = outputs['teacher_gate_logits']
    if choices.detach_teacher_distillation:
        teacher_logits = teacher_logits.detach()
        teacher_gate = teacher_gate.detach()
    if choices.gating_kl_direction == 'student_to_teacher_eq8':
        gate = _kl(outputs['student_gate_logits'], teacher_gate)
    else:
        gate = _kl(teacher_gate, outputs['student_gate_logits'])
    logits = _kl(teacher_logits, outputs['student_logits'])
    difference = true_grade_concepts.float() - outputs['pseudo_text'].float()
    text = (difference.square().sum(dim=-1) if choices.text_alignment == 'squared_l2_reference'
            else torch.linalg.vector_norm(difference, ord=2, dim=-1)).mean()
    auxiliary = F.cross_entropy(outputs['pattern_logits'].float().reshape(-1, 2), pattern_targets.reshape(-1))
    result = {'teacher_mse': F.mse_loss(teacher_grade, grades.float()),
        'student_mse': F.mse_loss(student_grade, grades.float()),
        'gating_kl': gate, 'output_kl': logits, 'text_alignment': text,
        'pattern_ce_mean_over3': auxiliary}
    result['total'] = (result['teacher_mse'] + result['student_mse'] + .1 * gate +
                       .1 * logits + .1 * text + .01 * auxiliary)
    if any(not torch.isfinite(value) for value in result.values()):
        raise FloatingPointError('DSPA complete objective produced a nonfinite term')
    return result
