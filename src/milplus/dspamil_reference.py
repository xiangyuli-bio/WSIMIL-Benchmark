"""Full native-width DSPA2025 mechanism candidate, with explicit reconstruction.

The original SAT transformer/token definitions are loaded unchanged. Missing
dual heads are independently implemented and named reference choices. Neither
this candidate nor its component tests establish a full admitted DSPA method.
DSPA-U uncertainty is not fabricated from its unavailable full specification.
"""
from dataclasses import asdict, dataclass
import hashlib
from types import MethodType

import torch
from torch import nn

from .dspamil_equations import DIMENSION, EquationChoices, concept_aggregation, expected_grade
from .dspamil_source import COMMIT, build_native_source_component


@dataclass(frozen=True)
class HeadReference:
    ptg_hidden: int
    teacher_hidden: int
    student_hidden: int
    activation: str
    dropout: float
    provenance: str

    def validate(self):
        if (any(type(getattr(self, k)) is not int or getattr(self, k) < 1
                for k in ['ptg_hidden', 'teacher_hidden', 'student_hidden'])
                or self.activation not in {'gelu', 'relu'}
                or not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1
                or not isinstance(self.provenance, str) or not self.provenance.strip()):
            raise ValueError('Declare all unpublished DSPA head widths/activation/dropout and reconstruction provenance')
        return self


def _mlp(input_dim, hidden, output_dim, spec):
    activation = nn.GELU if spec.activation == 'gelu' else nn.ReLU
    return nn.Sequential(nn.Linear(input_dim, hidden), activation(), nn.Dropout(spec.dropout),
                         nn.Linear(hidden, output_dim))


def _paper_post_softmax_attention(self, x):
    batch, count, width = x.shape
    qkv = self.qkv(x).reshape(batch, count, 3, self.num_heads, width // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    probability = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
    if self.clstoken_mask:
        probability = probability * self.gen_mask(probability)
    # Eq2's printed full denominator is retained: do not silently renormalize.
    value = (self.attn_drop(probability) @ v).transpose(1, 2).reshape(batch, count, width)
    return self.proj_drop(self.proj(value)), probability


class NativePAT(nn.Module):
    def __init__(self, mask_convention):
        super().__init__()
        if mask_convention not in {'released_zero_logits', 'paper_post_softmax_no_renormalization_reference'}:
            raise ValueError('Declare the released or printed-equation DSPA attention-mask convention')
        source = build_native_source_component()
        # Retain the exact12x1024/8head original backbone and three native tokens.
        # SAT's baseline-only fusion/auxiliary heads are replaced by the complete
        # dual-stream heads below, rather than reported as a full DSPA model.
        self.transformer = source.transformer
        self.virtual_token_feature = source.virtual_token_feature
        self.mask_convention = mask_convention
        assert sum(p.numel() for p in self.parameters()) == 151120896
        if mask_convention != 'released_zero_logits':
            for block in self.transformer:
                block.attn.forward = MethodType(_paper_post_softmax_attention, block.attn)

    def forward(self, features):
        if features.ndim != 3 or features.shape[-1] != 1024 or features.shape[1] < 1:
            raise ValueError('DSPA PAT keeps every native1024D instance and all12original blocks')
        value = torch.cat([self.virtual_token_feature.repeat(len(features), 1, 1), features], dim=1)
        for block in self.transformer:
            value, unused = block(value)
        return value[:, :3, :]


class DualConceptHeads(nn.Module):
    """Complete dual pooling, teacher, PTG, student and pattern auxiliaries."""
    def __init__(self, pattern_concepts, *, equations, heads, concept_contract):
        super().__init__()
        if not isinstance(equations, EquationChoices) or not isinstance(heads, HeadReference):
            raise ValueError('Declare all equation and missing-head reference choices')
        equations.validate()
        heads.validate()
        if (not isinstance(pattern_concepts, torch.Tensor) or pattern_concepts.shape != (3, 4, 1024)
                or pattern_concepts.dtype != torch.float32 or pattern_concepts.requires_grad
                or not torch.isfinite(pattern_concepts).all()
                or not isinstance(concept_contract, str) or not concept_contract.strip()):
            raise ValueError('DSPA needs all3x4frozen aligned1024D expert concepts and their provenance contract')
        self.equations, self.head_reference = equations, heads
        self.concept_contract = concept_contract
        self.concepts_sha256 = hashlib.sha256(pattern_concepts.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        self.register_buffer('pattern_concepts', pattern_concepts.detach().clone())
        self.teacher_gate = nn.Linear(1024, 3)
        self.student_gate = nn.Linear(1024, 3)
        self.pseudo_text_generator = _mlp(3 * 2048, heads.ptg_hidden, 1024, heads)
        self.teacher_classifier = _mlp(2048, heads.teacher_hidden, 6, heads)
        self.student_classifier = _mlp(2048, heads.student_hidden, 6, heads)
        self.pattern_classifiers = nn.ModuleList([nn.Linear(2048, 2) for unused in range(3)])

    def get_extra_state(self):
        return {'kind': 'DSPA_MICCAI2025_explicit_reference_heads_not_original_trainer',
            'equations': asdict(self.equations), 'heads': asdict(self.head_reference),
            'concept_contract': self.concept_contract, 'concepts_sha256': self.concepts_sha256,
            'concatenation': 'each_Pk_is_PAT1024_concat_concept1024_PTGenerator_input6144',
            'grade_interface': 'six_logits_expected_ordinal_grade_reference', 'formal_admission': False}

    def set_extra_state(self, value):
        if value != self.get_extra_state():
            raise ValueError('DSPA concept identities, equation conventions or declared head architecture changed')

    def forward(self, native_features, patterns, *, true_grade_concept=None):
        if patterns.shape != (len(native_features), 3, DIMENSION):
            raise ValueError('DSPA heads need all three actual PAT native1024D patterns')
        if self.training and true_grade_concept is None:
            raise ValueError('DSPA teacher training needs the actual TRAIN grade-concept embedding')
        if not self.training and true_grade_concept is not None:
            raise ValueError('DSPA inference must not consume ground-truth grade concepts')
        concepts, weights, unused = concept_aggregation(native_features, self.pattern_concepts,
            softmax_axis=self.equations.concept_softmax_axis)
        dual = torch.cat([patterns, concepts], dim=-1)
        pseudo = self.pseudo_text_generator(dual.flatten(start_dim=1))
        student_gate_logits = self.student_gate(pseudo)
        student_weights = student_gate_logits.softmax(dim=-1)
        student_feature = (dual * student_weights.unsqueeze(-1)).sum(dim=1)
        student_logits = self.student_classifier(student_feature)
        output = {'dual_patterns': dual, 'concept_pattern_features': concepts,
            'concept_attention': weights, 'pseudo_text': pseudo,
            'student_gate_logits': student_gate_logits, 'student_gate_weights': student_weights,
            'student_logits': student_logits, 'student_grade': expected_grade(student_logits),
            'pattern_logits': torch.stack([head(dual[:, i]) for i, head in enumerate(self.pattern_classifiers)], dim=1)}
        if self.training:
            if (true_grade_concept.shape != (len(native_features), DIMENSION)
                    or true_grade_concept.requires_grad or not torch.isfinite(true_grade_concept).all()):
                raise ValueError('Provide native frozen true TRAIN grade-concept embeddings')
            logits = self.teacher_gate(true_grade_concept)
            teacher_weights = logits.softmax(dim=-1)
            feature = (dual * teacher_weights.unsqueeze(-1)).sum(dim=1)
            teacher_logits = self.teacher_classifier(feature)
            output.update(teacher_gate_logits=logits, teacher_gate_weights=teacher_weights,
                          teacher_logits=teacher_logits, teacher_grade=expected_grade(teacher_logits))
        return output


class DSPAMICCAIReference(nn.Module):
    """Full151M native PAT plus every known dual teacher/student mechanism.

This candidate still needs source/asset/label/training/native/release admission.
The unavailable DSPA-U uncertainty branch is deliberately not guessed.
"""
    def __init__(self, pattern_concepts, *, equations, heads, concept_contract, mask_convention):
        super().__init__()
        self.pat = NativePAT(mask_convention)
        self.dual_heads = DualConceptHeads(pattern_concepts, equations=equations, heads=heads,
                                           concept_contract=concept_contract)

    def forward(self, native_features, *, true_grade_concept=None):
        return self.dual_heads(native_features, self.pat(native_features),
                               true_grade_concept=true_grade_concept)

    def get_extra_state(self):
        return {'source_commit': COMMIT, 'mask_convention': self.pat.mask_convention,
            'PAT_native_parameters': 151120896, 'original_SAT_baseline_parameters': 151141388,
            'original_baseline_only_heads_replaced_by_full_dual_heads': True,
            'known_mechanism_reference_not_original_reproduction': True,
            'DSPA_U_implemented': False, 'formal_admission': False}

    def set_extra_state(self, value):
        if value != self.get_extra_state():
            raise ValueError('DSPA original PAT graph, mask or conference/journal variant changed')
