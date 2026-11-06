"""Full fifty-epoch DSPA2025 training control with named missing details.

This is a reconstruction, not the broken released author's trainer. Every loss
and the fixed paper budget is retained; unspecified stopping, sampling and
ordinal-decision details must be explicitly declared by the caller.
"""
from dataclasses import asdict, dataclass
import math

import pandas as pd
import torch

from .dspamil_equations import distillation_losses


@dataclass(frozen=True)
class TrainingChoices:
    patience: int
    min_epochs: int
    ordinal_decision: str
    provenance: str
    evaluation_unit: str = 'slide'

    def validate(self):
        if (type(self.patience) is not int or not 1 <= self.patience <= 50
                or type(self.min_epochs) is not int or not 1 <= self.min_epochs <= 50
                or self.ordinal_decision not in {'half_up_expected_grade_reference', 'argmax_logits_reference'}
                or self.evaluation_unit not in {'slide', 'patient'}
                or not isinstance(self.provenance, str) or not self.provenance.strip()):
            raise ValueError('Declare DSPA unpublished early-stop settings, ordinal decision and reference provenance')
        return self


def reference_recipe(choices):
    if not isinstance(choices, TrainingChoices):
        raise ValueError('DSPA needs explicit training reference choices')
    choices.validate()
    return {'epochs': 50, 'optimizer': 'AdamW', 'lr': 2e-4, 'weight_decay': 1e-5,
        'betas': [.9, .999], 'eps': 1e-8, 'foreach': False, 'fused': False,
        'batch_size_reference': 1, 'gradient_accumulation_reference': 1,
        'precision_reference': 'float32_without_AMP', 'gradient_clipping_reference': None,
        'scheduler_reference': None, 'sampling_reference': 'uniform_without_replacement_per_epoch',
        'sampler_rng': 'separate_cpu_torch_generator_seeded_with_run_seed',
        'validation': 'student_only_QWK_all_six_labels_fixed',
        'selection': 'strictly_greater_validation_QWK', 'QWK_tie': 'retain_earliest_epoch',
        'early_stop': 'nonimproving_completed_epochs_count_stop_after_declared_min_epochs',
        'reference_choices': asdict(choices), 'test_used_for_selection': False,
        'author_complete_trainer_available': False, 'paper_reproduction': False,
        'formal_admission': False}


def validate_recipe(recipe):
    if (not isinstance(recipe, dict) or 'reference_choices' not in recipe
            or recipe != reference_recipe(TrainingChoices(**recipe['reference_choices']))):
        raise ValueError('DSPA full50epoch paper budget, losses or declared training reference changed')
    return recipe


def setup(model, *, seed, recipe):
    validate_recipe(recipe)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('DSPA requires a complete32bit training seed')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5,
        betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return optimizer, generator


def new_progress():
    return {'epochs': 0, 'train_batches': 0, 'optimizer_updates': 0, 'cursor': 0,
        'order': None, 'epoch_loss_sums': {}, 'epoch_batches': 0, 'best_epoch': None,
        'best_validation_qwk': None, 'bad_epochs': 0, 'history': [], 'completed': False}


def grade_predictions(logits, decision):
    if logits.ndim != 2 or logits.shape[-1] != 6 or not torch.isfinite(logits).all():
        raise ValueError('DSPA ordinal evaluation needs finite six-grade student logits')
    from .dspamil_metrics import decide
    unused, predicted = decide(logits.detach().float().softmax(-1).cpu().numpy(), decision)
    return torch.as_tensor(predicted, device=logits.device)


@torch.no_grad()
def validation(model, val_rows, inputs, device, *, decision, unit='slide'):
    if not val_rows or any(row['subset'] != 'val' for row in val_rows):
        raise ValueError('DSPA checkpoint selection consumes validation slides only')
    if len({row['slide_id'] for row in val_rows}) != len(val_rows):
        raise ValueError('DSPA validation cannot repeat a slide')
    model.eval()
    records = []
    for row in val_rows:
        features = inputs.load_features(row).to(device).unsqueeze(0)
        # Never call training_supervision or pass a true-grade concept here.
        result = model(features)
        p = result['student_logits'].float().softmax(-1).cpu().numpy()[0]
        records.append({'slide_id': row['slide_id'], 'target': int(row['target']),
            **({'global_patient_id': row['global_patient_id']} if 'global_patient_id' in row else {}),
            'dspamil_decision': decision, **{f'p_{i}': float(v) for i, v in enumerate(p)}})
    from .dspamil_metrics import metrics, point_predictions
    frame = pd.DataFrame(records)
    metric = metrics(frame, rule=decision, unit=unit)
    point, targets, p, scores, predictions = point_predictions(frame, rule=decision, unit=unit)
    qwk = metric['qwk']
    if qwk is None or not math.isfinite(qwk):
        raise FloatingPointError('DSPA validation QWK is undefined for this actual population')
    return {'qwk': qwk, 'targets': targets.tolist(), 'predictions': predictions.tolist(),
            'evaluation_unit': unit, 'prediction_rule': decision,
            'decision_arithmetic': metric['decision_arithmetic'],
            'patient_ids': point.global_patient_id.tolist() if 'global_patient_id' in point else None,
            'slide_ids': [row['slide_id'] for row in val_rows],
            'student_only': True, 'true_grade_concepts_consumed': False}


def fit(model, optimizer, generator, train_rows, val_rows, inputs, state, *, device,
        recipe, equations, on_checkpoint, stop_after_batch=None, on_batch=None):
    validate_recipe(recipe)
    if (not train_rows or any(row['subset'] != 'train' for row in train_rows)
            or len({row['slide_id'] for row in train_rows}) != len(train_rows)):
        raise ValueError('DSPA optimizer must consume each actual TRAIN slide exactly once per epoch')
    if (stop_after_batch is not None
            and (type(stop_after_batch) is not int or stop_after_batch < 1)):
        raise ValueError('DSPA recovery boundary must be a positive actual TRAIN update count')
    choices = recipe['reference_choices']
    while state['epochs'] < 50 and not state['completed']:
        if state['order'] is None:
            state['order'] = torch.randperm(len(train_rows), generator=generator).tolist()
        if (sorted(state['order']) != list(range(len(train_rows)))
                or not 0 <= state['cursor'] <= len(train_rows)
                or state['epoch_batches'] != state['cursor']):
            raise ValueError('DSPA recovery changed the complete TRAIN permutation or actual minibatch cursor')
        while state['cursor'] < len(state['order']):
            model.train()
            row = train_rows[state['order'][state['cursor']]]
            features = inputs.load_features(row).to(device).unsqueeze(0)
            grade, pattern, concept = inputs.training_supervision(row)
            grade, pattern, concept = grade.to(device), pattern.to(device), concept.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(features, true_grade_concept=concept.unsqueeze(0))
            losses = distillation_losses(result, grade.unsqueeze(0), pattern.unsqueeze(0),
                concept.unsqueeze(0), choices=equations)
            losses['total'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('DSPA full objective produced a nonfinite parameter gradient')
            optimizer.step()
            state['train_batches'] += 1
            state['optimizer_updates'] += 1
            state['epoch_batches'] += 1
            state['cursor'] += 1
            for key in sorted(losses):
                state['epoch_loss_sums'][key] = state['epoch_loss_sums'].get(key, 0.) + float(losses[key].detach())
            if on_batch is not None:
                on_batch(state, row, losses)
            if state['train_batches'] == stop_after_batch:
                on_checkpoint(state, False)
                return 'interrupted'
        model.zero_grad(set_to_none=True)
        heldout = validation(model, val_rows, inputs, device, decision=choices['ordinal_decision'],
                            unit=choices['evaluation_unit'])
        state['epochs'] += 1
        improved = state['best_validation_qwk'] is None or heldout['qwk'] > state['best_validation_qwk']
        if improved:
            state['best_validation_qwk'] = heldout['qwk']
            state['best_epoch'] = state['epochs']
            state['bad_epochs'] = 0
        else:
            state['bad_epochs'] += 1
        state['history'].append({'epoch': state['epochs'],
            'train_loss_terms': {key: value / state['epoch_batches'] for key, value in state['epoch_loss_sums'].items()},
            'train_order': [train_rows[i]['slide_id'] for i in state['order']],
            'validation': heldout, 'improved': improved, 'bad_epochs': state['bad_epochs']})
        state['order'] = None
        state['cursor'] = state['epoch_batches'] = 0
        state['epoch_loss_sums'] = {}
        state['completed'] = (state['epochs'] == 50 or
            (state['epochs'] >= choices['min_epochs'] and state['bad_epochs'] >= choices['patience']))
        on_checkpoint(state, improved)
    return 'completed'
