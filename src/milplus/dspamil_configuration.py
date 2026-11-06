"""Strict configuration for the public complete DSPA2025 reference profiles."""
from dataclasses import asdict

from .dspamil_equations import EquationChoices
from .dspamil_reference import HeadReference
from .dspamil_training import TrainingChoices, reference_recipe, validate_recipe

from .dspamil_spec import EXECUTION_CONDITIONS as PROFILES


def construct(kind, value):
    if not isinstance(value, dict):
        raise ValueError('DSPA reference choices must be explicit named mappings')
    try:
        return kind(**value).validate()
    except TypeError as error:
        raise ValueError('DSPA reference choices have missing or unknown named fields') from error


def validate(config):
    keys = {'mask_convention', 'equations', 'heads', 'recipe', 'threads', 'runtime_python', 'audit_checkpoint_updates'}
    if (not isinstance(config, dict) or set(config) != keys
            or config['mask_convention'] not in set(PROFILES.values())
            or type(config['threads']) is not int or config['threads'] < 1):
        raise ValueError('DSPA needs the complete native graph, equation and runtime reference configuration')
    audit = config['audit_checkpoint_updates']
    if (not isinstance(audit, list) or any(type(i) is not int or i < 1 for i in audit)
            or audit != sorted(set(audit))):
        raise ValueError('DSPA optional audit checkpoints require ordered unique positive actual update counts')
    construct(EquationChoices, config['equations'])
    heads = construct(HeadReference, config['heads'])
    if any(getattr(heads, key) != 1024 for key in ['ptg_hidden', 'teacher_hidden', 'student_hidden']):
        raise ValueError('This complete DSPA reference keeps all native1024-width dual heads; no reduced hidden graph')
    validate_recipe(config['recipe'])
    return config


def resolve(method, bundle, supplied):
    if method not in PROFILES:
        raise ValueError('Choose one of the explicit full DSPA2025 mask references; DSPA-U is not implemented')
    if (not isinstance(supplied, dict) or set(supplied) - {'threads', 'model'}
            or not isinstance(supplied.get('model'), dict)):
        raise ValueError('DSPA requires explicit equation/head/training choices, with no generic budget overrides')
    model = supplied['model']
    required = {'equations', 'heads', 'training'}
    if not required <= set(model) or set(model) - (required | {'runtime_python', 'stop_after_train_batches', 'audit_checkpoint_updates'}):
        raise ValueError('Declare all unpublished DSPA reference choices; keep the complete native graph and50epoch recipe')
    equations = construct(EquationChoices, model['equations'])
    heads = construct(HeadReference, model['heads'])
    training = construct(TrainingChoices, model['training'])
    if training.evaluation_unit != bundle.config.get('evaluation_unit', 'slide'):
        raise ValueError('DSPA validation selection and final task evaluation must use the same declared patient/slide unit')
    config = {'mask_convention': PROFILES[method], 'equations': asdict(equations),
        'heads': asdict(heads), 'recipe': reference_recipe(training),
        'threads': supplied.get('threads', 2), 'runtime_python': model.get('runtime_python'),
        'audit_checkpoint_updates': model.get('audit_checkpoint_updates', [])}
    return validate(config), model.get('stop_after_train_batches')
