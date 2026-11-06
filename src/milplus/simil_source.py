"""Private pinned SI-MIL source, with an explicit full Gaussian VJP repair."""
import sys
import types

import numpy as np
import torch

from .upstream import load_definitions, pinned_text, verified_source

PROFILES={'simil_source':'source','simil_full_vjp':'full_vjp'}
COMMIT='3ffb218672ffe0e4a183924af50e5920b498483d'


def load_model(variant):
    if variant not in PROFILES.values():
        raise ValueError('SI-MIL requires a named original/full-VJP derivative implementation')
    root,sha=verified_source('simil')
    if sha!=COMMIT:
        raise ValueError('SI-MIL source identity differs from the audited implementation')
    prefix='_milplus_simil_'+sha+'_'+variant
    if prefix+'.model' in sys.modules:
        return sys.modules[prefix+'.model']
    topk=types.ModuleType(prefix+'_topk')
    topk.__file__=str(root/'perturbedtopk.py')
    exec(compile(pinned_text(root,sha,'perturbedtopk.py'),topk.__file__,'exec'),topk.__dict__)  # noqa: S102 -- verified external Git object
    if variant=='full_vjp':
        class FullVJP(topk.PerturbedTopKFunction):
            @staticmethod
            def backward(ctx,grad_output):
                if grad_output is None:
                    return (None,)*4
                # Full score-function VJP, including all cross-coordinate derivatives,
                # contracted directly over samples without an N-by-N Jacobian.
                sample_values=torch.einsum('bnkd,bkd->bn',ctx.perturbed_output,grad_output)
                grad=torch.einsum('bn,bnd->bd',sample_values,ctx.noise)/ctx.num_samples/ctx.sigma
                return grad,None,None,None
        topk.PerturbedTopKFunction=FullVJP
    sys.modules[topk.__name__]=topk
    module=types.ModuleType(prefix+'.model');module.__file__=str(root/'si_mil.py')
    code=pinned_text(root,sha,'si_mil.py')
    before='from perturbedtopk import PerturbedTopK'
    if code.count(before)!=1:
        raise ValueError('SI-MIL import-only compatibility contract changed')
    code=code.replace(before,'from '+topk.__name__+' import PerturbedTopK')
    exec(compile(code,module.__file__,'exec'),module.__dict__)  # noqa: S102 -- verified source with import-only changes
    sys.modules[module.__name__]=module
    return module


def patch_dropout():
    return load_definitions('simil','train.py',['dropout_patches'],{'np':np}).dropout_patches
