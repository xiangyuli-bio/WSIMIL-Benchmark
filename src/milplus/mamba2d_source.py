"""Pinned original 2DMamba heads, scan variants, optimizer and censoring loss."""
import importlib.util
import sys
import types
from pathlib import Path

import torch

from .io import digest_file, read_json
from .mamba2d_scan import build_paper_scan, build_scan
from .upstream import load_definitions, package, pinned_text, verified_source

VARIANTS = {'mamba2dmil_source': 'source', 'mamba2dmil_autograd': 'autograd',
            'mamba2dmil_paper': 'paper'}


def author_model(profile, backend='torch', cuda_binary=None):
    if profile not in VARIANTS or backend not in {'torch', 'cuda'}:
        raise ValueError('2DMamba requires a registered source/repair and explicit torch/cuda backend')
    root, sha = verified_source('mamba2dmil')
    identity = {'backend': backend, 'variant': VARIANTS[profile], 'commit': sha}
    binary = None
    if backend == 'cuda':
        if profile != 'mamba2dmil_source':
            raise ValueError('2DMamba CPU source repairs cannot be silently applied to the CUDA kernel')
        if not torch.cuda.is_available():
            raise RuntimeError('2DMamba native CUDA requires an accessible GPU; no implicit torch substitution')
        if not cuda_binary or not Path(cuda_binary).is_file() or Path(cuda_binary).suffix != '.so':
            raise ValueError('2DMamba CUDA requires the explicitly built original pscan.so')
        binary = Path(cuda_binary).resolve()
        proof_path = binary.with_suffix('.so.json')
        if not proof_path.is_file():
            raise ValueError('2DMamba CUDA binary requires adjacent pscan.so.json build provenance')
        proof = read_json(proof_path)
        if (proof.get('source_commit') != sha or proof.get('sha256') != digest_file(binary)
                or proof.get('torch') != torch.__version__ or proof.get('cuda') != torch.version.cuda):
            raise ValueError('2DMamba CUDA build source, binary hash or Torch/CUDA ABI differs')
        identity.update(binary_sha256=digest_file(binary), build_metadata_sha256=digest_file(proof_path),
                        torch=torch.__version__, cuda=torch.version.cuda)
    elif cuda_binary is not None:
        raise ValueError('A CUDA binary is not used by an explicit torch scan')
    suffix = identity.get('binary_sha256', '')[:16]
    prefix = '_milplus_mamba2d_'+sha+'_'+VARIANTS[profile]+'_'+backend+suffix
    if prefix+'.head' in sys.modules:
        return sys.modules[prefix+'.head'].MambaMIL_2D, identity
    package(prefix, root/'2DMambaMIL/models')

    def load(name, filename, replacements=()):
        code = pinned_text(root, sha, '2DMambaMIL/models/'+filename)
        for before, after in replacements:
            if before not in code:
                raise ValueError('Pinned 2DMamba import compatibility contract changed')
            code = code.replace(before, after)
        module = types.ModuleType(prefix+'.'+name)
        module.__file__ = str(root/'2DMambaMIL/models'/filename)
        module.__package__ = prefix
        sys.modules[module.__name__] = module
        exec(compile(code, module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- verified Git source with explicit import-only changes
        return module

    scan = load('scan', 'pscan.py')
    if profile != 'mamba2dmil_source':
        factory = build_paper_scan if profile == 'mamba2dmil_paper' else build_scan
        scan.pscan_2d, _ = factory(pinned_text(root, sha, '2DMambaMIL/models/pscan.py'))
    if binary is not None:
        spec = importlib.util.spec_from_file_location(prefix+'.pscan', binary)
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)
        if not all(callable(getattr(kernel, name, None)) for name in ['fwd', 'bwd']):
            raise ValueError('Original 2DMamba pscan binary must export fwd/bwd')
        sys.modules[prefix+'.pscan'] = kernel
        ops = load('cuda_ops', 'pscan_2d.py', [('from models import pscan_cuda',
                                               'from '+prefix+' import pscan as pscan_cuda')])
        sys.modules[prefix+'.cuda_ops'] = ops
    load('mamba', 'mamba_simple.py', [
        ('from models.pscan import pscan, pscan_2d', 'from '+prefix+'.scan import pscan, pscan_2d'),
        ('from models.pscan_2d import selective_scan_fn as selective_scan_fn',
         'from '+prefix+'.cuda_ops import selective_scan_fn as selective_scan_fn')])
    head = load('head', 'MambaMIL_2D.py', [('from models.mamba_simple import',
                                         'from '+prefix+'.mamba import')])
    return head.MambaMIL_2D, identity


def original_optimizer(model, lr, weight_decay):
    optimizer = load_definitions('mamba2dmil', '2DMambaMIL/utils/utils.py', ['get_optim'],
                                 {'optim': torch.optim}).get_optim
    return optimizer(model, types.SimpleNamespace(opt='adamw', lr=lr, reg=weight_decay))


def original_nll():
    return load_definitions('mamba2dmil', '2DMambaMIL/utils/survival_utils.py',
                            ['nll_loss'], {'torch': torch}).nll_loss
