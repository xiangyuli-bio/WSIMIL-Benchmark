"""Import pinned author modules in a private namespace without republishing their code."""
import ast
import os
import subprocess
import sys
import types
from pathlib import Path

from .registry import resource


def source_root(source):
    base = Path(os.environ.get("MILPLUS_SOURCE_ROOT", Path.home() / ".cache/milplus/sources"))
    return base / source


def verified_source(source):
    path = source_root(source)
    item = resource("sources.lock.json")[source]
    if not path.is_dir():
        raise FileNotFoundError(f"{source} author code missing. Run `milplus setup --method {source}` on a networked CPU host")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    # Status must still inspect every tracked change, but a read-only verifier
    # must not refresh/write the author's index on a full or read-only volume.
    changes = subprocess.check_output(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True)
    if sha != item["commit"] or changes:
        raise ValueError(f"{source}: author source version mismatch or tracked local modifications")
    return path, sha


def package(name, path):
    obj = types.ModuleType(name)
    obj.__path__ = [str(path)]
    obj.__package__ = name
    sys.modules[name] = obj
    return obj


def pinned_text(root, sha, relative_path):
    # Read the tracked Git object, so an untracked replacement cannot become executable code.
    return subprocess.check_output(["git", "show", f"{sha}:{relative_path}"], cwd=root, text=True)


def load_rrt():
    root, sha = verified_source("rrt")
    name = f"_milplus_rrt_{sha}"
    if name + ".rrt" in sys.modules:
        return sys.modules[name + ".rrt"]
    package(name, root / "modules")
    for filename in ("nystrom_attention", "emb_position", "datten", "rmsa", "rrt"):
        path = root / "modules" / (filename + ".py")
        text = pinned_text(root, sha, f"modules/{filename}.py")
        text = text.replace("from modules.", "from .")
        text = text.replace("from timm.models.layers import DropPath", "from milplus.layers import DropPath")
        text = text.replace("from timm.models.layers import trunc_normal_", "from torch.nn.init import trunc_normal_")
        # The author class exposes mlp_dim but hardcodes its input projection to 512.
        text = text.replace("self.patch_to_emb = [nn.Linear(input_dim, 512)]",
                            "self.patch_to_emb = [nn.Linear(input_dim, mlp_dim)]")
        module = types.ModuleType(name + "." + filename)
        module.__file__, module.__package__ = str(path), name
        sys.modules[module.__name__] = module
        exec(compile(text, str(path), "exec"), module.__dict__)  # noqa: S102 -- pinned author source with explicit import rewrites
    return sys.modules[name + ".rrt"]


def load_single(source, relative_path):
    root, sha = verified_source(source)
    suffix = relative_path.replace('/', '_').replace('.', '_')
    name = f"_milplus_{source}_{sha}_{suffix}"
    if name not in sys.modules:
        module = types.ModuleType(name)
        path = root / relative_path
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(pinned_text(root, sha, relative_path), str(path), "exec"), module.__dict__)  # noqa: S102 -- import pinned Git blob
    return sys.modules[name]


def load_definitions(source, relative_path, names, namespace, replacements=()):
    """Load named author definitions without unrelated training imports or import-time side effects."""
    root, sha = verified_source(source)
    code = pinned_text(root, sha, relative_path)
    for old, new in replacements:
        if old not in code:
            raise ValueError(f'{source}: compatibility patch no longer matches {old[:80]}')
        code = code.replace(old, new)
    tree = ast.parse(code)
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f'{source}: requested author definitions missing')
    name = f'_milplus_definitions_{source}_{sha}_{relative_path.replace("/", "_").replace(".", "_")}'
    module = types.ModuleType(name)
    module.__file__ = str(root / relative_path)
    module.__dict__.update(namespace)
    sys.modules[name] = module
    exec(compile(ast.Module(body=selected, type_ignores=[]), module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- named definitions from verified Git blob
    return module


def load_dgr():
    root, sha = verified_source("dgrmil")
    name = f"_milplus_dgrmil_{sha}"
    if name + ".dgrmil" in sys.modules:
        return sys.modules[name + ".dgrmil"]
    parent = package(name, root / "models")
    for filename in ("utils", "incremental_decode_utils", "quant_noise", "nystrom_attention", "linearatt", "dgrmil"):
        path = root / "models" / (filename + ".py")
        source = pinned_text(root, sha, f"models/{filename}.py").replace("from models.multiheadatt import MultiheadLinearAttention", "")
        source = source.replace("from models.", "from .").replace("from models import utils", "from . import utils")
        module = types.ModuleType(name + "." + filename)
        module.__file__, module.__package__ = str(path), name
        sys.modules[module.__name__] = module
        setattr(parent, filename, module)
        exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 -- pinned author source with isolated imports
    return sys.modules[name + ".dgrmil"]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Pinned source compatibility patch no longer matches: {old[:80]}')
    return text.replace(old, new, 1)


def load_survival(source):
    """Author architecture, with explicit input/device/output compatibility changes only."""
    root, sha = verified_source(source)
    name = f'_milplus_survival_{source}_{sha}'
    if name + '.model' in sys.modules:
        return sys.modules[name + '.model']
    package(name, root / 'models')
    if source == 'survpath':
        package(name + '.layers', root / 'models/layers')
        modules = [('layers.cross_attention', 'models/layers/cross_attention.py'),
                   ('model', 'models/model_SurvPath.py')]
    elif source == 'porpoise':
        modules = [('model', 'models/model_porpoise.py')]
    else:
        modules = [('model_utils', 'models/model_utils.py'),
                   ('model', 'models/model_coattn.py' if source == 'mcat' else 'models/model_graph_mil.py')]
    for module_name, relative in modules:
        code = pinned_text(root, sha, relative).replace('from models.', 'from ' + name + '.')
        for vector in ('o1', 'o2'):
            code = code.replace(f'torch.cuda.FloatTensor({vector}.shape[0], 1).fill_(1)',
                                f'{vector}.new_ones(({vector}.shape[0], 1))')
        if source == 'mcat' and module_name == 'model':
            code = replace_once(code, "fusion='concat', omic_sizes=", "path_input_dim=1024, fusion='concat', omic_sizes=")
            code = replace_once(code, '"small": [1024, 256, 256]', '"small": [path_input_dim, 256, 256]')
            code = replace_once(code, 'from torch.nn.modules.linear import _LinearWithBias',
                                'from torch.nn.modules.linear import NonDynamicallyQuantizableLinear as _LinearWithBias')
            code = replace_once(code, 'return hazards, S, Y_hat, attention_scores',
                                'return logits, hazards, S, Y_hat, attention_scores')
        if source == 'patchgcn' and module_name == 'model':
            code = replace_once(code, 'nn.Linear(1024, 128)', 'nn.Linear(num_features, hidden_dim)')
            code = code.replace('hidden_dim*4', 'hidden_dim*num_layers')
            # Both graph models share this return, but only PatchGCN_Surv is admitted here.
            code = code.replace('return hazards, S, Y_hat, A_path, None', 'return logits, hazards, S, Y_hat, A_path, None')
        module = types.ModuleType(name + '.' + module_name)
        module.__file__ = str(root / relative)
        module.__package__ = name + ('.layers' if module_name.startswith('layers.') else '')
        sys.modules[module.__name__] = module
        exec(compile(code, module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- verified pinned author Git blobs
    return sys.modules[name + '.model']


def load_crossmodal(source):
    """Preserve pinned CMTA/MOTCat mechanisms with audited device, dimension and POT compatibility."""
    if source not in {'cmta', 'motcat'}:
        raise ValueError(f'Unknown cross-modal author source: {source}')
    root, sha = verified_source(source)
    name = f'_milplus_crossmodal_{source}_{sha}'
    if name + '.model' in sys.modules:
        return sys.modules[name + '.model']
    package(name, root / 'models')
    modules = ([('util', 'models/cmta/util.py'), ('model', 'models/cmta/network.py')]
               if source == 'cmta' else
               [('model_utils', 'models/model_utils.py'), ('model', 'models/model_motcat.py')])
    for module_name, relative in modules:
        code = pinned_text(root, sha, relative).replace('from models.', 'from ' + name + '.')
        for vector in ('o1', 'o2'):
            code = code.replace(f'torch.cuda.FloatTensor({vector}.shape[0], 1).fill_(1)',
                                f'{vector}.new_ones(({vector}.shape[0], 1))')
        if module_name == 'model' and source == 'cmta':
            code = replace_once(code, 'def __init__(self, omic_sizes=', 'def __init__(self, path_input_dim=1024, omic_sizes=')
            code = replace_once(code, '"small": [1024, 256, 256]', '"small": [path_input_dim, 256, 256]')
            code = replace_once(code, 'self.cls_token.expand(B, -1, -1).cuda()', 'self.cls_token.expand(B, -1, -1)')
            code = replace_once(code, 'self.cls_token.expand(features.shape[0], -1, -1).cuda()',
                                'self.cls_token.expand(features.shape[0], -1, -1)')
            code = replace_once(code, 'return hazards, S, cls_token_pathomics_encoder',
                                'return logits, hazards, S, cls_token_pathomics_encoder')
        if module_name == 'model' and source == 'motcat':
            code = replace_once(code, "def __init__(self, fusion='concat',", "def __init__(self, path_input_dim=1024, fusion='concat',")
            code = replace_once(code, '"small": [1024, 256, 256]', '"small": [path_input_dim, 256, 256]')
            code = replace_once(code, 'flow.type(torch.FloatTensor).cuda()', 'flow.to(device=weight1.device, dtype=weight1.dtype)')
            code = replace_once(code, 'reg=self.ot_reg,reg_m=self.ot_tau)',
                                'reg=self.ot_reg,reg_m=self.ot_tau, reg_type="entropy")')
            code = code.replace('cost_map_detach.max()', 'cost_map_detach.max().clamp_min(torch.finfo(cost_map_detach.dtype).tiny)')
            for vector, prefix in [('weight1', 'src'), ('weight2', 'dst')]:
                code = replace_once(code, f'{prefix}_weight = {vector}.sum(dim=1) / {vector}.sum()',
                    f'{prefix}_weight = ({vector}.sum(dim=1) / {vector}.sum() if {vector}.sum() > 0 '
                    f'else {vector}.new_full(({vector}.shape[0],), 1 / {vector}.shape[0]))')
            code = replace_once(code, 'x = x.squeeze()', 'x = x.squeeze(1)')
            code = replace_once(code, 'y = y.squeeze()', 'y = y.squeeze(1)')
            code = replace_once(code, 'torch.mm(A_coattn.squeeze(), h_path_bag.squeeze())',
                                'torch.mm(A_coattn.squeeze(0).squeeze(0), h_path_bag.squeeze(1))')
            code = replace_once(code, 'return hazards, S, Y_hat, attention_scores',
                                'return logits, hazards, S, Y_hat, attention_scores')
        module = types.ModuleType(name + '.' + module_name)
        module.__file__, module.__package__ = str(root / relative), name
        sys.modules[module.__name__] = module
        try:
            exec(compile(code, module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- verified pinned author Git blobs
        except BaseException:
            del sys.modules[module.__name__]
            raise
    return sys.modules[name + '.model']
