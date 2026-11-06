"""Private verified-Git imports of the six original TTIS heads and optimizer."""
import sys
import types

from .upstream import package, pinned_text, verified_source

HEADS={'abmil':'ABMIL','clam':'CLAMSB','rrt':'RRTMIL','s4':'S4MIL','transmil':'TransMIL','wikg':'WIKG'}


def load_ttis(kind):
    if kind not in HEADS:
        raise ValueError('Unknown original TTIS backbone')
    root,sha=verified_source('ttis')
    prefix='_milplus_ttis_'+sha
    target=prefix+'.models.'+HEADS[kind]
    if target in sys.modules:
        return getattr(sys.modules[target],HEADS[kind])
    if prefix not in sys.modules:package(prefix,root/'src')
    if prefix+'.models' not in sys.modules:package(prefix+'.models',root/'src/models')  # Skip unrelated Lightning trainer import in __init__.
    if prefix+'.models.modules' not in sys.modules:package(prefix+'.models.modules',root/'src/models/modules')
    if prefix+'.ttis' not in sys.modules:package(prefix+'.ttis',root/'src/ttis')
    def module(name,path):
        fullname=prefix+'.'+name
        if fullname in sys.modules:return sys.modules[fullname]
        code=pinned_text(root,sha,path).replace('from src.', 'from '+prefix+'.')
        if name=='models.S4MIL':
            code=code.replace('import opt_einsum as oe\n','')  # No oe references exist; unused dependency only.
        if name=='models.RRTMIL':
            code=code.replace('from timm.models.layers import DropPath','from milplus.layers import DropPath')
        value=types.ModuleType(fullname);value.__file__=str(root/path);value.__package__=fullname.rpartition('.')[0]
        sys.modules[fullname]=value;exec(compile(code,value.__file__,'exec'),value.__dict__)  # noqa: S102 -- verified pinned Git definitions with explicit import-only compatibility rewrites
        return value
    for name in ['nystrom_attention','layers','mil_template']:
        module('models.modules.'+name,'src/models/modules/'+name+'.py')
    core=sys.modules[prefix+'.models.modules']
    for cls_name,file_name in [('NystromAttention','nystrom_attention'),('MIL','mil_template'),
                                ('create_mlp','layers'),('GlobalAttention','layers'),('GlobalGatedAttention','layers')]:
        setattr(core,cls_name,getattr(sys.modules[prefix+'.models.modules.'+file_name],cls_name))
    sampler=module('ttis.sampling','src/ttis/sampling.py')
    sys.modules[prefix+'.ttis'].Sampling=sampler.Sampling
    module('ttis.aggregation','src/ttis/aggregation.py')
    result=module('models.'+HEADS[kind],'src/models/'+HEADS[kind]+'.py')
    return getattr(result,HEADS[kind])


def load_ttis_optimizer():
    root,sha=verified_source('ttis');prefix='_milplus_ttis_'+sha
    if prefix not in sys.modules:package(prefix,root/'src')
    name=prefix+'.optimizers'
    if name+'.optim_factory' in sys.modules:return sys.modules[name+'.optim_factory'].create_optimizer
    package(name,root/'src/optimizers')
    for file in ['lookahead','optim_factory']:
        fullname=name+'.'+file
        code=pinned_text(root,sha,'src/optimizers/'+file+'.py').replace('from src.', 'from '+prefix+'.')
        value=types.ModuleType(fullname);value.__file__=str(root/'src/optimizers'/(file+'.py'));value.__package__=name
        sys.modules[fullname]=value;exec(compile(code,value.__file__,'exec'),value.__dict__)  # noqa: S102 -- verified pinned Git definitions with explicit import-only compatibility rewrites
        if file=='lookahead':sys.modules[name].Lookahead=value.Lookahead
    return value.create_optimizer
