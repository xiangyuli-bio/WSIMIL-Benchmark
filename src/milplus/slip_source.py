"""Private original SLIP mechanisms and exact pretrained CLIP text towers.

Development only: no training profile is admitted by this module's existence.
Image encoders are not executed for a frozen-feature MIL input. Every text weight
comes from the full original public checkpoint, retaining native dimensions.
"""
import contextlib
import gc
import io
import os
import types
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from .io import digest_file, read_json
from .upstream import load_definitions, load_single, pinned_text, verified_source

COMMIT = '9351c4480ec8d7a8216316a1861025b4f086ac78'
ASSETS = {
    'clip_vit_b16': ('ViT-B-16.pt', '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f', 512),
    'clip_rn50': ('RN50.pt', 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762', 1024),
}


def fixed():
    root, sha = verified_source('slip')
    if sha != COMMIT:
        raise ValueError('SLIP requires the reviewed original author commit')
    return root, sha


def definitions(path, names, namespace=None, replacements=()):
    fixed()
    return load_definitions('slip', path, names,
        {'torch': torch, 'nn': nn, 'F': F, 'OrderedDict': OrderedDict,
         'Tuple': Tuple, 'Union': Union, 'List': List, **(namespace or {})}, replacements)


@lru_cache(maxsize=1)
def tokenizer():
    root, sha = fixed()
    # The BPE reader opens a file path; require the bytes to match the Git blob,
    # including for an untracked replacement that a Git status cannot detect.
    import subprocess
    rel = 'clip/bpe_simple_vocab_16e6.txt.gz'
    data = subprocess.check_output(['git', 'show', sha+':'+rel], cwd=root)
    import hashlib
    if digest_file(root/rel) != hashlib.sha256(data).hexdigest():
        raise ValueError('SLIP BPE vocabulary differs from the fixed Git object')
    return load_single('slip', 'clip/simple_tokenizer.py').SimpleTokenizer()


@lru_cache(maxsize=1)
def tokenize_function():
    return definitions('clip/clip.py', ['tokenize'], {'_tokenizer': tokenizer()}).tokenize


@lru_cache(maxsize=2)
def text_backbone(architecture):
    if architecture not in ASSETS:
        raise ValueError('SLIP native CLIP branch must be clip_vit_b16 or clip_rn50')
    name, expected, dimension = ASSETS[architecture]
    root = Path(os.environ.get('MILPLUS_SLIP_CLIP_ASSETS',
        Path.home()/'.cache/milplus/assets/slip_clip'))
    meta = read_json(root/'manifest.json')
    if (meta.get('state') != 'passed_exact_original_CLIP_assets'
            or meta['files'].get(name, {}).get('sha256') != expected
            or digest_file(root/name) != expected):
        raise ValueError('SLIP requires the exact actual original CLIP checkpoint: '+name)
    blocks = definitions('clip/model.py',
        ['LayerNorm', 'QuickGELU', 'ResidualAttentionBlock', 'Transformer'])
    clip_class = definitions('clip/model.py', ['CLIP']).CLIP

    class NativeText(nn.Module):
        def __init__(self):
            super().__init__()
            mask = torch.empty(77, 77).fill_(float('-inf')).triu_(1)
            self.transformer = blocks.Transformer(512, 12, 8, attn_mask=mask)
            self.token_embedding = nn.Embedding(49408, 512)
            self.positional_embedding = nn.Parameter(torch.empty(77, 512))
            self.ln_final = blocks.LayerNorm(512)
            self.text_projection = nn.Parameter(torch.empty(512, dimension))
            self.logit_scale = nn.Parameter(torch.empty(()))
            # The prompt constructor checks input resolution only. This is
            # metadata, never a fabricated executable image encoder.
            self.visual = types.SimpleNamespace(input_resolution=224)

        @property
        def dtype(self):
            return self.token_embedding.weight.dtype

        def encode_text(self, tokens):
            return clip_class.encode_text(self, tokens.to(self.token_embedding.weight.device))

    with torch.random.fork_rng(devices=[]):
        model = NativeText()
    checkpoint = torch.jit.load(str(root/name), map_location='cpu').state_dict()
    allowed = set(model.state_dict())
    required = {k: v.float() for k, v in checkpoint.items() if k in allowed}
    if set(required) != allowed:
        raise ValueError('SLIP actual checkpoint is missing native text parameters')
    unexpected = {k for k in checkpoint if k not in allowed
                  and not k.startswith('visual.')
                  and k not in {'input_resolution', 'context_length', 'vocab_size'}}
    if unexpected:
        raise ValueError('SLIP actual CLIP checkpoint has unexpected nonvisual state')
    model.load_state_dict(required, strict=True)
    model.requires_grad_(False); model.eval()
    del checkpoint, required
    gc.collect()
    return model


@lru_cache(maxsize=1)
def source_pooling():
    # The loaded class is only used as a namespace for its actual methods.
    return definitions('methods/slidecoop.py', ['SLIP'], {'SlideCoOp': object}).SLIP


@lru_cache(maxsize=1)
def source_loss():
    # Preserve the original contrastive C*C objective. Ports only device and
    # shape handling, so singleton bags keep their N,D dimensions.
    return definitions('methods/slidecoop.py', ['SLIP'], {'SlideCoOp': object}, [
        ('wsi_feats = wsi_feats.cuda().squeeze()',
         'wsi_feats = wsi_feats.to(next(self.prompt_learner.parameters()).device).reshape(-1, self.dimension)'),
        ('wsi_label = wsi_label.cuda().squeeze()',
         'wsi_label = wsi_label.to(next(self.prompt_learner.parameters()).device).reshape(())'),
    ]).SLIP.compute_loss


class SLIPReference(nn.Module):
    """Actual author prompts, frozen native text, detached attention and loss."""
    def __init__(self, architecture, templates, slide_classnames, tissue_classnames,
                 context_size=1, context_gain=.01, temperature=.01):
        super().__init__()
        if (architecture not in ASSETS or not isinstance(templates,list) or not templates
                or any(not isinstance(s,str) or s.count('{}') != 1 for s in templates)
                or not isinstance(slide_classnames,list) or len(slide_classnames)<2
                or not isinstance(tissue_classnames,list) or not tissue_classnames
                or any(not isinstance(g,list) or not g or any(not isinstance(s,str) or not s.strip() for s in g)
                       for g in slide_classnames+tissue_classnames)
                or type(context_size) is not int or not 1<=context_size<=70
                or not 0<context_gain<100 or not 0<temperature<100):
            raise ValueError('Invalid ordered SLIP classes/tissues/templates/native prompt configuration')
        self.dimension = ASSETS[architecture][2]
        self.temperature = temperature
        self.native = text_backbone(architecture)
        tokenize = tokenize_function()
        token = tokenizer()
        prompt_module = definitions('networks/coop.py', ['PromptLearner','TextEncoder'],
            {'clip': types.SimpleNamespace(tokenize=tokenize), '_tokenizer': token},[
                ('prompt = clip.tokenize(ctx_init)',
                 'prompt = clip.tokenize(ctx_init).to(next(clip_model.parameters()).device)'),
                ('tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])',
                 'tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(next(clip_model.parameters()).device)'),
            ])
        formatted = [[templates[0].format(group[0])] for group in slide_classnames]
        with contextlib.redirect_stdout(io.StringIO()):
            self.prompt_learner = prompt_module.PromptLearner(formatted,self.native,
                context_size=context_size,imgsize=224,context_init=None,
                cls_token_position='end',context_gain=context_gain)
        tokens = self.prompt_learner.tokenized_prompts
        del self.prompt_learner.tokenized_prompts
        self.prompt_learner.register_buffer('tokenized_prompts',tokens)
        self.text_encoder = prompt_module.TextEncoder(self.native)
        self.register_buffer('tokenized_prompts',tokens.clone())
        weights=[]
        # Original zeroshot_classifier: normalize each description/template,
        # average within each tissue, renormalize and stack as columns.
        with torch.no_grad():
            for tissue in tissue_classnames:
                texts = [template.format(alias) for alias in tissue for template in templates]
                embeddings=F.normalize(self.native.encode_text(tokenize(texts)),dim=-1)
                weights.append(F.normalize(embeddings.mean(0),dim=0))
        self.register_buffer('tissue_weights',torch.stack(weights,dim=1))
        self.architecture = architecture

    def _apply(self,fn,recurse=True):
        result=super()._apply(fn,recurse)
        # Preserve the original shared frozen text Parameters even when the
        # caller enables PyTorch's overwrite-module-parameters conversion policy.
        for name in ['transformer','positional_embedding','ln_final','text_projection']:
            setattr(self.text_encoder,name,getattr(self.native,name))
        self.prompt_learner.tokenized_prompts=self.tokenized_prompts
        return result

    def forward(self, features, eval=False, return_metadata=False):
        if (features.ndim!=2 or features.shape[0]<1 or features.shape[1]!=self.dimension
                or not torch.isfinite(features).all()):
            raise ValueError('SLIP needs a complete finite N,D bag in its actual native CLIP feature space')
        # Both paths recompute current learned class text. This equals the
        # author's freshly cached eval slide_weights, without a stale cache.
        return source_pooling().forward(self,features,eval=False,return_metadata=return_metadata)

    def loss(self, features, label):
        label=torch.as_tensor(label,device=features.device)
        if label.numel()!=1 or not 0<=int(label)<self.tokenized_prompts.shape[0]:
            raise ValueError('SLIP requires one valid slide class target')
        return source_loss()(self,features,label)

    def protocol(self):
        return {'source_commit':COMMIT,'architecture':self.architecture,
            'actual_asset_sha256':ASSETS[self.architecture][1],
            'text_hidden':512,'text_layers':12,'text_heads':8,'text_context_length':77,
            'feature_dimension':self.dimension,'trainable':'Only original class-specific prompt context',
            'precision':'Native original text weights in float32, as the literal CLIP CPU loading branch; no GPU-half equality claim',
            'objective':'Original contrastive C*C class-correlation cross entropy; attention detached as source',
            'images':'Not loaded: input is frozen native CLIP patch embeddings',
            'ports':['Explicit CPU/device-safe prompt buffers','Singleton N,D preservation',
                     'Recompute current learned eval text instead of mutable prediction cache'],
            'state':'development_mechanism_only_not_admitted'}
