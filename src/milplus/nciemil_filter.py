"""Paper-defined full Swin filtering stage with explicitly reconstructed settings.

The author release omits this stage. Public Torchvision initialization, binary
projection, deterministic RGB256 normalization and user-declared pretraining
settings are a named reference reconstruction, never claimed as the missing
author training recipe. Every original patch participates in max selection.
"""
import copy
import math
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Bundle, resolve
from .io import atomic_torch_save, digest_file, digest_object, write_json

WEIGHTS_URL='https://download.pytorch.org/models/swin_t-704ceda3.pth'
WEIGHTS_SHA256='704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b'
IMPLEMENTATION='torchvision0201_swin_t_rgb256_maxinstance_reference_reconstruction'


def torch_rng():
    return {'cpu':torch.get_rng_state().clone(),
            'cuda':[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None}


def restore_torch_rng(value):
    torch.set_rng_state(value['cpu'].cpu())
    if value['cuda'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in value['cuda']])


class SwinFilter(nn.Module):
    def __init__(self,weights,expected_sha256):
        super().__init__()
        from torchvision import __version__ as vision_version
        from torchvision.models import swin_t
        if (vision_version.split('+')[0]!='0.20.1' or expected_sha256!=WEIGHTS_SHA256
                or digest_file(weights)!=expected_sha256):
            raise ValueError('NcIEMIL reference requires the exact public pretrained Torchvision0.20.1 Swin-Tiny weight bytes')
        self.encoder=swin_t(weights=None)
        self.encoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        if sum(p.numel() for p in self.encoder.parameters())!=28_288_354:
            raise ValueError('NcIEMIL requires complete native Swin-Tiny, not a reduced backbone')
        self.encoder.head=nn.Identity()
        self.projection=nn.Linear(768,1)
        self.weight_sha256=expected_sha256

    def forward(self,images):
        if (images.ndim!=4 or images.shape[1:]!=(3,256,256) or not len(images)
                or not images.is_floating_point() or not torch.isfinite(images).all()):
            raise ValueError('NcIEMIL filter needs complete normalized RGB256 patches')
        features=self.encoder(images)
        return features,self.projection(features).squeeze(-1)

    def protocol(self):
        return {'implementation':IMPLEMENTATION,'weights_url':WEIGHTS_URL,
            'initial_weights_sha256':self.weight_sha256,'feature_dimension':768,
            'patch_size':256,'normalization':'ImageNet mean/std; full RGB256, no crop or resize',
            'depths':[2,2,6,2],'heads':[3,6,12,24],'window':7,'stochastic_depth':.2,
            'head':'new learned Linear768to1; BCEWithLogits of maximum positive logit',
            'undisclosed_author_settings':'Exact filtering checkpoint, head architecture, optimizer, pretraining epochs and bag minibatch were not supplied; named choices are explicit reference reconstruction',
            'paper_reproduction':'not_run','real_wsi_validation':'not_run'}


def replay_max_instance_backward(model,read_chunk,count,target,device,normalization=1,patch_batch=128):
    """Exact max-instance derivative with unchanged stochastic full-chunk draws.

    Scan every patch in training mode without retaining activation graphs. Replay
    the winning original full chunk with its saved RNG state, so DropPath and
    dropout choose exactly the same functions. Retain post-scan RNG for the next
    bag. No eval-to-train switch, subset resampling or selected-only batch shape.
    """
    if (not model.training or type(count) is not int or count<1 or target not in {0,1}
            or isinstance(target,bool) or type(normalization) is not int or normalization<1
            or patch_batch!=128):
        raise ValueError('NcIEMIL max backward requires original complete TRAIN bag, binary target and patch batch128')
    if any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Exact replay requires Swin LayerNorm without mutable BatchNorm running statistics')
    if str(device).startswith('cuda'):torch.cuda.get_rng_state_all()
    winner=None;maximum=None;winner_rng=None;winner_values=None
    with torch.no_grad():
        for start in range(0,count,patch_batch):
            images=read_chunk(start,min(start+patch_batch,count)).to(device)
            if len(images)!=min(patch_batch,count-start):raise ValueError('Incomplete original RGB chunk')
            before=torch_rng();_,logits=model(images)
            if logits.shape!=(len(images),) or not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite or misaligned complete filter logits')
            value,index=logits.max(0);value=float(value)
            # Strict > preserves the first original patch when maxima tie.
            if maximum is None or value>maximum:
                maximum=value;winner=(start,int(index));winner_rng=before;winner_values=logits.detach().clone()
    after_scan=torch_rng();start,index=winner
    images=read_chunk(start,min(start+patch_batch,count)).to(device)
    restore_torch_rng(winner_rng)
    try:
        _,replayed=model(images)
        torch.testing.assert_close(replayed.detach(),winner_values,rtol=0,atol=0,
            msg='NcIEMIL stochastic full-chunk replay changed its actual scanned logits')
        loss=F.binary_cross_entropy_with_logits(replayed[index],replayed.new_tensor(float(target)))
        if not torch.isfinite(loss):raise FloatingPointError('NcIEMIL max-instance binary objective is nonfinite')
        (loss/normalization).backward()
    finally:
        restore_torch_rng(after_scan)
    return {'loss':float(loss.detach()),'winner':start+index,'patches_scanned':count,
            'replayed_chunk_start':start,'replayed_chunk_count':len(images),'maximum_logit':maximum}


class FilterInputs:
    """All raw RGB patch identities plus explicit bag weak-label mapping."""
    def __init__(self,path):
        self.bundle=Bundle(path);b=self.bundle;s=b.encoder.get('nciemil_raw')
        if (b.task_type!='classification' or b.dim!=3 or not isinstance(s,dict)
                or s.get('kind')!='nciemil_rgb256_uint8_hdf5' or s.get('schema_version')!=1
                or b.config.get('training_unit','slide')!='slide'
                or b.config.get('evaluation_unit','slide')!='slide'
                or not isinstance(s.get('image_provenance'),str) or not s['image_provenance'].strip()
                or not isinstance(s.get('patch_provenance'),str) or not s['patch_provenance'].strip()):
            raise ValueError('NcIEMIL needs its explicit complete slide RGB256 HDF5 input contract')
        mapping=s.get('weak_targets')
        if (not isinstance(mapping,dict) or set(mapping)!=set(b.classes)
                or any(type(v) is not int or v not in {0,1} for v in mapping.values())
                or set(mapping.values())!={0,1}):
            raise ValueError('NcIEMIL requires explicit positive/negative weak-target mapping for every task class')
        self.schema=copy.deepcopy(s);self.mapping=mapping;self.records={};self.rows=b.rows.to_dict('records')
        paths=[]
        for row in self.rows:
            p=resolve(b.root,row['path']).resolve()
            with h5py.File(p,'r') as f:
                x=f['pixels'];ids=f['patch_ids'].asstr()[:].tolist()
                if x.ndim!=4 or x.shape[1:]!=(256,256,3) or x.dtype!=np.dtype('uint8') or len(x)<1:
                    raise ValueError('NcIEMIL raw bags must retain full nonempty uint8 NHWC RGB256 patches')
                if len(ids)!=len(x) or len(ids)!=len(set(ids)) or any(not str(x).strip() for x in ids):
                    raise ValueError('NcIEMIL raw patch identities must be aligned, unique and nonempty')
            self.records[row['slide_id']]={'path':str(p),'sha256':digest_file(p),'patch_ids':ids,
                'patch_ids_sha256':digest_object(ids),'count':len(ids),'bytes':p.stat().st_size,
                'mtime_ns':p.stat().st_mtime_ns,'weak_target':mapping[row['label']]};paths.append(str(p))
        if len(paths)!=len(set(paths)):raise ValueError('NcIEMIL different slides cannot alias the same RGB bag')
        self.files={str(p):digest_file(p) for p in [b.root/'task.yaml',resolve(b.root,b.config.get('slides','slides.csv')),
            resolve(b.root,b.config['splits'][b.split]),resolve(b.root,b.encoder['index'])]}
        self.contract=digest_object({'bundle':b.fingerprint,'schema':self.schema,'rows':self.rows,'records':self.records})

    def check(self,deep=False):
        if self.bundle.rows.to_dict('records')!=self.rows:raise ValueError('NcIEMIL patient/slide/label/split identity changed')
        if any(digest_file(p)!=sha for p,sha in self.files.items()):raise ValueError('NcIEMIL original task/labels/splits/index changed')
        for item in self.records.values():
            p=Path(item['path'])
            if p.stat().st_size!=item['bytes'] or p.stat().st_mtime_ns!=item['mtime_ns'] or (deep and digest_file(p)!=item['sha256']):
                raise ValueError('NcIEMIL complete original RGB bytes changed')

    @contextmanager
    def open(self,row):
        self.check();item=self.records[row['slide_id']]
        with h5py.File(item['path'],'r') as f:
            def read(start,end):
                x=torch.from_numpy(f['pixels'][start:end].copy()).permute(0,3,1,2).float()/255
                mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1)
                std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
                return (x-mean)/std
            yield read


def train_filter(inputs,weights,weight_sha256,output,*,epochs,lr,weight_decay,bag_batch_size,seed=3407,
                 device='cpu',resume=False,stop_after=None):
    """Complete TRAIN-only weak Swin stage; missing paper parameters are explicit.

    Save the actual extractor, binary head, all optimizer moments, RNG, bag
    order and winners at epoch boundaries. Last epoch is the declared stage
    selection; there is no look at held-out labels or substituted pretrained FM.
    """
    from .engine import seed_everything,rng_state,restore_rng
    from .provenance import software_identity
    if (type(epochs) is not int or epochs<1 or type(bag_batch_size) is not int or bag_batch_size<1
            or type(seed) is not int or isinstance(lr,bool) or not math.isfinite(lr) or lr<=0
            or isinstance(weight_decay,bool) or not math.isfinite(weight_decay) or weight_decay<0
            or (stop_after is not None and (type(stop_after) is not int or stop_after<1))):
        raise ValueError('NcIEMIL filtering recipe requires explicit valid epochs, AdamW LR/decay and bag minibatch')
    inputs=inputs if isinstance(inputs,FilterInputs) else FilterInputs(inputs)
    out=Path(output);out.mkdir(parents=True,exist_ok=resume)
    cfg={'epochs':epochs,'lr':lr,'weight_decay':weight_decay,'bag_batch_size':bag_batch_size,
        'patch_batch':128,'optimizer':'AdamW','seed':seed,'selection':'declared_last_pretraining_epoch',
        'implementation':IMPLEMENTATION,'input_contract':inputs.contract,'initial_weights_sha256':weight_sha256,
        'software':software_identity()}
    seed_everything(seed);model=SwinFilter(weights,weight_sha256).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    history=[];updates=0;start=0;rows=inputs.bundle.records('train')
    if resume:
        state=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if state['config']!=cfg:raise ValueError('NcIEMIL filter recovery changed raw inputs, software or explicit recipe')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];updates=state['updates'];start=state['completed_epochs']
        if len(history)!=start or updates!=start*math.ceil(len(rows)/bag_batch_size):raise ValueError('Incomplete NcIEMIL filter checkpoint')
        restore_rng(state['rng'])
    else:write_json(out/'request.json',cfg)
    if start>=epochs:raise ValueError('NcIEMIL filter already completed its declared training budget')
    for epoch in range(start,epochs):
        inputs.check();model.train();order=torch.randperm(len(rows)).tolist();traces=[]
        for offset in range(0,len(order),bag_batch_size):
            batch=order[offset:offset+bag_batch_size];optimizer.zero_grad(set_to_none=True)
            for index in batch:
                row=rows[index];item=inputs.records[row['slide_id']]
                with inputs.open(row) as read:
                    trace=replay_max_instance_backward(model,read,item['count'],item['weak_target'],device,len(batch))
                traces.append({'slide_id':row['slide_id'],**trace})
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('NcIEMIL full Swin filtering gradient became nonfinite')
            optimizer.step();updates+=1
        history.append({'epoch':epoch,'order':[rows[i]['slide_id'] for i in order],'traces':traces})
        state={'config':cfg,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),
            'history':history,'updates':updates,'completed_epochs':epoch+1,'protocol':model.protocol()}
        atomic_torch_save(state,out/'last.pt')
        interrupted=stop_after is not None and epoch+1>=stop_after and epoch+1<epochs
        write_json(out/'status.json',{'state':'interrupted' if interrupted else 'completed' if epoch+1==epochs else 'running',
            'epochs':epoch+1,'updates':updates,'trained_slides':[r['slide_id'] for r in rows],
            'only_TRAIN_used_for_gradients':True,'formal_admission':False})
        if interrupted:return out
    return out
