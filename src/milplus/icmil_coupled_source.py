"""Original MICCAI2023 ICMIL encoder, MIL heads and three-term distillation.

The separate confidence_based branch is a different method version and is not
implicitly substituted. These components alone do not register a full method.
"""
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .io import digest_file
from .upstream import load_definitions,verified_source

COMMIT='c2fd2faa4e85cf85527e228573b434a8cc461cba'
IMAGENET_SHA256='11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'


@lru_cache(maxsize=1)
def definitions():
    _,sha=verified_source('icmil_coupled')
    if sha!=COMMIT:raise ValueError('ICMIL coupled requires the fixed MICCAI2023 source, not in-context ICMIL or confidence_based')
    ns={'torch':torch,'nn':nn,'F':F}
    network=load_definitions('icmil_coupled','model/network.py',
        ['Classifier_1fc','residual_block','DimReduction'],ns)
    attention=load_definitions('icmil_coupled','model/Attention.py',
        ['Attention2','Attention_Gated','Attention_with_Classifier'],{**ns,'Classifier_1fc':network.Classifier_1fc})
    encoder=load_definitions('icmil_coupled','model/feature_extraction.py',
        ['Bottleneck_Baseline','ResNet_Baseline'],ns)
    return network,attention,encoder


def backbone(weights):
    if digest_file(weights)!=IMAGENET_SHA256:
        raise ValueError('Actual original ImageNet resnet50-11ad3fa6 weights are required; no random or different encoder')
    _,_,source=definitions()
    model=source.ResNet_Baseline(source.Bottleneck_Baseline,[3,4,6,3])
    state=torch.load(weights,map_location='cpu',weights_only=True)
    result=model.load_state_dict(state,strict=False)
    expected={k for k in state if k.startswith('layer4.') or k.startswith('fc.')}
    if result.missing_keys or set(result.unexpected_keys)!=expected:
        raise ValueError('Original truncated ResNet50/1024 initialization changed')
    return model


class BagModel(nn.Module):
    def __init__(self,kind):
        super().__init__()
        if kind not in {'abmil','maxpool','dtfd'}:raise ValueError('Use an actual released ICMIL MIL branch')
        network,attn,_=definitions();self.kind=kind
        # Preserve the literal constructor and parameter traversal order.
        self.classifier=network.Classifier_1fc(512,2,0.)
        from .icmil_coupled_pool import SourceMaxPool
        self.attention=SourceMaxPool() if kind=='maxpool' else attn.Attention_Gated(512)
        self.dim_reduction=network.DimReduction(1024,512,numLayer_Res=0)
        if kind=='dtfd':self.att_classifier=attn.Attention_with_Classifier(L=512,num_cls=2,droprate=0.)

    def first_tier(self,x):
        if x.ndim!=2 or x.shape[1]!=1024 or not len(x):raise ValueError('ICMIL original bag features are nonempty1024D')
        z=self.dim_reduction(x.unsqueeze(0)).squeeze(0)
        if self.kind=='maxpool':return self.attention(z.t()).squeeze(0).t()
        a=self.attention(z).squeeze(0)
        return torch.einsum('ns,n->ns',z,a).sum(0).unsqueeze(0)

    def forward(self,x):
        if self.kind!='dtfd':return self.classifier(self.first_tier(x))
        return self.att_classifier(torch.cat([self.first_tier(c) for c in torch.chunk(x,5,dim=0)],dim=0))

    def optimizers(self):
        from .vendor.torch14_adam import Adam as Adam14
        optimizer=Adam14 if self.kind=='dtfd' else torch.optim.Adam
        lr=2e-4 if self.kind=='abmil' else 1e-4
        groups=list(self.classifier.parameters())+list(self.attention.parameters())+list(self.dim_reduction.parameters())
        optimizers=[optimizer(groups,lr=lr,weight_decay=1e-4)]
        if self.kind=='dtfd':optimizers.append(optimizer(self.att_classifier.parameters(),lr=lr,weight_decay=1e-4))
        return optimizers

    def clip_first(self):
        for block in [self.dim_reduction,self.attention,self.classifier]:
            torch.nn.utils.clip_grad_norm_(block.parameters(),5.)


def bag_step(model,x,target,optimizers,legacy_adam14=False):
    if target.shape!=(1,) or target.dtype!=torch.long:raise ValueError('ICMIL bag training requires one native binary bag')
    if model.kind!='dtfd':
        loss=F.cross_entropy(model(x),target,reduction='none').mean()
        optimizers[0].zero_grad();loss.backward();model.clip_first();optimizers[0].step()
        return {'bag_ce':float(loss.detach()),'loss':float(loss.detach()),'optimizer_updates':1}
    from .vendor.torch14_adam import Adam as Adam14
    if not legacy_adam14 or any(not isinstance(o,Adam14) for o in optimizers):
        raise ValueError('Original DTFD interleaves optimizer mutation and second backward; use the actual Torch1.4 Adam semantics')
    # Execute the original update ordering. Original Adam14 mutates p.data,
    # exactly as in the author-required runtime; no version-check suppression,
    # delayed update, detached pseudo-bag or saved-tensor hook is used.
    features=[];predictions=[];labels=[]
    for chunk in torch.chunk(x,5,dim=0):
        z=model.first_tier(chunk);features.append(z);predictions.append(model.classifier(z));labels.append(target)
    pseudo=torch.cat(features);loss0=F.cross_entropy(torch.cat(predictions),torch.cat(labels),reduction='none').mean()
    optimizers[0].zero_grad();loss0.backward(retain_graph=True);model.clip_first();optimizers[0].step()
    loss1=F.cross_entropy(model.att_classifier(pseudo),target,reduction='none').mean()
    optimizers[1].zero_grad();loss1.backward();torch.nn.utils.clip_grad_norm_(model.att_classifier.parameters(),5.);optimizers[1].step()
    return {'bag_ce':float(loss0.detach()),'second_tier_ce':float(loss1.detach()),
        'loss':float((loss0+loss1).detach()),'optimizer_updates':2}


class Distiller(nn.Module):
    def __init__(self,weights,encoder_state,bag):
        super().__init__()
        self.kind=bag.kind
        network,_,_=definitions()
        self.teacher_encoder=backbone(weights);self.teacher_encoder.load_state_dict(encoder_state,strict=True)
        self.student_encoder=backbone(weights);self.student_encoder.load_state_dict(encoder_state,strict=True)
        self.teacher_encoder.eval()
        self.teacher_classifier=network.Classifier_1fc(512,2,0.)
        self.teacher_reduction=network.DimReduction(1024,512,numLayer_Res=0)
        head=bag.att_classifier.classifier if bag.kind=='dtfd' else bag.classifier
        self.teacher_classifier.load_state_dict(head.state_dict(),strict=True)
        self.teacher_reduction.load_state_dict(bag.dim_reduction.state_dict(),strict=True)
        for block in [self.teacher_encoder,self.teacher_classifier,self.teacher_reduction]:
            for p in block.parameters():p.requires_grad=False
        self.student_classifier=network.Classifier_1fc(512,2,0.)
        self.student_reduction=network.DimReduction(1024,512,numLayer_Res=0)
        self.student_classifier.load_state_dict(head.state_dict(),strict=True)
        self.student_reduction.load_state_dict(bag.dim_reduction.state_dict(),strict=True)

    def optimizer(self):
        from .vendor.torch14_adam import Adam as Adam14
        optimizer=Adam14 if self.kind=='dtfd' else torch.optim.Adam
        return optimizer(list(self.student_encoder.parameters())+list(self.student_reduction.parameters())+
            list(self.student_classifier.parameters()),lr=1e-5,weight_decay=1e-4)

    def objective(self,original,view1):
        with torch.no_grad():
            teacher_feature=self.teacher_encoder(original)
            teacher_z=self.teacher_reduction(teacher_feature)
            teacher_logits=self.teacher_classifier(teacher_z)
        student_z=self.student_reduction(self.student_encoder(view1));student_logits=self.student_classifier(student_z)
        consistency_z=self.student_reduction(teacher_feature);consistency_logits=self.student_classifier(consistency_z)
        kl=nn.KLDivLoss(reduction='none')
        loss0=kl(F.log_softmax(student_logits,dim=-1),F.softmax(teacher_logits.detach(),dim=-1)).mean()
        loss1=kl(F.log_softmax(consistency_logits,dim=-1),F.softmax(teacher_logits,dim=-1)).mean()
        loss2=kl(F.log_softmax(consistency_z,dim=-1),F.softmax(teacher_z,dim=-1)).mean()
        return {'loss':loss0+.5*loss1+.5*loss2,'patch_prediction_kl':loss0,
            'classifier_consistency_kl':loss1,'feature_consistency_kl':loss2}

    def step(self,original,view1,optimizer):
        values=self.objective(original,view1);optimizer.zero_grad();values['loss'].backward()
        for block in [self.student_reduction,self.student_classifier]:nn.utils.clip_grad_norm_(block.parameters(),5.)
        optimizer.step();return {k:float(v.detach()) for k,v in values.items()}


@lru_cache(maxsize=1)
def source_views():
    import glob,random,PIL.Image
    import torchvision.transforms as T
    source=load_definitions('icmil_coupled','dataset/RandomPatchDistillationDataset.py',
        ['RandomPatchDistillationDataset'],{'torch':torch,'glob':glob,'random':random,'PIL':PIL,'T':T})
    # Exact original defaults: ToTensor and BOTH independently randomized
    # flip/flip/RandomResizedCrop256/ToTensor views. The third is deliberately
    # generated even though the source loss never consumes its pixels.
    return source.RandomPatchDistillationDataset.__init__.__defaults__[-3:]


def views(image):
    if not isinstance(image,np.ndarray) or image.shape!=(256,256,3) or image.dtype!=np.uint8:
        raise ValueError('ICMIL original images are RGB uint8 256x256; no silent resizing/normalization')
    from PIL import Image
    original=Image.fromarray(image)
    return tuple(transform(original) for transform in source_views())
