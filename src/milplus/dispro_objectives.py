"""Full DisPro survival/distillation objectives and outcome-free inference APIs.

These helpers retain the original UniPro and MultiPro networks; they do not
implement a replacement aggregator. The complete three-stage trainer is pending.
"""
import torch

from .dispro_source import survival_loss


def training_target(event, interval, device):
    e = torch.as_tensor(event,device=device)
    y = torch.as_tensor(interval,device=device)
    if (e.numel()!=1 or y.numel()!=1 or not torch.isfinite(e).all() or not torch.isfinite(y).all()
            or e.item() not in {0,1} or y.item()!=int(y.item()) or not 0<=int(y.item())<4):
        raise ValueError('DisPro training uses one observed-event0/1 and one genuine TRAIN-fitted interval0..3')
    return 1-e.reshape(1).float(),y.reshape(1).long()


def check_hazards(hazards,survival):
    if (hazards.shape!=(1,4) or survival.shape!=(1,4) or not torch.isfinite(hazards).all()
            or not torch.isfinite(survival).all() or (hazards<0).any() or (hazards>1).any()):
        raise ValueError('DisPro requires four finite hazard/survival bins')
    if not torch.allclose(survival,(1-hazards).cumprod(1),rtol=1e-6,atol=1e-7):
        raise ValueError('DisPro survival must be the original inclusive product of (1-hazard)')


def unimodal_objective(model,features,event,interval):
    device=next(model.parameters()).device
    c,y=training_target(event,interval,device)
    logits=model(features,c)
    if logits.shape!=(1,4) or not torch.isfinite(logits).all():
        raise ValueError('DisPro UniPro returned invalid original four-bin logits')
    hazards=logits.sigmoid();survival=(1-hazards).cumprod(1)
    loss=survival_loss()(hazards=hazards,S=survival,Y=y,c=c)
    return {'loss':loss,'hazards':hazards,'survival':survival,'risk':-survival.sum(1)}


def multimodal_objective(model,inputs,event,interval):
    if set(inputs)!={'x_WSI','x_Omics'} or inputs['x_WSI'] is None and inputs['x_Omics'] is None:
        raise ValueError('DisPro needs explicit observed inputs and at least one actual modality')
    device=next(model.parameters()).device;c,y=training_target(event,interval,device)
    result=model(**inputs,censor=c,label=y)
    check_hazards(result['hazards'],result['S'])
    criterion=survival_loss()
    fused=criterion(hazards=result['hazards'],S=result['S'],Y=y,c=c)
    loss=fused;terms={'fused_survival':fused}
    for modal in ['wsi','omics']:
        if inputs['x_WSI' if modal=='wsi' else 'x_Omics'] is None:
            h,s=result['hazards_'+modal],result['S_'+modal]
            check_hazards(h,s)
            term=criterion(hazards=h,S=s,Y=y,c=c)
            terms['missing_'+modal+'_unipro_distillation']=term
            loss=loss+term
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full DisPro joint objective')
    return {'loss':loss,**terms,'hazards':result['hazards'],'survival':result['S'],
            'risk':-result['S'].sum(1),'outputs':result}


@torch.no_grad()
def predict_unimodal(model,features):
    """Use fixed uncensored risk prompts: a declared outcome-free selection port.

    The public validation code feeds true censorship to choose prompt suffixes.
    That output-conditioned rule is not an eligible independent prediction API.
    This reference uses the four death-risk descriptions for all patients.
    """
    model.eval();device=next(model.parameters()).device
    logits=model(features,torch.zeros(1,device=device))
    if logits.shape!=(1,4) or not torch.isfinite(logits).all():
        raise ValueError('DisPro UniPro inference needs finite outcome-independent four-bin logits')
    hazards=logits.sigmoid();survival=(1-hazards).cumprod(1)
    return {'hazards':hazards,'survival':survival,'risk':-survival.sum(1)}


@torch.no_grad()
def predict_multimodal(model,inputs):
    """Fused inference uses original all-eight-prototype token scoring, label=None.

    A constant censorship argument satisfies the author's raw auxiliary-head
    interface; no patient's event/time/bin is accepted by this function.
    """
    if set(inputs)!={'x_WSI','x_Omics'} or inputs['x_WSI'] is None and inputs['x_Omics'] is None:
        raise ValueError('DisPro prediction requires at least one genuinely observed modality')
    model.eval();device=next(model.parameters()).device
    result=model(**inputs,censor=torch.zeros(1,device=device),label=None)
    check_hazards(result['hazards'],result['S'])
    return {'hazards':result['hazards'],'survival':result['S'],'risk':-result['S'].sum(1)}
