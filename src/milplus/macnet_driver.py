"""Complete raw Swin/GFK/adaptive-cluster MacNet, source reset-RMSprop recipe."""
from pathlib import Path
import importlib.metadata,math
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from . import macnet_source as source
from .macnet_data import CONDITIONS
from .data import Bundle
from .io import read_json,write_json,digest_file,digest_object
from .provenance import software_identity
from .icmil_coupled_driver import _lock
from .icmil_coupled_training import save,load,exists,checkpoint_path,Interrupted
PROFILES={f'macnet_{condition}_{adaptive}_patient_reference':{'condition':condition,'adaptive':adaptive,'native_epochs':100}
    for condition in CONDITIONS for adaptive in ['source_cached','refresh_repaired']}


class RawBags(Dataset):
    def __init__(self,assets,subset):self.assets=assets;self.rows=assets.bundle.records(subset);self.order=[]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];self.order.append(r['slide_id']);return self.assets.tensor(r),r['target']


def runtime():
    versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','timm','einops','Pillow','numpy']}
    if versions['torch'].split('+')[0]!='2.5.1' or versions['torchvision'].split('+')[0]!='0.20.1' or versions['timm']!='0.9.12':raise ValueError('Use declared isolated MacNet Torch2.5.1/torchvision0.20.1/timm0.9.12 source-reference runtime')
    return {'versions':versions,'precision':'source fp32 unless explicit AMP','author_environment_not_pinned':True}


def device(value):
    d=torch.device(value)
    if d.type!='cuda' or not torch.cuda.is_available():raise ValueError('Complete original MacNet needs actual CUDA; no frozen-feature/CPU substitute')
    torch.cuda.set_device(d);return torch.device('cuda',torch.cuda.current_device())


def recipe(assets,method,config):
    if method not in PROFILES or PROFILES[method]['condition']!=assets.condition:raise ValueError('Choose actual prepared public5/LSCC81/CSCC64 MacNet profile')
    if set(config)-{'epochs','threads','use_amp','lr_policy'}:raise ValueError('Keep full original Swin768, true raw/guidance bags, GFK, clustering, native losses and RMSprop')
    cfg={**PROFILES[method],'epochs':100,'threads':2,'use_amp':False,'lr_policy':'source_mutated_group',
        'batch_size':CONDITIONS[assets.condition]['batch_size'],'max_input_len':2000,'selection':'best_source_validation_batch_accuracy_first_valid',
        'original_requested_lr_schedule':[1e-5,5e-6,1e-6],'optimizer_reset_every_epoch':True}
    cfg.update(config)
    if any(type(cfg[k]) is not int or cfg[k]<1 for k in ['epochs','threads']) or type(cfg['use_amp']) is not bool or cfg['lr_policy'] not in {'source_mutated_group','intended_schedule_repaired'}:raise ValueError('Declare actual MacNet budgets and source or explicitly repaired LR policy')
    return cfg


def _cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def _gradients(model):return {k:None if p.grad is None else p.grad.detach().cpu().clone() for k,p in model.named_parameters()}
def _groups(model):return [{'params':list(model.feature.parameters())},{'params':list(model.head.parameters())}]


def _restore_model(model,state,dev):
    model.load_state_dict(state['model'],strict=True)
    for k,p in model.named_parameters():p.grad=None if state['gradients'][k] is None else state['gradients'][k].to(dev).clone()


def train(data,method,output,fm=None,split=None,seed=2026,device_name='cuda:0',config=None,resume=False,stop_after=None):
    from .engine import seed_everything,rng_state,restore_rng
    bundle=Bundle(data,fm,split);assets=bundle.macnet
    if assets is None:raise ValueError('Prepare actual raw RGB and annotated MacNet guide inputs')
    cfg=recipe(assets,method,dict(config or {}));dev=device(device_name);out=Path(output).resolve()
    if type(seed) is not int or (stop_after is not None and (type(stop_after) is not int or stop_after<1)):raise ValueError('Declare actual seed and positive completed-epoch interruption count')
    request={'backend':'macnet_complete_raw_reference','method':method,'data':str(bundle.root),'fm':bundle.fm,'split':bundle.split,'seed':seed,'device':str(dev),
        'recipe':cfg,'contract':assets.contract,'bundle_sha256':bundle.fingerprint,'software':software_identity(),'runtime':runtime(),'source_commit':source.COMMIT,
        'pretrained_weight_sha256':source.SWIN22K_SHA256,'guidance_scope':assets.value['guidance_scope'],'additional_query_region_annotations':assets.value['additional_annotations_at_query'],
        'protocol_ports':['strict raw RGB and ordered annotated guidance','source parser tabs expanded8','source wrong import namespace isolated',
            'literal cuda0 mapped to selected current CUDA device','single-device full raw feature graph instead of source4GPU DataParallel',
            'original cached adaptive matrix saved and its gradient connection restored per head call','explicit separately named refresh-adaptive variant',
            'TRAIN/VAL patient disjoint; TEST deferred to evaluation','first valid source VAL accuracy checkpoint initialization'],
        'synthetic':bool(bundle.config.get('synthetic',False)),'test_task_labels_used_for_fit_or_selection':False,'formal_admission':False}
    if resume:
        if read_json(out/'request.json')!=request:raise ValueError('Actual MacNet source/runtime/recipe/raw/guidance inputs changed during resume')
    else:out.mkdir(parents=True,exist_ok=False);write_json(out/'request.json',request)
    lock=_lock(out,resume);last=out/'last.pt';best_path=out/'selected.pt';performed=0
    try:
        seed_everything(seed);torch.set_num_threads(cfg['threads']);model=source.FullMacNet(assets.weights,len(bundle.classes),cfg['condition'],cfg['adaptive']).to(dev)
        groups=_groups(model);scaler=torch.cuda.amp.GradScaler();state=load(last) if exists(last) else None;start=0;best=None;selected_epoch=None;history=[];updates=0
        if state is not None:
            if state['signature']!=digest_object(request):raise ValueError('Actual raw MacNet learned checkpoint signature changed')
            _restore_model(model,state,dev);scaler.load_state_dict(state['scaler']);old_optimizer=torch.optim.RMSprop(groups,lr=1e-5);old_optimizer.load_state_dict(state['optimizer']);groups=old_optimizer.param_groups
            start=state['next_epoch'];best=state['best'];selected_epoch=state['selected_epoch'];history=state['history'];updates=state['updates'];restore_rng(state['rng'])
            if state['finished']:
                chosen=checkpoint_path(best_path);write_json(out/'selected.json',{'model':str(chosen),'sha256':digest_file(chosen),'selected_epoch':selected_epoch,'selection':cfg['selection']});assets.verify_bytes()
                write_json(out/'status.json',{'state':'completed','epochs':start,'updates':updates,'test_evaluated':False,'formal_admission':False});return out
        datasets={s:RawBags(assets,s) for s in ['train','val']}
        loaders={s:DataLoader(datasets[s],batch_size=cfg['batch_size'],shuffle=s=='train',num_workers=0) for s in datasets}
        for epoch in range(start,cfg['epochs']):
            datasets['train'].order=[];datasets['val'].order=[]
            if cfg['lr_policy']=='intended_schedule_repaired':
                for g in groups:g['lr']=source.training_definitions().vit_lr_schedule(epoch)
            optimizer,losses=source.train_epoch(model,groups,loaders['train'],epoch,scaler,cfg['use_amp']);groups=optimizer.param_groups;updates+=len(losses)
            accuracy,val_loss,_,_=source.training_definitions().view_results(mil_feature=model.feature,mil_head=model.head,train_loader=loaders['val'],loss_fn=nn.CrossEntropyLoss(),
                gpu_device=dev.index,batch_size=cfg['batch_size'],bags_len=model.original_bag_length)
            actual_accuracy=float(np.mean(accuracy));actual_loss=float(np.mean(val_loss))
            if not math.isfinite(actual_accuracy) or not math.isfinite(actual_loss):raise FloatingPointError('Original MacNet validation became nonfinite')
            row={'epoch':epoch+1,'train_batch_losses':losses,'train_mean_loss':float(np.mean(losses)),'source_validation_batch_accuracy':actual_accuracy,'validation_CE':actual_loss,
                'requested_lr':source.training_definitions().vit_lr_schedule(epoch),'actual_group_lrs':[g['lr'] for g in groups],
                'train_order':list(datasets['train'].order),'val_order':list(datasets['val'].order),'updates':updates};history.append(row)
            if best is None or actual_accuracy>best:
                best=actual_accuracy;selected_epoch=epoch+1;save({'model':_cpu_state(model),'signature':digest_object(request),'selected_epoch':selected_epoch,'validation_source_accuracy':best},best_path)
            snapshot={'signature':digest_object(request),'model':_cpu_state(model),'gradients':_gradients(model),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                'next_epoch':epoch+1,'best':best,'selected_epoch':selected_epoch,'history':history,'updates':updates,'rng':rng_state(),'finished':epoch+1==cfg['epochs']}
            save(snapshot,last);write_json(out/'history.json',history);performed+=1
            write_json(out/'status.json',{'state':'running','epochs':epoch+1,'updates':updates,'selected_epoch':selected_epoch,'formal_admission':False})
            if stop_after is not None and performed>=stop_after:raise Interrupted()
        selected=checkpoint_path(best_path);write_json(out/'selected.json',{'model':str(selected),'sha256':digest_file(selected),'selected_epoch':selected_epoch,'selection':cfg['selection']})
        assets.verify_bytes();write_json(out/'status.json',{'state':'completed','epochs':cfg['epochs'],'updates':updates,'selected_epoch':selected_epoch,'test_evaluated':False,'formal_admission':False});return out
    except Interrupted:
        write_json(out/'status.json',{**read_json(out/'status.json'),'state':'interrupted'});return out
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','error':repr(error)});raise
    finally:lock.unlink(missing_ok=True)


@torch.no_grad()
def evaluate(run,subset='test',device_name='cuda:0'):
    from .engine import seed_everything
    from .metrics import classification_metrics
    out=Path(run);req=read_json(out/'request.json')
    if subset not in {'val','test'} or req['backend']!='macnet_complete_raw_reference' or read_json(out/'status.json')['state']!='completed':raise ValueError('Complete actual MacNet training before held-out evaluation')
    if req['software']!=software_identity() or req['runtime']!=runtime():raise ValueError('MacNet source/runtime changed')
    bundle=Bundle(req['data'],req['fm'],req['split']);assets=bundle.macnet
    if assets is None or assets.contract!=req['contract'] or bundle.fingerprint!=req['bundle_sha256']:raise ValueError('Actual raw MacNet/guidance task identity changed')
    selected=read_json(out/'selected.json')
    if digest_file(selected['model'])!=selected['sha256']:raise ValueError('Actual selected raw MacNet weights or cached adaptive matrix changed')
    dev=device(device_name);seed_everything(req['seed']);torch.set_num_threads(req['recipe']['threads']);model=source.FullMacNet(assets.weights,len(bundle.classes),req['recipe']['condition'],req['recipe']['adaptive']).to(dev)
    state=load(selected['model'])
    if state['signature']!=digest_object(req):raise ValueError('Selected raw MacNet checkpoint belongs to another request')
    model.load_state_dict(state['model'],strict=True);model.eval();records=[]
    # Private head accepts a single native-sized bag at inference; batch_size
    # is not used to invent another WSI or pad the final patient.
    for r in bundle.records(subset):
        logits,minimum,maximum=model(assets.tensor(r).to(dev));prob=torch.softmax(logits,dim=-1)[0].cpu().numpy()
        records.append({**{k:r[k] for k in ['slide_id','patient_id','global_patient_id','target']},'source_slides':r['slide_id'],'subset':subset,'prediction':int(np.argmax(prob)),
            **{f'p_{i}':float(x) for i,x in enumerate(prob)},'source_min_manifold_distance':float(minimum),'source_max_manifold_distance':float(maximum)})
    table=pd.DataFrame(records);metrics=classification_metrics(table,len(bundle.classes),'slide');metrics.update(subset=subset,selection=req['recipe']['selection'],
        selected_epoch=selected['selected_epoch'],guidance_scope=req['guidance_scope'],additional_query_region_annotations=req['additional_query_region_annotations'],synthetic=req['synthetic'],test_used_for_selection=False)
    assets.verify_bytes();table.to_csv(out/(subset+'_predictions.csv'),index=False);write_json(out/(subset+'_metrics.json'),metrics);return metrics
