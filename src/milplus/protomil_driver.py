"""Complete SAE -> annotated concepts/review -> original ProtoMIL training."""
from pathlib import Path
from types import SimpleNamespace
import copy,random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from . import protomil_source as source
from . import protomil_training as sae_training
from .protomil_stages import normalized_inputs,model_sha,conceptualize,load_concepts,probe_bank,read_review,review_path
from .data import Bundle
from .io import read_json,write_json,digest_file,digest_object
from .provenance import software_identity
from .akdpmp_driver import _device
from .icmil_coupled_driver import _lock
from .icmil_coupled_training import save,load,exists,checkpoint_path,Interrupted

PROFILES={f'protomil_{condition}_{protocol}_patient_reference':{'intervened':condition=='intervened',
    'sae_epochs':60 if protocol=='paper' else 200,'mil_epochs':200 if protocol=='paper' else 1,
    'selection':'best_VAL_AUROC_first_valid' if protocol=='paper' else 'last_epoch_source_eval_entry'}
    for condition in ['concept','intervened'] for protocol in ['paper','released_defaults']}
DEFAULT_PROFILE='protomil_concept_paper_patient_reference'


class Progress:
    def __init__(self,root,stop_after):self.root=root;self.stop_after=stop_after;self.steps=0
    def record(self,phase,completed,finished):
        self.steps+=1;write_json(self.root/'status.json',{'state':'running','phase':phase,'phase_steps':completed,'phase_finished':finished,'steps_this_invocation':self.steps,'formal_admission':False})
        if self.stop_after is not None and self.steps>=self.stop_after:raise Interrupted()


def recipe(assets,config,method):
    if method not in PROFILES:raise ValueError('Choose explicit full ProtoMIL concept/intervened and paper/released-default conditions')
    if set(config)-{'sae_epochs','mil_epochs','probe_per_class','threads','cache_root','verify_existing_cache'}:raise ValueError('Preserve original512/2048 SAE,4096batch, resampler, native losses/Adam and full ProtoMIL')
    cfg={**PROFILES[method],'probe_per_class':10000,'threads':2,'verify_existing_cache':False,
        'native_sae_batch':4096,'sae_lr':1e-4,'sae_activation_L1':3e-4,'MIL_lr':1e-4,'classifier_mean_L1':1.,
        'sae_val_frequency':50000,'sae_checkpoint_frequency':250000,'resample_epoch_frequency':500000,
        'source_MIL_train_and_VAL_shuffle':True,'TEST_used_for_training_or_selection':False}
    cfg.update({k:v for k,v in config.items() if k not in {'cache_root'}})
    if any(type(cfg[k]) is not int or cfg[k]<1 for k in ['sae_epochs','mil_epochs','threads']) or type(cfg['probe_per_class']) is not int or cfg['probe_per_class']<5 or type(cfg['verify_existing_cache']) is not bool:
        raise ValueError('Declare valid actual phase budgets, threads and normal/tumor probe counts')
    if cfg['intervened'] and not assets.review_file:raise ValueError('Intervened ProtoMIL needs a fixed path for actual SAE-bound review; it can be supplied after concept inspection')
    if any(sum(p['region_label']==label for p in assets.probes)!=cfg['probe_per_class'] for label in [0,1]):raise ValueError('Prepared actual probing population must exactly match the explicit normal/tumor budget')
    return cfg


def _save_final_sae(pipeline,path,assets):
    selected={'model':{k:v.detach().cpu().clone() for k,v in pipeline.autoencoder.state_dict().items()},
        'sae_state_sha256':model_sha(pipeline.autoencoder),'contract':assets.contract,'selection':'original final SAE epoch; no validation-side model replacement'}
    save(selected,path);return checkpoint_path(path)


def sae_phase(assets,cfg,cache,out,device,seed,progress):
    from .engine import seed_everything,restore_rng
    out.mkdir(parents=True,exist_ok=True);last=out/'last.pt';selected=out/'selected_SAE.pt'
    source.definitions()  # Verify actual private dependency before trusted own checkpoint loading.
    previous=load(last) if exists(last) else None
    if previous is not None and previous['finished']:
        if previous['contract']!=assets.contract or previous['sae_epochs']!=cfg['sae_epochs']:raise ValueError('Actual finished SAE phase identity changed')
        restore_rng(previous['rng']);return checkpoint_path(selected)
    seed_everything(seed);paths=normalized_inputs(assets,cache);random.shuffle(paths)
    pivot=int(.8*len(paths));train_files=paths[:pivot];val_files=paths[pivot:]
    if not train_files or not val_files:raise ValueError('Original SAE80/20 file split produced an empty partition')
    pipeline,state=sae_training.context(train_files,val_files,out/'source_checkpoints',epochs=cfg['sae_epochs'],device=device)
    if previous is not None:
        if previous['contract']!=assets.contract or previous['sae_epochs']!=cfg['sae_epochs']:raise ValueError('Actual SAE phase inputs/budget changed')
        sae_training.restore(pipeline,state,previous['snapshot'])
    while state['next_piece']<pipeline.actual_epochs:
        sae_training.piece(pipeline,state);finished=state['next_piece']==pipeline.actual_epochs
        if finished:
            pipeline.save_checkpoint(is_final=True);_save_final_sae(pipeline,selected,assets)
        snapshot=sae_training.snapshot(pipeline,state)
        save({'snapshot':snapshot,'finished':finished,'sae_epochs':cfg['sae_epochs'],'contract':assets.contract,'rng':snapshot['rng']},last)
        write_json(out/'history.json',{'actual_completed_pieces':state['next_piece'],'native_epochs':cfg['sae_epochs'],
            'pieces_per_epoch':len(paths[:pivot]),'TRAIN_piece_files':train_files,'pretraining_VAL_piece_files':val_files,
            'last_validation_counter':state['last_validated'],'last_checkpoint_counter':state['last_checkpoint'],
            'reported_activation_counter_source_rounds_tail_to4096':pipeline.total_activations_trained_on,
            'source_resampler_state':{k:v for k,v in pipeline.activation_resampler.__dict__.items() if isinstance(v,(int,float,str,bool)) and (not isinstance(v,float) or np.isfinite(v))}})
        progress.record('SAE',state['next_piece'],finished)
    return checkpoint_path(selected)


def restored_sae(selected,assets,device):
    source.definitions();state=load(selected)
    if state['contract']!=assets.contract:raise ValueError('Actual selected SAE input identity changed')
    network=source.autoencoder().to(device);network.load_state_dict(state['model'],strict=True);network.eval();network.requires_grad_(False)
    if model_sha(network)!=state['sae_state_sha256']:raise ValueError('Actual selected full SAE weights changed')
    return network,state['sae_state_sha256']


class Concepts(Dataset):
    def __init__(self,rows,index,removed):self.rows=rows;self.index=index;self.removed=removed;self.order=[]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];self.order.append(row['slide_id'])
        return {'feature':load_concepts(self.index[row['slide_id']],self.removed),'label':row['target']}


def _mil_args():
    args=source.definitions().mil.get_parser().parse_args([])
    args.feature_type='concept';args.which_classifier='proto_mil';args.autoencoder_input_dim=512;args.expansion_factor=4
    args.num_classes=2;args.probe_nclasses=2;args.train_mode='train';args.use_wandb=False
    return args


def _gradients(model):return {k:None if p.grad is None else p.grad.detach().cpu().clone() for k,p in model.named_parameters()}


def mil_phase(assets,cfg,cache,selected_sae,review,out,device,seed,progress):
    from .engine import seed_everything,rng_state,restore_rng
    out.mkdir(parents=True,exist_ok=True);last=out/'last.pt';selected=out/'selected_MIL.pt';source.definitions()
    previous=load(last) if exists(last) else None
    signature={'contract':assets.contract,'sae':load(selected_sae)['sae_state_sha256'],'review':review,'recipe':cfg}
    if previous is not None and previous['signature']!=signature:raise ValueError('Actual concept/MIL recipe, review or fitted SAE changed')
    if previous is not None and previous['finished']:
        restore_rng(previous['rng']);return checkpoint_path(selected)
    # Original stages are separate seeded programs. Reset at the start of MIL;
    # frozen feature extraction and concept inspection do not pick the seed.
    seed_everything(seed);sae,_=restored_sae(selected_sae,assets,device)
    index={}
    for subset in ['train','val']:
        for row in assets.bundle.records(subset):index[row['slide_id']]=conceptualize(assets,sae,assets.rows[row['slide_id']],cache,device,cfg['verify_existing_cache'])
    del sae
    seed_everything(seed);args=_mil_args();model,optimizer,classification_loss,sparsity_loss=source.definitions().mil.prepare_model(args,device)
    generators={name:torch.Generator().manual_seed(seed) for name in ['train','val']}
    datasets={name:Concepts(assets.bundle.records(name),index,review['removed_concepts']) for name in ['train','val']}
    loaders={name:DataLoader(datasets[name],batch_size=1,shuffle=True,num_workers=0,generator=generators[name]) for name in ['train','val']}
    start=0;best=None;history=[];updates=0;selected_epoch=None
    if previous is not None:
        model.load_state_dict(previous['model'],strict=True);optimizer.load_state_dict(previous['optimizer'])
        for name,p in model.named_parameters():p.grad=None if previous['gradients'][name] is None else previous['gradients'][name].to(device).clone()
        for name,g in generators.items():g.set_state(previous['generators'][name])
        start=previous['next_epoch'];best=previous['best'];history=previous['history'];updates=previous['updates'];selected_epoch=previous['selected_epoch'];restore_rng(previous['rng'])
    for epoch in range(start,cfg['mil_epochs']):
        model.train();datasets['train'].order=[];datasets['val'].order=[]
        losses=source.mil_epoch(args,model,optimizer,loaders['train'],device);updates+=losses['updates']
        metrics=source.mil_validation(args,model,loaders['val'],device)
        if not all(np.isfinite(v) for v in [losses['loss'],metrics['loss'],metrics['auroc']]):raise FloatingPointError('Original ProtoMIL objective or validation became nonfinite')
        row={'epoch':epoch+1,**losses,'epoch_updates':losses['updates'],'updates':updates,'validation':metrics,'train_order':list(datasets['train'].order),'val_order':list(datasets['val'].order)};history.append(row)
        improved=best is None or metrics['auroc']>best
        if improved:best=metrics['auroc'];selected_epoch=epoch+1
        choose=improved if cfg['selection']=='best_VAL_AUROC_first_valid' else True
        if choose:
            save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'signature':signature,'selected_epoch':epoch+1,
                'val_auroc':metrics['auroc'],'selection':cfg['selection']},selected)
        finished=epoch+1==cfg['mil_epochs']
        if finished and cfg['selection']=='last_epoch_source_eval_entry':selected_epoch=epoch+1
        state={'signature':signature,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),
            'gradients':_gradients(model),'generators':{k:g.get_state() for k,g in generators.items()},'rng':rng_state(),
            'next_epoch':epoch+1,'history':history,'best':best,'selected_epoch':selected_epoch,'updates':updates,'finished':finished}
        save(state,last);write_json(out/'history.json',history);progress.record('MIL',epoch+1,finished)
    return checkpoint_path(selected)


def train(data,output,fm=None,split=None,seed=2026,device='cpu',config=None,resume=False,stop_after=None,method=DEFAULT_PROFILE):
    from .engine import seed_everything
    config=dict(config or {});bundle=Bundle(data,fm,split);assets=bundle.protomil
    if assets is None:raise ValueError('Prepare actual CONCH, SAE source populations and annotated TRAIN probes')
    if type(seed) is not int or (stop_after is not None and (type(stop_after) is not int or stop_after<1)):raise ValueError('Declare valid seed and real SAE-piece/MIL-epoch interruption count')
    cfg=recipe(assets,config,method);dev=_device(device);out=Path(output).resolve();cache=Path(config.get('cache_root',out.parent/'.protomil_cache')).expanduser().resolve()
    request={'backend':'protomil_complete_reference','method':method,'data':str(bundle.root),'fm':bundle.fm,'split':bundle.split,'seed':seed,'device':str(dev),
        'recipe':cfg,'cache_root':str(cache),'contract':assets.contract,'bundle_sha256':bundle.fingerprint,'software':software_identity(),
        'source_commit':source.COMMIT,'reviewed_DTN_dependency_commit':source.DTN_COMMIT,'runtime':source.runtime_identity(),
        'protocol_ports':['explicit TRAIN-only SAE population','declared probe order; positive before negative','fixed source private SAE dependency',
            'local logging, no external telemetry','lossless float32 HDF5 concept cache','independent TRAIN/VAL generators and single-process I/O',
            'TEST deferred to explicit evaluation','first valid VAL initialization for paper best-model selection'],
        'step_unit':'actual SAE WSI piece, concept preparation, or actual MIL epoch','synthetic':bool(bundle.config.get('synthetic',False)),
        'test_used_for_training_or_selection':False,'formal_admission':False}
    if resume:
        if read_json(out/'request.json')!=request:raise ValueError('ProtoMIL resume changed actual inputs, source/runtime, full recipe or cache')
    else:out.mkdir(parents=True,exist_ok=False);write_json(out/'request.json',request)
    lock=_lock(out,resume);progress=Progress(out,stop_after)
    try:
        seed_everything(seed);torch.set_num_threads(cfg['threads'])
        selected_sae=sae_phase(assets,cfg,cache,out/'SAE',dev,seed,progress)
        probe_dir=out/'concepts';bank_path=probe_dir/'concept_bank.json'
        if not bank_path.exists():
            sae,sha=restored_sae(selected_sae,assets,dev);bank=probe_bank(assets,sae,cache,probe_dir,dev,cfg['probe_per_class']);del sae
            progress.record('concepts',1,True)
        else:
            bank=read_json(bank_path);expected=bank.pop('bank_sha256')
            if digest_object(bank)!=expected or bank['contract']!=assets.contract or bank['sae_state_sha256']!=load(selected_sae)['sae_state_sha256']:raise ValueError('Actual source concept bank changed')
            bank['bank_sha256']=expected
        review=read_review(assets,bank,cfg['intervened'])
        if review is None:
            write_json(out/'review_request.json',{'schema_version':1,'sae_state_sha256':bank['sae_state_sha256'],'concept_bank_sha256':bank['bank_sha256'],
                'active_concepts':bank['active_concepts'],'inspection_report':str(probe_dir/'concept_inspection.html'),'review_file':str(review_path(assets,bank)),
                'required_fields':['reviewer','provenance','removed_concepts','concept_names'],'software_has_not_performed_pathologist_review':True})
            write_json(out/'status.json',{'state':'awaiting_concept_review','phase':'concepts','review_file':str(review_path(assets,bank)),'formal_admission':False});return out
        bound=out/'review_bound.json'
        if bound.exists() and read_json(bound)!=review:raise ValueError('Actual frozen concept review changed')
        write_json(bound,review)
        selected_mil=mil_phase(assets,cfg,cache,selected_sae,review,out/'MIL',dev,seed,progress)
        assets.verify_bytes();write_json(out/'selected.json',{'SAE':str(selected_sae),'SAE_file_sha256':digest_file(selected_sae),'MIL':str(selected_mil),
            'MIL_file_sha256':digest_file(selected_mil),'concept_bank_sha256':bank['bank_sha256'],'review':review,'selection':cfg['selection']})
        write_json(out/'status.json',{'state':'completed','phases':['SAE','concepts','MIL'],'test_evaluated':False,'formal_admission':False});return out
    except Interrupted:
        write_json(out/'status.json',{**read_json(out/'status.json'),'state':'interrupted'});return out
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','error':repr(error)});raise
    finally:lock.unlink(missing_ok=True)


@torch.no_grad()
def evaluate(run,subset='test',device='cpu'):
    from .engine import seed_everything
    from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
    out=Path(run).resolve();request=read_json(out/'request.json')
    if subset not in {'val','test'} or request.get('backend')!='protomil_complete_reference' or read_json(out/'status.json')['state']!='completed':raise ValueError('Complete actual SAE, concepts/review and MIL before held-out evaluation')
    if request['software']!=software_identity() or request['runtime']!=source.runtime_identity():raise ValueError('ProtoMIL source/dependency runtime changed')
    bundle=Bundle(request['data'],request['fm'],request['split']);assets=bundle.protomil;selected=read_json(out/'selected.json')
    if assets is None or assets.contract!=request['contract'] or bundle.fingerprint!=request['bundle_sha256']:raise ValueError('Actual ProtoMIL patient/feature/annotation identity changed')
    for key in ['SAE','MIL']:
        if digest_file(selected[key])!=selected[key+'_file_sha256']:raise ValueError('Actual selected full ProtoMIL model bytes changed')
    bank=read_json(out/'concepts/concept_bank.json');review=read_review(assets,bank,request['recipe']['intervened'])
    if review!=selected['review']:raise ValueError('Actual final ProtoMIL concept review changed')
    dev=_device(device);seed_everything(request['seed']);torch.set_num_threads(request['recipe']['threads']);sae,_=restored_sae(selected['SAE'],assets,dev)
    state=load(selected['MIL']);args=_mil_args();model,_,_,_=source.definitions().mil.prepare_model(args,dev);model.load_state_dict(state['model'],strict=True);model.eval();records=[]
    for row in bundle.records(subset):
        cache=conceptualize(assets,sae,assets.rows[row['slide_id']],request['cache_root'],dev,request['recipe']['verify_existing_cache'])
        x=load_concepts(cache,review['removed_concepts']);result=model(x.unsqueeze(0).to(dev));p=result['Y_prob'][0].cpu().numpy()
        records.append({**{k:row[k] for k in ['slide_id','patient_id','global_patient_id','target']},'source_slides':row['slide_id'],
            'subset':subset,'prediction':int(result['Y_hat'][0]),'p_0':float(p[0]),'p_1':float(p[1])})
    truth=[r['target'] for r in records];prediction=[r['prediction'] for r in records];auc=float(roc_auc_score(truth,[r['p_1'] for r in records]))
    metrics={'subset':subset,'accuracy':float(accuracy_score(truth,prediction)),'macro_f1':float(f1_score(truth,prediction,average='macro',zero_division=0)),
        'auroc':auc,'macro_auroc':auc,'slides':len(records),'independent_patients':len({r['global_patient_id'] for r in records}),
        'selection':request['recipe']['selection'],'test_used_for_selection':False,'source':source.COMMIT,'review':review,'synthetic':request['synthetic']}
    assets.verify_bytes();pd.DataFrame(records).to_csv(out/(subset+'_predictions.csv'),index=False);write_json(out/(subset+'_metrics.json'),metrics);return metrics
