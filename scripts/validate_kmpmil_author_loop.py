"""Independent literal original _run_training oracle, full model and real Adam."""
import argparse,copy,time
from pathlib import Path
from types import SimpleNamespace
import torch
from milplus import kmpmil_source as source
from milplus.kmpmil_driver import train,loaders,recipe,learned_state,gradients,restore_model,tc_state
from milplus.kmpmil_demo import create_demo
from milplus.icmil_coupled_training import load
from milplus.engine import seed_everything
from milplus.data import Bundle
from milplus.io import write_json
from validate_kmpmil_lifecycle import same


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--weights',required=True);p.add_argument('--sha256',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();data=create_demo(out/'data',a.weights,a.sha256)
    actual=train(data,out/'adapted',config={'epochs':3},device=a.device,stop_after=3);actual_state=load(actual/'last.pt');assert actual_state['next_task']==1
    assets=Bundle(data).kmpmil;cfg=recipe(assets,{'epochs':3});cfg['task_num']=1;dev=torch.device(a.device);cfg['cuda_id']=dev.index or 0
    seed_everything(2026);torch.set_num_threads(2);base=source.load_base(a.weights,a.sha256,dev);data_loaders,generators=loaders(assets,2026)
    tc=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(len(t['classes']),512,dtype=base.dtype,device=dev),requires_grad=False) for t in assets.tasks])
    d=source.definitions();d.utilities.activate_current_tc_residual(tc,1);pools=source.task_pools(cfg,base,len(assets.descriptors(0)))
    model=d.model.KMPMIL(cfg,base,dev,pools[0],pools[1],tc,assets.ensemble(0),pfc_gamma_delta_pool=pools[2],pfc_beta_pool=pools[3],past_keys=[],current_descriptors=assets.descriptors(0)).eval()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['adam_lr'],eps=cfg['adam_eps'],weight_decay=cfg['adam_weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode=cfg['lrs_mode'],factor=cfg['lrs_factor'],patience=3,threshold=cfg['lrs_threshold'],threshold_mode=cfg['lrs_threshold_mode'])
    early=d.utilities.EarlyStopping(warmup=cfg['es_warmup'],patience=10,verbose=False,threshold=cfg['es_threshold'])
    ctx=SimpleNamespace(cfg=cfg,model=model,optimizer=optimizer,device=dev,lr_scheduler=scheduler,early_stop=early,loss_function=torch.nn.CrossEntropyLoss(),current_dataset=assets.tasks[0]['id'])
    ctx.data_loader={ctx.current_dataset:{'train':data_loaders['0:train'],'val':data_loaders['0:val']}}
    ctx._update_network=d.training.TrainingMixin._update_network.__get__(ctx)
    ctx._train_each_epoch=d.training.TrainingMixin._train_each_epoch.__get__(ctx)
    ctx.eval_model=d.evaluation.EvaluationMixin.eval_model.__get__(ctx)
    ctx._eval_and_print=d.evaluation.EvaluationMixin._eval_and_print.__get__(ctx)
    ctx._compute_metrics=d.metrics.MetricsMixin._compute_metrics.__get__(ctx)
    saved={}
    def save_original(epoch,ckpt_type):saved[ckpt_type]={'epoch':epoch,'model':learned_state(model),'gradients':gradients(model),'optimizer':copy.deepcopy(optimizer.state_dict()),'scheduler':copy.deepcopy(scheduler.state_dict()),'early':copy.deepcopy(early.__dict__)}
    ctx._save_model=save_original
    d.training.TrainingMixin._run_training(ctx)
    same(saved['last']['model'],load(actual/'task_0_last.pt')['model'],'literal_last_model')
    best=load(actual/'task_0_best.pt');same(saved['best']['model'],best['model'],'literal_best_model');assert saved['best']['epoch']==best['epoch']
    restore_model(model,saved['best']['model']);same(learned_state(model),actual_state['model'],'best_reloaded_before_bank')
    same(gradients(model),actual_state['gradients'],'source_gradients_after_best_reload')
    same(tc_state(tc),actual_state['tc'],'actual_tc')
    same({k:g.get_state() for k,g in generators.items()},actual_state['generators'],'all_original_loader_generators')
    assert actual_state['snapshots'][0]['selected_epoch']==saved['best']['epoch']
    write_json(out/'validation.json',{'state':'passed','literal_author_run_training':True,'full_original_CONCH_text_parameters':110129664,
        'last_model_best_model_gradient_TC_generators_and_bank_selection_exact':True,'source_commit':source.COMMIT,'elapsed_seconds':time.monotonic()-start,'synthetic':True,'paper_reproduction':False})

if __name__=='__main__':main()
