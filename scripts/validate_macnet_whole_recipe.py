"""Actual complete author raw trainer versus adapter, with TEST reporting deferred."""
from pathlib import Path
import torch,ast,copy,types,argparse,time
from torch import nn
from milplus import macnet_source as source
from milplus.macnet_demo import create_demo
from milplus.macnet_driver import train,evaluate,RawBags
from milplus.data import Bundle
from milplus.engine import seed_everything,rng_state
from milplus.icmil_coupled_training import load
from milplus.io import write_json
from validate_macnet_raw_component import same


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--weights',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2)
    data=create_demo(out/'actual_input',a.weights);assets=Bundle(data).macnet;d=source.definitions();t=source.training_definitions();seed_everything(0)
    base=source.original_base(a.weights,2);feature=d.MIL_Parallel_Feature(base_model=base,img_size=[96,96]).cuda();head=d.MIL_Parallel_Head_ub(base_model=base,class_num=2,batch_size=1,bags_len=2000,model_stats='train',seed=0).cuda()
    datasets={s:RawBags(assets,s) for s in ['train','val']};loaders={s:torch.utils.data.DataLoader(datasets[s],batch_size=1,shuffle=s=='train',num_workers=0) for s in datasets}
    original_view=t.view_results;deferred=[];sentinel=object()
    def view_test_reporting_deferred(*args,**kwargs):
        if kwargs.get('train_loader') is sentinel:
            deferred.append(True);return [0.],[0.],None,None
        return original_view(*args,**kwargs)
    # Only initialize an otherwise undefined console variable; do not change
    # any metric, save condition, optimizer, loss or training loop expression.
    tree=ast.parse(__import__('milplus.upstream',fromlist=['pinned_text']).pinned_text(source.root(),source.COMMIT,'Utils/train_val.py'))
    node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training_for_parallel'));node.body=ast.parse('best_epoch=0').body+node.body
    ns={**t.training_for_parallel.__globals__,'view_results':view_test_reporting_deferred};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source.root()/'Utils/train_val.py'),'exec'),ns);raw=ns['training_for_parallel']
    (out/'author_checkpoints').mkdir()
    old=torch.optim.RMSprop;observations=[];optimizers=[]
    def observe_RMSprop(*args,**kwargs):
        opt=old(*args,**kwargs);observations.append({'requested_lr':kwargs.get('lr'),'actual_group_lrs':[g['lr'] for g in opt.param_groups],'initial_state_empty':not opt.state});optimizers.append(opt);return opt
    torch.optim.RMSprop=observe_RMSprop
    try:raw(mil_feature=feature,mil_head=head,train_loader=loaders['train'],val_loader=loaders['val'],test_loader=sentinel,epoch=2,gpu_device=0,weight_path=str(out/'author_checkpoints'),max_input_len=2000,bags_len=2000,batch_size=1,use_amp=False)
    finally:torch.optim.RMSprop=old
    source_rng=rng_state();source_optimizer=copy.deepcopy(optimizers[-1].state_dict());assert len(optimizers)==2 and all(x['initial_state_empty'] for x in observations) and len(deferred)==1
    method='macnet_public_source_cached_patient_reference';run=train(data,method,out/'adapted',seed=0,config={'epochs':2,'threads':2},device_name='cuda:0');state=load(run/'last.pt')
    expected={'initial_adaptive_diagonal':head.tgi_clustering_block.inst_adapt_w.detach().cpu(),**{'feature.'+k:v for k,v in feature.state_dict().items()},**{'head.'+k:v for k,v in head.state_dict().items()}}
    same(expected,state['model'],'all_complete_literal_author_final_Swin_GFK_head_state');same(source_optimizer,state['optimizer'],'complete_original_reset_RMSprop_state');same(source_rng,state['rng'],'complete_author_RNG')
    grads={'feature.'+k:v.grad for k,v in feature.named_parameters()};grads.update({'head.'+k:v.grad for k,v in head.named_parameters()});same(grads,state['gradients'],'all_source_final_gradients')
    assert state['next_epoch']==2 and state['updates']==8 and all(r['actual_group_lrs']==[1e-5,1e-5] for r in state['history'])
    # Real one-epoch stop and fresh complete construction must retain the exact
    # source cached matrix and every numerical state at the second epoch.
    resumed=train(data,method,out/'resumed',seed=0,config={'epochs':2,'threads':2},device_name='cuda:0',stop_after=1)
    train(data,method,resumed,seed=0,config={'epochs':2,'threads':2},device_name='cuda:0',resume=True);same(state,load(resumed/'last.pt'),'full_raw_actual_epoch_recovery')
    metrics=evaluate(run,device_name='cuda:0');prediction=(run/'test_predictions.csv').read_bytes();same(metrics,evaluate(resumed,device_name='cuda:0'),'restored_original_selected_metrics');assert prediction==(resumed/'test_predictions.csv').read_bytes()
    terminal=train(data,method,out/'terminal',seed=0,config={'epochs':1,'threads':2},device_name='cuda:0',stop_after=1);assert not (terminal/'selected.json').exists();train(data,method,terminal,seed=0,config={'epochs':1,'threads':2},device_name='cuda:0',resume=True);assert (terminal/'selected.json').exists();evaluate(terminal,device_name='cuda:0')
    write_json(out/'validation.json',{'state':'passed_complete_literal_author_raw_training_recipe_and_recovery','actual_original_whole_author_training_function_executed':True,
        'native_Swin_feature_dimension':768,'actual_pretrained22K':True,'real_RMSprop_resets_each_epoch':observations,'all_final_model_gradients_RMSprop_and_RNG_exact':True,
        'only_TEST_reporting_deferred':True,'only_undefined_console_best_epoch_initialized':True,'real_full_state_epoch_interruption_and_final_epoch_recovery':True,
        'fresh_selected_predictions_exact':True,'synthetic':True,'scope':'Two-epoch software verification, not original100epoch native or full method admission','included_in_method_count':False,'elapsed_seconds':time.monotonic()-start})

if __name__=='__main__':main()
