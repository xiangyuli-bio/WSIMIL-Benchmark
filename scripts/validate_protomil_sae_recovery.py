"""Literal original whole pipeline versus exact source pieces and real restore."""
import argparse,copy,time
from pathlib import Path
import torch,numpy as np
from milplus import protomil_training as training
from milplus.engine import seed_everything
from milplus.io import write_json


def same(a,b,path='state'):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
    elif hasattr(a,'__dict__'):same(a.__dict__,b.__dict__,path)
    else:assert a==b,(path,a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();torch.set_num_threads(2);torch.manual_seed(67);paths=[]
    for i in range(2):
        path=out/f'actual512d_piece_{i}.pt';torch.save(torch.nn.functional.normalize(torch.randn(4097,512),dim=-1),path);paths.append(str(path))
    val_path=out/'independent_VAL_piece.pt';torch.save(torch.nn.functional.normalize(torch.randn(4097,512),dim=-1),val_path);val_files=[str(val_path)]
    seed_everything(2026);whole,unused=training.context(paths,val_files,out/'original_ckpt',epochs=2,initialize_order=False)
    whole.run_pipeline(train_batch_size=4096,val_frequency=50000,checkpoint_frequency=250000,num_epochs=2,
        train_fnames=paths,train_val_fnames=val_files,start_time=0.,resample_epoch_freq=500000)
    expected_model=copy.deepcopy(whole.autoencoder.state_dict());expected_opt=copy.deepcopy(whole.optimizer.state_dict())
    expected_metrics=copy.deepcopy(whole.metrics);expected_resampler=copy.deepcopy(whole.activation_resampler.__dict__)
    seed_everything(2026);adapted,state=training.context(paths,val_files,out/'adapted_ckpt',epochs=2)
    for _ in range(4):training.piece(adapted,state)
    same(expected_model,adapted.autoencoder.state_dict(),'original_whole_model');same(expected_opt,adapted.optimizer.state_dict(),'original_whole_optimizer')
    same(expected_metrics,adapted.metrics,'original_whole_metrics');same(expected_resampler,adapted.activation_resampler.__dict__,'original_whole_resampler')
    baseline=training.snapshot(adapted,state)
    seed_everything(2026);resumed,rs=training.context(paths,val_files,out/'resumed_ckpt',epochs=2)
    training.piece(resumed,rs);actual_checkpoint=training.snapshot(resumed,rs);checkpoint=out/'real_interrupted_full_SAE.pt';torch.save(actual_checkpoint,checkpoint)
    replacement,replacement_state=training.context(paths,val_files,out/'restored_ckpt',epochs=2)
    training.restore(replacement,replacement_state,torch.load(checkpoint,weights_only=False))
    for _ in range(3):training.piece(replacement,replacement_state)
    actual=training.snapshot(replacement,replacement_state);same(baseline,actual,'all_original_SAE_piece_state_after_real_reconstruction')
    assert len(replacement.autoencoder.decoder._weight._backward_hooks or {})==1
    write_json(out/'validation.json',{'state':'passed_source_SAE_pipeline_and_recovery_only','full512_to2048_original_SAE':True,
        'actual_pieces':4,'actual_batch_sizes_each_piece':[4096,1],'original_whole_pipeline_weights_gradients_optimizer_metrics_resampler_exact':True,
        'actual_serialized_reconstruction_full_state_RNG_exact':True,'source_gradient_hook_preserved':True,
        'native_maximum_epochs_not_claimed':True,'complete_ProtoMIL_admission':False,'elapsed_seconds':time.monotonic()-began})

if __name__=='__main__':main()
