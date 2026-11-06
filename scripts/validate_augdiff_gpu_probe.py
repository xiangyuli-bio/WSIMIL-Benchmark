"""Independent-wheel CUDA AMP diagnostics and complete short MIL continuation.

This is a real-device SOFTWARE budget, not the200max/p10/native-budget gate.
"""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['wheel','output','source-tree','prepared-data']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    target=out/'wheel_import';target.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        for name in archive.namelist():
            if name.startswith('milplus/') and not name.endswith('/'):
                assert archive.read(name)==(Path(args.source_tree)/'src'/name).read_bytes(),name
        archive.extractall(target)
    program=r'''
import gc,sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import numpy as np
import torch,milplus
from milplus.augdiff_source import build_diffusion,epsilon_loss
from milplus.data import Bundle
from milplus.engine import train,evaluate
from milplus.io import read_json,write_json,digest_file
from milplus.provenance import software_identity

target,out,data=map(Path,sys.argv[1:]);assert Path(milplus.__file__).resolve().is_relative_to(target.resolve())
assert milplus.__version__=='0.68.0' and torch.cuda.is_available()
torch.set_num_threads(2)
report={'state':'running','software':software_identity(),'device':torch.cuda.get_device_name(0),'formal_admission':False}
write_json(out/'progress.json',report)
# Actual full native DAE forward/backward under CUDA FP16. No smaller block,
# bypass of checkpointing or FP32 training fallback can satisfy this check.
model=build_diffusion(512).cuda();model.train();optimizer=model.configure_optimizers()
scaler=torch.amp.GradScaler('cuda');updates=0;losses=[]
for step in range(2):
 optimizer.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):loss,_=epsilon_loss(model,torch.randn(2,512,device='cuda'))
 assert torch.isfinite(loss)
 scale=scaler.get_scale();scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
 updates+=int(scaler.get_scale()>=scale);model.on_train_batch_end();losses.append(float(loss.detach()))
assert updates>0 and int(model.model_ema.num_updates)==2
report['full_DAE_CUDA_fp16']={'state':'passed','parameters':sum(p.numel() for p in model.parameters()),
 'attempts':2,'actual_updates':updates,'losses':losses,'EMA_updates':2,'batch_size':2,'scope':'Short software compatibility budget only'}
assert report['full_DAE_CUDA_fp16']['parameters']==122055680
write_json(out/'progress.json',report);del model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()
# This input was actually prepared on CPU. Its producer Torch/backend identity
# stays recorded; the consumer CUDA build is separately audited here.
bundle=Bundle(data);assert bundle.augdiff.metadata['pretrain_protocol']['device_type']=='cpu'
assert bundle.dim==512 and len(bundle.records('train'))==17
profile='augdiff_amil_schedule_repaired_reconstruction';cfg={'epochs':2,'patience':2,'threads':2}
train(data,profile,out/'continuous',device='cuda',seed=2026,config=cfg)
train(data,profile,out/'resumed',device='cuda',seed=2026,config=cfg,stop_after=1)
assert read_json(out/'resumed/status.json')['state']=='interrupted'
train(data,profile,out/'resumed',device='cuda',seed=2026,config=cfg,resume=True)
a=torch.load(out/'continuous/last.pt',map_location='cpu',weights_only=False)
b=torch.load(out/'resumed/last.pt',map_location='cpu',weights_only=False)
def same(x,y):
 if torch.is_tensor(x):return torch.equal(x,y)
 if isinstance(x,np.ndarray):return np.array_equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return type(x)==type(y) and len(x)==len(y) and all(same(i,j) for i,j in zip(x,y))
 return x==y
for key in ['model','optimizers','rng','history','epoch','best','bad']:assert same(a[key],b[key]),key
extra=a['model']['_extra_state'];assert extra['completed_epochs']==2 and extra['optimizer_updates']>0
assert extra['scaler'] and all(len(order)==119 for order in extra['train_orders'])
assert all(len(w[-1]['slides'])==1 and w[-1]['divisor']==2 for w in extra['window_history'])
metrics=evaluate(out/'continuous',device='cuda');sha=digest_file(out/'continuous/test_predictions.csv')
assert evaluate(out/'continuous',device='cuda')==metrics and digest_file(out/'continuous/test_predictions.csv')==sha
assert evaluate(out/'resumed',device='cuda')==metrics and digest_file(out/'resumed/test_predictions.csv')==sha
report.update(state='passed',CPU_producer_CUDA_consumer='passed',MIL_CUDA_fp16_epochs=2,
 MIL_actual_updates=extra['optimizer_updates'],MIL_attempted_windows=sum(len(w) for w in extra['window_history']),
 exact_resume=['classifier','RAdam','Lookahead','scaler','window_orders','VALstream','allRNG','history'],
 selected_checkpoint_reloaded_twice=True,prediction_sha256=sha,
 scope='Actual H200 full122MDAE CUDAFP16 backward/EMA and CPU-prepared frozenDAE consumer, complete2epoch AMP MIL/119draws/e and exact interrupted-resume. Not native200epoch/369preparation budget or natural WSI/paper-score reproduction.')
for name,module in list(sys.modules.items()):
 if name.startswith('milplus') and getattr(module,'__file__',None):assert Path(module.__file__).resolve().is_relative_to(target.resolve()),name
write_json(out/'validation.json',report)
print(json.dumps({'state':report['state'],'device':report['device'],'MIL_updates':report['MIL_actual_updates']}),flush=True)
'''
    env=dict(os.environ)
    env['PYTHONPATH']=os.pathsep.join(p for p in env.get('PYTHONPATH','').split(os.pathsep)
        if p and Path(p).resolve()!=(Path(args.source_tree)/'src').resolve())
    with (out/'program.log').open('w') as log:
        subprocess.run([sys.executable,'-c',program,str(target),str(out),str(Path(args.prepared_data).resolve())],
            cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
