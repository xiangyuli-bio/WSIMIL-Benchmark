"""Original Python3.11 inference runtime; no query labels and no downstream optimizer."""
import argparse
import importlib.metadata
from pathlib import Path
import platform

import numpy as np
import torch

from .icmil_context_source import ARCH,COMMIT,WEIGHTS,load_native
from .io import digest_file,digest_object,read_json,write_json


def identity():
    root=Path(__file__).resolve().parent
    return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'source_sha256':digest_object({str(p.relative_to(root)):digest_file(p) for p in sorted(root.rglob('*.py'))}),
        'original_commit':COMMIT,'cuda':torch.version.cuda,
        'versions':{n:importlib.metadata.version(n) for n in ['h5py','scikit-learn','scipy','schedulefree']}}


def predict(call):
    allowed={'native_runtime','weights','weights_sha256','support','support_sha256','queries','queries_sha256',
        'threads','device','model_seed','output'}
    if set(call)!=allowed:raise ValueError('Native ICMIL accepts only frozen weights, TRAIN support and unlabelled queries')
    if identity()!=call['native_runtime']:raise ValueError('Actual original ICMIL Python/Torch/source environment changed')
    for name,key in [('weights','weights_sha256'),('support','support_sha256'),('queries','queries_sha256')]:
        if digest_file(call[name])!=call[key]:raise ValueError('ICMIL checkpoint/support/query bytes changed')
    if call['weights_sha256']!=WEIGHTS[call['model_seed']]:raise ValueError('ICMIL seed identity mismatch')
    if type(call['threads']) is not int or call['threads']<1:raise ValueError('Native thread count must be positive')
    torch.set_num_threads(call['threads']);device=torch.device(call['device'])
    if device.type=='cuda':
        if not torch.cuda.is_available():raise ValueError('Actual requested ICMIL CUDA environment unavailable; no CPU fallback')
        torch.cuda.set_device(device)
    elif device.type!='cpu':raise ValueError('Use explicit CPU or CUDA')
    with np.load(call['support'],allow_pickle=False) as f:
        x=f['features'];y=f['targets']
    q=np.load(call['queries'],allow_pickle=False)
    if (x.ndim!=3 or q.ndim!=3 or not len(x) or not len(q) or x.shape[1:]!=q.shape[1:]
            or not 1<=x.shape[2]<=25 or x.dtype!=np.float32 or q.dtype!=np.float32
            or y.shape!=(len(x),) or y.dtype!=np.int64 or set(y.tolist())!={0,1}
            or not np.isfinite(x).all() or not np.isfinite(q).all()):
        raise ValueError('ICMIL needs native fixed-length finite bags, complete binary TRAIN context and at most25 explicit features')
    model=load_native(call['weights'],call['model_seed'],device)
    before={n:p._version for n,p in model.named_parameters()}
    with torch.no_grad():
        logits=model(torch.from_numpy(x).unsqueeze(0).to(device),torch.from_numpy(y).unsqueeze(0).to(device),
            torch.from_numpy(q).unsqueeze(0).to(device))
    if logits.shape!=(1,len(q),2) or not torch.isfinite(logits).all():raise ValueError('ICMIL did not return the original binary query logits')
    if before!={n:p._version for n,p in model.named_parameters()} or any(p.grad is not None for p in model.parameters()):
        raise ValueError('The frozen pretrained context model must not update parameters or optimize against query labels')
    if digest_file(call['weights'])!=call['weights_sha256']:raise ValueError('Original ICMIL checkpoint changed during prediction')
    output=Path(call['output']);np.save(output,logits.squeeze(0).cpu().numpy(),allow_pickle=False)
    probabilities=output.with_suffix('.probabilities.npy')
    np.save(probabilities,torch.softmax(logits.squeeze(0),dim=-1).cpu().numpy(),allow_pickle=False)
    receipt={'state':'passed','architecture':ARCH,'parameters':8257552,'pretraining_epochs':8000,
        'downstream_optimizer_updates':0,'model_seed':call['model_seed'],'context_bags':len(x),'query_bags':len(q),
        'bag_size':x.shape[1],'input_features':x.shape[2],'source_right_padding_to25':x.shape[2]<25,
        'query_labels_available_to_native_process':False,'all_context_and_queries_used_without_hidden_sampling':True,
        'device':str(device),'parameter_devices':sorted({str(p.device) for p in model.parameters()}),
        'output_sha256':digest_file(output),'probabilities_sha256':digest_file(probabilities),'native_runtime':identity()}
    write_json(output.with_suffix('.receipt.json'),receipt);return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--call',required=True);a=p.parse_args();predict(read_json(a.call))


if __name__=='__main__':main()
