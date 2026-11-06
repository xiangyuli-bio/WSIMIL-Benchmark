"""Explicit CUDA-kernel admission check; never treats an unavailable GPU as a pass."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from milplus.io import write_json
from milplus.msmmil import author_network
from milplus.provenance import software_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--lengths',nargs='+',type=int,default=[1,7,64,257])
    p.add_argument('--input-dim',type=int,default=1024)
    p.add_argument('--hidden',type=int,default=256)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError('Use a new validation output; existing evidence is not overwritten')
    if min(args.lengths+[args.input_dim,args.hidden])<1:raise ValueError('Dimensions and lengths must be positive')
    evidence={'software':software_identity(),'scope':'synthetic CUDA fused/reference outputs and gradients; no clinical metrics','checks':[]}
    if not torch.cuda.is_available():
        write_json(out,{**evidence,'state':'not_run','reason':'CUDA GPU is unavailable'})
        raise SystemExit(2)
    try:
        fused,identity=author_network(args.hidden,'cuda',64)
        reference,_=author_network(args.hidden,'torch',64)
        evidence['backend']=identity
        evidence['gpu']=torch.cuda.get_device_name(0)
        for profile in ['msmmil']:
            for length in args.lengths:
                torch.manual_seed(2026)
                a=fused(3,0.,'relu',n_features=args.input_dim,layer=2,rate=10,type='AFWMamba').cuda()
                b=reference(3,0.,'relu',n_features=args.input_dim,layer=2,rate=10,type='AFWMamba').cuda()
                b.load_state_dict(a.state_dict())
                x=torch.randn(length,args.input_dim,device='cuda');target=torch.tensor([1],device='cuda')
                outputs=[]
                for net in (a,b):
                    y=net(data=x[None])['logits'];outputs.append(y)
                    F.cross_entropy(y,target).backward()
                torch.testing.assert_close(*outputs,rtol=3e-4,atol=2e-5)
                gradient_max=0.
                for (name,left),(other,right) in zip(a.named_parameters(),b.named_parameters()):
                    if name!=other or left.grad is None or right.grad is None:raise AssertionError('Missing or reordered parameter gradient')
                    torch.testing.assert_close(left.grad,right.grad,rtol=2e-3,atol=5e-5,msg=lambda msg,name=name:name+': '+msg)
                    gradient_max=max(gradient_max,float((left.grad-right.grad).abs().max()))
                evidence['checks'].append({'profile':profile,'length':length,'state':'passed',
                    'logit_max_abs':float((outputs[0]-outputs[1]).abs().max()),'gradient_max_abs':gradient_max})
                write_json(out,{**evidence,'state':'running'})
        write_json(out,{**evidence,'state':'passed'})
    except Exception as error:
        write_json(out,{**evidence,'state':'failed','error':repr(error)})
        raise
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
