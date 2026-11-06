"""Execute full synthetic SMMILe CUDA routes when hardware is actually present."""
import argparse
from pathlib import Path

import torch

from milplus.engine import evaluate, train
from milplus.io import write_json
from milplus.provenance import software_identity
from milplus.smmile_demo import create_smmile_demo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    identity=software_identity()
    if not torch.cuda.is_available():
        result={'state':'not_run','reason':'CUDA hardware/runtime unavailable','software':identity}
        write_json(output/'validation.json',result)
        print('not_run: CUDA hardware/runtime unavailable')
        return
    checks=[]
    for variant,method,dimension in [('binary','smmile_binary',512),('multiclass','smmile',1024),('multilabel','smmile_multilabel',1536)]:
        data=create_smmile_demo(output/(variant+'_data'),variant,dimension=dimension)
        run=train(data,method,output/method,device='cuda',config={'epochs':3,'threads':1,'model':{'warmup_epochs':1,'groups':2}})
        first=evaluate(run,device='cuda');second=evaluate(run,device='cuda')
        if first!=second:
            raise ValueError('CUDA SMMILe repeated metrics differ')
        checks.append({'method':method,'dimension':dimension,'state':'passed','metrics':first})
    if software_identity()!=identity:
        raise ValueError('SMMILe runtime source changed during CUDA verification')
    write_json(output/'validation.json',{'state':'passed','software':identity,'device':torch.cuda.get_device_name(0),
               'checks':checks,'scope':'synthetic CUDA source-recipe verification; no WSI/paper-score claim'})


if __name__=='__main__':main()
