"""Compare every original TCGA draw and seed with unmodified author inference.

Use the actual isolated Python3.11 environment. Every released parameter is
loaded; all 90 context and 10 query bags and 25 UNI2 channels are retained.
This validates author-provided feature inference, not raw-WSI extraction.
"""
import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from milplus.icmil_context_native import identity,predict
from milplus.icmil_context_source import ARCH,WEIGHTS,checked_source,checkpoint
from milplus.io import digest_file,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);a=parser.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    source=checked_source();sys.path.insert(0,str(source))
    from icmil import load_icmil
    torch.set_num_threads(2);native=identity();rows=[]
    dataset=source/'datasets/tcga_uni2_luad_vs_lusc.h5'
    try:
        with h5py.File(dataset) as f:
            splits=sorted(k for k in f if k.startswith('split_'));assert len(splits)==50
            for seed in WEIGHTS:
                weights=checkpoint(seed);original=load_icmil(source=weights,seed=seed,device='cpu')
                assert sum(p.numel() for p in original.parameters())==8257552
                for split in splits:
                    folder=out/seed/split;folder.mkdir(parents=True)
                    g=f[split];x=g['X_train'][:].astype(np.float32);y=g['y_train'][:].astype(np.int64)
                    q=g['X_test'][:].astype(np.float32);truth=g['y_test'][:].astype(np.int64)
                    assert x.shape==(90,10,25) and q.shape==(10,10,25)
                    train_ids=g['train_bag_ids'][:];test_ids=g['test_bag_ids'][:]
                    assert len(train_ids)==90 and len(test_ids)==10
                    assert not set(train_ids)&set(test_ids)
                    np.savez(folder/'support.npz',features=x,targets=y);np.save(folder/'queries.npy',q,allow_pickle=False)
                    call={'native_runtime':native,'weights':str(weights),'weights_sha256':WEIGHTS[seed],
                        'support':str(folder/'support.npz'),'support_sha256':digest_file(folder/'support.npz'),
                        'queries':str(folder/'queries.npy'),'queries_sha256':digest_file(folder/'queries.npy'),
                        'threads':2,'device':'cpu','model_seed':seed,'output':str(folder/'logits.npy')}
                    write_json(folder/'native_call.json',call);receipt=predict(call)
                    with torch.no_grad():
                        literal=original(torch.from_numpy(x).unsqueeze(0),torch.from_numpy(y).unsqueeze(0),
                            torch.from_numpy(q).unsqueeze(0))
                    native_logits=np.load(folder/'logits.npy',allow_pickle=False)
                    assert torch.equal(literal.squeeze(0),torch.from_numpy(native_logits)),(seed,split,'literal source logits differ')
                    probs=np.load(folder/'logits.probabilities.npy',allow_pickle=False)
                    assert torch.equal(torch.softmax(literal.squeeze(0),-1),torch.from_numpy(probs))
                    rows.append({'seed':seed,'split':split,'source_logits_bitwise_equal':True,
                        'context_shape':list(x.shape),'query_shape':list(q.shape),'full_parameters':8257552,
                        'downstream_optimizer_updates':0,'roc_auc':float(roc_auc_score(truth,probs[:,1])),
                        'patient_disjoint':True,'receipt':str(folder/'logits.receipt.json')})
                    write_json(out/'progress.json',{'state':'running','completed_conditions':len(rows),'last':rows[-1]})
                    print(seed,split,'exact',rows[-1]['roc_auc'],flush=True)
        assert len(rows)==150
        result={'state':'passed','conditions':rows,'total_conditions':150,'architecture':ARCH,
            'dataset_sha256':digest_file(dataset),'native_runtime':native,
            'seed_mean_auc':{seed:float(np.mean([r['roc_auc'] for r in rows if r['seed']==seed])) for seed in WEIGHTS},
            'scope':'all original three pretrained seeds and all fifty supplied overlapping TCGA feature draws; not new WSI or new TRIDENT extraction',
            'formal_admission':False}
        write_json(out/'validation.json',result)
    except BaseException as error:
        write_json(out/'validation.json',{'state':'failed','completed_conditions':rows,'error':repr(error)});raise


if __name__=='__main__':main()
