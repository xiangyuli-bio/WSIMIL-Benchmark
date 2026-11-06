"""Check the author's cached BLCA example against published notebook outputs.

This is inference on an author-provided cached example, not a natural-WSI
training run, independently extracted feature validation or paper-score claim.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from milplus.io import digest_file, write_json
from milplus.vlsa_source import COMMIT, build_network, fixed


def binary(root, path):
    expected = subprocess.check_output(['git', '-C', str(root), 'rev-parse',
                                       COMMIT+':'+path], text=True).strip()
    file = root/path
    content = file.read_bytes()
    actual = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    if actual != expected:
        raise ValueError('Published author binary differs from its fixed Git object: '+path)
    return file


def validate(weights, output, device='cpu',legacy_tf32=False):
    root, _ = fixed()
    torch.set_num_threads(2)
    if legacy_tf32:
        if not str(device).startswith('cuda'):
            raise ValueError('Legacy TF32 diagnostic is a separately named CUDA-only precision condition')
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    checkpoint = binary(root, 'assert/blca-train-VLSA/train_model-last.pth')
    features = binary(root, 'assert/blca-test-WSI-TCGA-XF-A9ST.pt')
    notebook = binary(root, 'notebook/VLSA-Walkthrough.ipynb')
    # Cell12's retained author output has four decimal places. The tolerance
    # permits only its final rounding plus float32 backend roundoff.
    expected = np.array([.4904,.2488,.1200,.0419,.0143,.0079,
                         .0047,.0031,.0038,.0071,.0167,.0412])
    notebook_data = json.loads(notebook.read_text())
    outputs = json.dumps(notebook_data['cells'][12].get('outputs', []))
    if '0.4904' not in outputs or '0.2488' not in outputs:
        raise ValueError('Author notebook retained reference output changed')
    model = build_network(weights, root/'tools/survival_prompts.json',
                          root/'tools/survival_text_prototypes.json', 'tcga_blca_0', 12)
    learned = {n:p for n,p in model.named_parameters() if p.requires_grad}
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
    if set(saved) != set(learned):
        raise ValueError('Published example checkpoint differs from the exact six learned parameter groups')
    with torch.no_grad():
        for name, value in saved.items():
            parameter = learned[name]
            if parameter.shape != value.shape or parameter.dtype != value.dtype or not torch.isfinite(value).all():
                raise ValueError('Published example parameter shape/dtype/values differ: '+name)
            parameter.copy_(value)
    model.to(device).eval()
    x = torch.load(features, map_location='cpu', weights_only=True)
    if x.shape != (2798,512) or not torch.isfinite(x).all():
        raise ValueError('Published example native feature tensor changed')
    with torch.no_grad():
        logits, *_ = model(x.to(device)[None])
        probabilities = logits.softmax(-1).cpu().numpy()[0]
    max_error = float(np.max(np.abs(probabilities-expected)))
    passed=bool(np.allclose(probabilities, expected, atol=6e-5, rtol=0))
    receipt = {'state':'passed' if passed else 'failed', 'source_commit':COMMIT, 'device':device,
        'legacy_tf32_diagnostic':legacy_tf32,
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'torch_version':torch.__version__,
        'scope':'Author cached2798x512 BLCA example inference only; no natural WSI extraction/training or paper-score reproduction',
        'learned_shapes':{n:list(p.shape) for n,p in learned.items()},
        'weights_sha256':digest_file(weights),
        'author_binaries':{str(p.relative_to(root)):digest_file(p) for p in [checkpoint,features,notebook]},
        'published_rounded_probabilities':expected.tolist(), 'actual_probabilities':probabilities.tolist(),
        'atol':6e-5, 'max_absolute_error':max_error}
    write_json(output,receipt)
    if not passed:
        raise AssertionError({'actual':probabilities.tolist(), 'expected':expected.tolist(),
                              'max_absolute_error':max_error,'receipt':str(output)})
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--legacy-tf32',action='store_true',help='Separate diagnostic precision condition, never silently enabled for training')
    args=parser.parse_args()
    receipt=validate(args.weights,Path(args.output),args.device,args.legacy_tf32)
    print(json.dumps({'state':receipt['state'],'max_absolute_error':receipt['max_absolute_error']}))
