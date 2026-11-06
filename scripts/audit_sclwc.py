"""Independent diagnostic of pinned original SCL-WC; not an adapter."""
import argparse
import ast
import contextlib
import copy
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from milplus.io import write_json
from milplus.provenance import software_identity
from milplus.upstream import pinned_text, verified_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    ROOT, COMMIT = verified_source('sclwc')
    assert COMMIT == 'd8d0ff44e887cadeeaff199cd771af9ee23e5855'
    torch.set_num_threads(1)
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'random': random}


    def load(path, names):
        source = pinned_text(ROOT, COMMIT, path)
        tree = ast.parse(source)
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/path), 'exec'), namespace)  # noqa: S102 -- definitions from the verified pinned Git object


    load('utils/utils.py', ['initialize_weights'])
    load('utils/memory.py', ['Memory'])
    load('utils/loss.py', ['SupConLoss'])
    load('models/public_models/SCL.py', ['Att_Head', 'Attn_Net', 'Attn_Net_Gated', 'CLAM_SB', 'SCL'])
    random.seed(2026)
    torch.manual_seed(2026)
    with contextlib.redirect_stdout(io.StringIO()):
        initial = namespace['SCL'](size_arg='small', n_classes=2, pos_num=256, neg_num=256, hard_neg_num=256)
    features = torch.randn(60, 768)
    records = []
    for label in [0, 1]:
        for epoch in [0, 5]:
            model = copy.deepcopy(initial).train()
            random.seed(2026)
            torch.manual_seed(2026)
            result = model(features, label=torch.tensor([label]), instance_eval=True, epoch=epoch)
            contra = result['contra_loss']
            loss = .8*F.cross_entropy(result['bag_logits'], torch.tensor([label]))+.2*result['inst_loss']+.01*contra
            loss.backward()
            assert all(p.grad is None for p in model.att_head.parameters())
            assert not isinstance(contra, torch.Tensor) or not contra.requires_grad
            records.append({'label': label, 'epoch': epoch, 'contrastive': float(contra),
                            'contrastive_requires_grad': getattr(contra, 'requires_grad', False),
                            'pnm_head_parameters_with_gradient': sum(p.grad is not None for p in model.att_head.parameters()),
                            'network_parameters_with_gradient': sum(p.grad is not None for p in model.parameters()),
                            'state': 'diagnostic_confirmed'})
    record = {'source': str(ROOT), 'commit': COMMIT, 'cases': records,
              'scope': 'Original CPU synthetic diagnostic only. No method admitted or full recipe asserted.',
              'findings': ['Published trainer omits epoch so WSCL is never enabled.',
                            'Even explicitly enabled WSCL has no current-network gradient.',
                            'Class-agnostic PNM head exists but is never called in the released model forward.',
                            'Paper Eq.4 and Eq.6 printed signs conflict with cross-entropy prose; reconstruction needs explicit interpretation.']}
    output = Path(args.output).resolve()
    assert not output.exists(), 'Use a fresh audit receipt path'
    record.update(software=software_identity(), seed=2026, diagnostic_queue_capacity=256, state='confirmed_incomplete_source_recipe')
    write_json(output, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
