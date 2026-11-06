"""Full original trainer, old BN/preprocessing and mathematical loss references."""
import ast
import copy
import random
import time
from types import SimpleNamespace

import cv2
import faiss
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import normalized_mutual_info_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from milplus.c2c import C2C
from milplus.c2c_assets import components, default_root
from milplus.c2c_demo import create_c2c_demo
from milplus.c2c_source import build_model, definitions, objective, ToTensor
from milplus.data import Bundle
from milplus.engine import seed_everything, rng_state, restore_rng
from milplus.upstream import load_definitions


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b): equal(left, right)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    else:
        assert a == b


def test_full_original_two_epoch_trainer_matches_all_network_buffers_adam_and_rng(tmp_path):
    torch.set_num_threads(2); faiss.omp_set_num_threads(2)
    root = create_c2c_demo(tmp_path/'data', patients=10, patches=64, patch_pixels=64)
    bundle = Bundle(root); seed_everything(12)
    model = C2C(bundle, 'c2c_cluster_source')
    cfg = dict(epochs=2, lr=1e-4, weight_decay=0., threads=2, selection_metric='c2c_val_accuracy',
               max_instances=None, grad_clip=None)
    model.configure_training(cfg); optimizer = model.make_optimizer(cfg['lr'], cfg['weight_decay'])
    original_model = copy.deepcopy(model.network)
    original_optimizer = torch.optim.Adam(original_model.parameters(), lr=cfg['lr'])
    rows = []
    # Use the ORIGINAL OpenCV-reading datasets and original public train loop.
    # PNG round trip preserves every RGB byte in the separately packed archive.
    for row in bundle.rows.to_dict('records'):
        if row['subset'] == 'test': continue
        folder = tmp_path/'original_png'/row['slide_id']; folder.mkdir(parents=True)
        with bundle.c2c.open(row) as pixels:
            for index in range(len(pixels)):
                path = folder/f'{index:04d}.png'
                assert cv2.imwrite(str(path), cv2.cvtColor(pixels[index], cv2.COLOR_RGB2BGR))
                rows.append({'wsi': row['slide_id'], 'path': str(path), 'label': row['target'],
                             'is_valid': int(row['subset'] == 'val')})
    class FixedSourceTransform:
        def __call__(self, image):
            return {'image': model.transform(image)}
    transform = FixedSourceTransform()
    namespace = dict(torch=torch, nn=nn, F=F, np=np, pd=pd, copy=copy, time=time,
        random=random, Dataset=Dataset, DataLoader=DataLoader, cv2=cv2, tqdm=tqdm,
        normalized_mutual_info_score=normalized_mutual_info_score)
    original_data = load_definitions('c2c', 'C2C/dataloader.py',
        ['WSIDataloader', 'WSIClusterDataloader', 'reinitialize_dataloader'], namespace)
    namespace.update(WSIDataloader=original_data.WSIDataloader,
        WSIClusterDataloader=original_data.WSIClusterDataloader,
        reinitialize_dataloader=original_data.reinitialize_dataloader,
        Enc=model.source.Enc, EncAttn=model.source.EncAttn, PatchClassifier=model.source.PatchClassifier)
    from sklearn.preprocessing import normalize
    namespace.update(faiss=faiss, normalize=normalize)
    original_cluster = load_definitions('c2c', 'C2C/cluster.py',
        ['get_representation', 'cluster_representation', 'run_clustering'], namespace,
        replacements=[('input_image.cuda()', 'input_image.to(next(enc.parameters()).device)'),
                      ('enc = enc.cuda()', 'enc = enc.to(next(model_base.parameters()).device)')])
    original_predict = load_definitions('c2c', 'C2C/eval_model.py',
        ['compute_attn_df', 'eval_model'], namespace,
        replacements=[('sample[0].cuda()', 'sample[0].to(next(model.parameters()).device)')])
    from milplus.c2c_source import definitions
    source, criterion, _, _ = definitions(bundle.c2c.assets)
    utilities = load_definitions('c2c', 'C2C/utils.py', ['save_ckp', 'cal_nmi'], namespace)
    namespace.update(run_clustering=original_cluster.run_clustering, eval_model=original_predict.eval_model,
                     save_ckp=utilities.save_ckp, cal_nmi=utilities.cal_nmi)
    original_train = load_definitions('c2c', 'C2C/train.py', ['train_model'], namespace,
        replacements=[('inputs = inputs.cuda()', 'inputs = inputs.to(next(model.parameters()).device)'),
                      ('labels = labels.cuda()', 'labels = labels.to(next(model.parameters()).device)'),
                      ('return model', 'return model, model_final')])
    before = rng_state()
    selected, last = original_train.train_model(original_model,
        {'CE': nn.CrossEntropyLoss(), 'KLD': criterion}, original_optimizer, pd.DataFrame(rows),
        data_transforms=transform, alpha=1., beta=.01, gamma=.01, num_cluster=8,
        num_img_per_cluster=8, num_epochs=2, fpath=str(tmp_path/'original_best.pt'), topk=False)
    expected_rng = rng_state()
    restore_rng(before); validations, best_states = [], []
    for epoch in range(2):
        model.start_epoch(epoch); model.epoch(bundle, optimizer, 'cpu')
        frame = model.predict_subset(bundle, 'val', 'cpu', cfg)
        accuracy = model.validation_metrics(frame)['c2c_val_accuracy']; validations.append(accuracy)
        best_states.append(copy.deepcopy(model.network.state_dict()))
    equal(last.state_dict(), model.network.state_dict())
    equal(original_optimizer.state_dict(), optimizer.state_dict())
    equal(expected_rng, rng_state())
    best_epoch = max(i for i, value in enumerate(validations) if value == max(validations))
    equal(selected.state_dict(), best_states[best_epoch])
    assert model.optimizer_updates == 2*len(bundle.records('train'))
    assert len(model.cluster_history) == 2 and model.cluster_history[1]['nmi'] is not None


@pytest.mark.parametrize('tracking,training', [(False, False), (False, True), (True, False), (True, True)])
def test_original_torch151_bn_outputs_buffer_updates_and_counter_semantics(tracking, training):
    source_bn, _, _ = components(default_root())
    seed_everything(12)
    actual = source_bn(3); actual.track_running_stats = tracking; actual.train(training)
    actual.running_mean.copy_(torch.tensor([.2, -.5, 1.]))
    actual.running_var.copy_(torch.tensor([.7, 1.2, 2.]))
    means, variances = actual.running_mean.clone(), actual.running_var.clone()
    x = torch.randn(4, 3, 7, 5)
    expected = F.batch_norm(x, means, variances, actual.weight, actual.bias,
                           training or not tracking, .1, actual.eps)
    output = actual(x)
    assert torch.equal(output, expected)
    assert torch.equal(actual.running_mean, means) and torch.equal(actual.running_var, variances)
    assert int(actual.num_batches_tracked) == int(training and tracking)
    if not tracking:
        modern = nn.BatchNorm2d(3); modern.load_state_dict(actual.state_dict()); modern.track_running_stats = False; modern.eval()
        assert not torch.allclose(modern(x), output, atol=1e-5, rtol=1e-5)


def test_original_albumentations_compose_tensor_pixels_and_python_rng():
    # Execute the two actual historical __call__ methods without importing
    # irrelevant old augmentation plugins. Fixed p1 and no bbox processors are
    # exactly the author's public Compose([ToTensor()]) configuration.
    tree = ast.parse((default_root()/'albumentations_v046_composition.py').read_text())
    compose = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Compose')
    call = copy.deepcopy(next(n for n in compose.body if isinstance(n, ast.FunctionDef) and n.name == '__call__'))
    call.name = 'source_compose'
    tree = ast.parse((default_root()/'albumentations_v046_transforms.py').read_text())
    tensor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ToTensor')
    tensor_call = copy.deepcopy(next(n for n in tensor.body if isinstance(n, ast.FunctionDef) and n.name == '__call__'))
    tensor_call.name = 'source_tensor'
    _, _, convert = components(default_root())
    namespace = {'random': random, 'img_to_tensor': convert}
    exec(compile(ast.Module(body=[call, tensor_call], type_ignores=[]), '<actual_albumentations046_calls>', 'exec'), namespace)
    class ActualSourceTensor:
        _additional_targets = {}
        normalize = None
        __call__ = namespace['source_tensor']
    config = SimpleNamespace(p=1., processors={}, transforms=[ActualSourceTensor()])
    image = np.arange(12*13*3, dtype=np.uint8).reshape(12, 13, 3)
    seed_everything(12); state = random.getstate()
    expected = namespace['source_compose'](config, image=image)['image']; end = random.getstate()
    random.setstate(state); actual = ToTensor(default_root())(image)
    assert torch.equal(actual, expected) and random.getstate() == end
    assert actual.dtype == torch.float32 and actual.shape == (3, 12, 13)


def test_original_cluster_kl_formula_skip_threshold_and_all_gradients():
    _, criterion, _, _ = definitions(default_root())
    attention = torch.tensor([[.1], [.2], [-.4], [.8], [.3], [.7], [.4], [.2], [.6], [-.1]], requires_grad=True)
    cluster = np.array([0]*5+[1]*5)
    actual = criterion(attention, cluster)
    other = attention.detach().clone().requires_grad_()
    expected = sum(torch.logsumexp(group.flatten(), 0)-group.mean()-np.log(5.) for group in [other[:5], other[5:]])
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)
    torch.testing.assert_close(torch.autograd.grad(actual, attention)[0], torch.autograd.grad(expected, other)[0])
    assert criterion(attention[:4], np.zeros(4, dtype=int)) == 0
    logits = torch.tensor([[.3, -.7, .4]], requires_grad=True)
    patch = torch.randn(10, 3, requires_grad=True)
    with pytest.raises(ValueError, match='binary'):
        objective(logits, patch, attention, torch.tensor([2]), cluster, criterion, repaired=False)
    values = objective(logits, patch, attention, torch.tensor([2]), cluster, criterion, repaired=True)
    assert torch.equal(values['patch_ce'], F.cross_entropy(patch, torch.full((10,), 2)))
    values['loss'].backward(); assert patch.grad is not None and attention.grad is not None and logits.grad is not None


def test_no_silent_checkpoint_or_frozen_feature_substitution():
    with pytest.raises(ValueError, match='binary'):
        build_model(default_root(), 3, 'c2c_cluster_source')
