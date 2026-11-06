"""Full source loops, all phase states and genuine interrupted native runs."""
import copy
import logging
import os
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from milplus.data import Bundle
from milplus.engine import evaluate, rng_state, seed_everything, train
from milplus.io import read_json
from milplus.lrmil import LRMIL, PatchExamples
from milplus.lrmil_demo import create_lrmil_demo
from milplus.lrmil_source import native_backbone
from milplus.upstream import load_definitions


@pytest.fixture(autouse=True)
def runtime():
    torch.set_num_threads(1)


def identical(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            identical(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            identical(left, right)
    else:
        assert a == b


@pytest.fixture
def bundle(tmp_path):
    root, _ = create_lrmil_demo(tmp_path/'data', patients=18, low_patches=8)
    return Bundle(root)


def configuration(budgets):
    return {'epochs': sum(budgets), 'patience': 20, 'lr': .0001, 'weight_decay': 0.,
            'max_instances': None, 'grad_clip': None, 'selection_metric': 'lrmil_stage_selection'}


def author_scope(model, datasets, tmp_path, captured, monkeypatch):
    from milplus.lrmil_source import author_components
    authors, losses, helpers = author_components()
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    metrics = load_definitions('lrmil', 'utils/metrics.py', ['classification_metrics'],
        {'np': np, 'accuracy_score': accuracy_score, 'f1_score': f1_score, 'roc_auc_score': roc_auc_score})
    original_adam = torch.optim.Adam
    def capture_adam(*args, **kwargs):
        optimizer = original_adam(*args, **kwargs)
        captured['optimizer'] = optimizer
        return optimizer
    monkeypatch.setattr(torch.optim, 'Adam', capture_adam)
    def classifier(*args, **kwargs):
        result = authors.MILClassifier(*args, **kwargs)
        captured['model'] = result
        return result
    return {'torch': torch, 'logging': logging, 'os': os, 'DataLoader': DataLoader,
        'tqdm': lambda loader, **kwargs: loader, 'get_device': lambda cfg: 'cpu',
        'parse_folds': lambda cfg: [1], 'set_seed': seed_everything,
        'setup_logging': lambda *a, **k: None, 'classification_metrics': metrics.classification_metrics,
        'MILClassifier': classifier, 'TeacherMILDataset': datasets,
        **{key: getattr(losses, key) for key in ['ce_loss','patch_kd_loss','attention_weighted_feature_loss']},
        'build_teacher_guided_attention': helpers.build_teacher_guided_attention}


def test_actual_full_author_teacher_loop_weights_all_adam_and_rng_match(bundle, tmp_path, monkeypatch):
    seed_everything(75)
    model = LRMIL(bundle, stage_epochs=[2, 1, 1], patch_batch_size=4)
    model.configure_training(configuration([2, 1, 1]))
    model.prepare(bundle, 75, 'cpu')
    optimizers = model.make_optimizers(.0001, 0.)
    for epoch in range(2):
        model.start_epoch(epoch)
        model.epoch(bundle, optimizers, 'cpu')
        frame = model.predict_subset(bundle, 'val', 'cpu', {})
        model.after_validation(model.validation_metrics(frame))
    expected_rng = rng_state()
    class TeacherDataset(Dataset):
        def __init__(self, *args, split='train', **kwargs):
            self.rows = model.rows(split)
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            return {'E20': bundle.lrmil.high(row),
                'coords_20x': bundle.load(row, require_coords=True)[1],
                'label': torch.tensor(row['target']), 'wsi_name': row['slide_id'],
                'pid': row['global_patient_id']}
    captured = {}
    scope = author_scope(model, TeacherDataset, tmp_path, captured, monkeypatch)
    source = load_definitions('lrmil', 'utils/trainer.py',
        ['_num_classes','_dataset_name','validate_mil','save_teacher_meta','train_stage2_teacher'], scope)
    cfg = NS(experiment=NS(seed=75, output_dir=str(tmp_path/'author')),
        data=NS(dataset='synthetic', task='subtype', teacher_aggregated_root='unused', csv_path='unused'),
        training=NS(epochs=2, lr=.0001, patience=20), model=NS(embed_dim=768,num_classes=2,gated_attention=False))
    source.train_stage2_teacher(cfg)
    identical(model.teacher.state_dict(), captured['model'].state_dict())
    identical(optimizers[0].state_dict(), captured['optimizer'].state_dict())
    identical(expected_rng, rng_state())
    assert model.updates['teacher'] == 2*len(model.rows('train'))


def test_actual_full_author_patch_trainer_native_encoder_all_adam_rng_match(bundle, tmp_path, monkeypatch):
    seed_everything(75)
    model = LRMIL(bundle, stage_epochs=[1, 1, 1], patch_batch_size=4)
    model.configure_training(configuration([1, 1, 1])); model.prepare(bundle,75,'cpu')
    optimizers = model.make_optimizers(.0001,0.)
    model.start_epoch(0); model.epoch(bundle,optimizers,'cpu')
    model.after_validation(model.validation_metrics(model.predict_subset(bundle,'val','cpu',{})))
    model.start_epoch(1); model.epoch(bundle,optimizers,'cpu'); model.finish_epoch(bundle,'cpu',{})
    expected_rng = rng_state()
    rows = model.rows('train')
    metadata = {row['slide_id']:model.metadata(row,'cpu') for row in rows}
    class PatchDataset(PatchExamples):
        def __init__(self,*args,**kwargs):
            super().__init__(bundle.lrmil,rows,metadata)
    captured = {}
    scope = author_scope(model,None,tmp_path,captured,monkeypatch)
    def factory(config):
        result = native_backbone(bundle.lrmil.meta['checkpoint'])
        captured['model'] = result
        return result,None
    scope.update(load_conch_model=factory, StudentBackbone=lambda conch,**kwargs: conch,
                 PatchKDDataset=PatchDataset, _collate_list=lambda batch:batch)
    source = load_definitions('lrmil','utils/trainer.py',['_dataset_name','train_stage1'],scope)
    cfg=NS(experiment=NS(seed=75,output_dir=str(tmp_path/'author')),data=NS(dataset='synthetic',
        task='subtype',lr_patch_root='unused',teacher_aggregated_root='unused',teacher_mil_root='unused',csv_path='unused'),
        training=NS(lr=.0001,epochs=1,batch_size=4,num_workers=0),
        stage1=NS(kd_layers=[0,1,10],lambda_layer=1.,lambda_attn=1.))
    source.train_stage1(cfg)
    identical(model.encoder.visual.state_dict(),captured['model'].visual.state_dict())
    identical(optimizers[1].state_dict(),captured['optimizer'].state_dict())
    identical(expected_rng,rng_state())
    assert model.updates['patch'] == len(PatchDataset())//4


def test_genuine_interruptions_at_all_three_boundaries_match_continuous_run(bundle,tmp_path):
    cfg={'threads':4,'model':{'stage_epochs':[1,1,2],'patch_batch_size':4,'inference_batch_size':4}}
    full=train(bundle.root,'LRMIL',tmp_path/'full',seed=2026,config=cfg)
    partial=train(bundle.root,'LRMIL',tmp_path/'partial',seed=2026,config=cfg,stop_after=1)
    assert not (partial/'best.pt').exists()
    for boundary in [2,3]:
        train(bundle.root,'LRMIL',partial,seed=2026,config=cfg,resume=True,stop_after=boundary)
        if boundary==2:
            assert not (partial/'best.pt').exists()
        assert not (partial/'test_predictions.csv').exists()
    train(bundle.root,'LRMIL',partial,seed=2026,config=cfg,resume=True)
    left,right=[torch.load(p/'last.pt',weights_only=False,map_location='cpu') for p in [full,partial]]
    for key in ['model','optimizers','rng','history','best','bad']:
        identical(left[key],right[key])
    extra=left['model']['_extra_state']
    assert extra['phase']=='student' and extra['training_done']
    assert all(value>0 for value in extra['updates'].values())
    assert len(left['optimizers'])==3 and len(extra['selection_history'])==4
    evaluate(full);first=pd.read_csv(full/'test_predictions.csv')
    evaluate(full);pd.testing.assert_frame_equal(first,pd.read_csv(full/'test_predictions.csv'))
    for mutation in ['updates','student_instance_updates','phase','train_orders','best_encoder_hash','run_seed','last_patch_loss']:
        state=copy.deepcopy(extra)
        if mutation=='updates':
            state['updates']['patch']-=1
        elif mutation=='student_instance_updates':
            state[mutation]-=1
        elif mutation=='phase':
            state[mutation]='patch'
        elif mutation=='train_orders':
            state[mutation][0]=state[mutation][0][:-1]
        elif mutation=='best_encoder_hash':
            state[mutation]='changed'
        elif mutation=='run_seed':
            state[mutation]+=1
        else:
            state[mutation]+=1
        model=LRMIL(bundle,stage_epochs=[1,1,2],patch_batch_size=4,inference_batch_size=4)
        model.configure_training(configuration([1,1,2]));model.prepare(bundle,2026,'cpu')
        with pytest.raises(ValueError):
            model.set_extra_state(state)
        del model


def test_no_test_guidance_before_explicit_evaluation(bundle,tmp_path):
    model=LRMIL(bundle,stage_epochs=[1,1,1])
    with pytest.raises(ValueError,match='TRAIN-only'):
        model.metadata(model.rows('test')[0],'cpu')
    with pytest.raises(ValueError):
        model.predict_subset(bundle,'test','cpu',{})


@pytest.mark.parametrize('override',[{'grad_clip':1.},{'max_instances':8},
                                    {'epochs':2},{'weight_decay':.01},
                                    {'selection_metric':'macro_auc'}])
def test_full_source_recipe_refuses_partial_pipeline_or_objective_override(bundle,override):
    model=LRMIL(bundle,stage_epochs=[1,1,1])
    with pytest.raises(ValueError):
        model.configure_training({**configuration([1,1,1]),**override})


def test_real_multiclass_three_phase_run_and_low_only_prediction(tmp_path,monkeypatch):
    root,_=create_lrmil_demo(tmp_path/'multi',patients=27,classes=3,low_patches=8)
    run=train(root,'LRMIL',tmp_path/'run',seed=75,
        config={'threads':4,'model':{'stage_epochs':[1,1,1],'patch_batch_size':4,'inference_batch_size':4}})
    def reject_teacher(*args,**kwargs):
        raise AssertionError('LR student inference must not call the HR teacher')
    monkeypatch.setattr('milplus.lrmil.teacher_metadata',reject_teacher)
    evaluate(run)
    frame=pd.read_csv(run/'test_predictions.csv')
    assert {'p_0','p_1','p_2'}<=set(frame) and set(frame.target)=={0,1,2}
    protocol=read_json(run/'method_protocol.json')
    assert protocol['actual_student_instance_updates']>0
