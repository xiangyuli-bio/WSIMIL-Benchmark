"""Reproducible training, isolated validation selection and explicit test evaluation."""
import copy
import importlib.metadata
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import __version__
from .data import Bundle
from .io import atomic_torch_save, digest_object, read_json, write_json
from .metrics import classification_metrics, harrell_c_index
from .models import build_bundle_model
from .numerics import initialize_cpu_vector_math
from .provenance import software_identity
from .recipes import classification_step, cox_breslow_loss, make_optimizers, probabilities

DEFAULTS = {"epochs": 50, "lr": 0.0002, "weight_decay": 0.00001, "patience": 10,
            "grad_clip": 5.0, "bag_weight": 0.7, "max_instances": None,
            "model": {}, "selection_metric": None, "threads": 2}


def seed_everything(seed):
    initialize_cpu_vector_math()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def units(bundle, subset):
    records = bundle.records(subset)
    patient = bundle.task_type == "survival" or bundle.config.get("training_unit", "slide") == "patient"
    if not patient:
        return [[r] for r in records]
    groups = {}
    for row in records:
        groups.setdefault(row["global_patient_id"], []).append(row)
    if bundle.task_type in {"classification", "ordinal"} and any(len({r["target"] for r in g}) != 1 for g in groups.values()):
        raise ValueError("Patient training requires consistent labels")
    if bundle.task_type == 'multilabel' and any(len({r['targets'] for r in g}) != 1 for g in groups.values()):
        raise ValueError('Patient training requires consistent multilabel targets')
    return list(groups.values())


def load_unit(bundle, rows, device, max_instances=None, train=False):
    bags = [bundle.load(row) for row in rows]
    x = torch.cat([a for a, _ in bags])
    # Native runnable methods do not need coords. Multi-slide coordinates have separate frames.
    coords = bags[0][1] if len(rows) == 1 else None
    if max_instances and len(x) > max_instances:
        idx = torch.randperm(len(x))[:max_instances] if train else torch.linspace(0, len(x)-1, max_instances).long()
        x = x[idx]
        coords = None if coords is None else coords[idx]
    return x.to(device), None if coords is None else coords.to(device)


def forward_unit(model, bundle, rows, device, config, train=False):
    if hasattr(model, 'forward_unit'):
        return model.forward_unit(bundle, rows, device, config, train)
    x, coords = load_unit(bundle, rows, device, config['max_instances'], train)
    return model(x, coords=coords)


@torch.no_grad()
def predict(model, bundle, method, subset, device, config, instance_output=None):
    model.eval()
    if hasattr(model, 'predict_subset'):
        if instance_output is not None:
            raise ValueError('This method has its own registered prediction export, without generic instance output')
        return model.predict_subset(bundle, subset, device, config)
    predictions = []
    artifacts = []
    if hasattr(model, 'start_prediction'):
        model.start_prediction(subset)
    for rows in units(bundle, subset):
        row = rows[0]
        result = forward_unit(model, bundle, rows, device, config)
        record = {"slide_id": row["slide_id"] if len(rows) == 1 else "patient:" + row["global_patient_id"],
                  "patient_id": row["patient_id"], "global_patient_id": row["global_patient_id"],
                  "subset": subset, "source_slides": ";".join(r["slide_id"] for r in rows)}
        if bundle.task_type == "survival":
            record.update(time=row["time"], event=row["event"], risk=float(result["logits"].squeeze()))
            for key in ('hazards', 'survival'):
                if key in result:
                    record.update({f'{key}_{i}': float(value) for i, value in enumerate(result[key][0])})
        elif bundle.task_type == 'regression':
            targets = bundle.regression_targets
            patient = row['global_patient_id']
            record['project'] = targets.projects.loc[patient]
            record.update({f'target_{i}': float(v) for i, v in enumerate(targets.tensor(patient))})
            for key in ['raw', 'prediction']:
                record.update({f'{key}_{i}': float(v) for i, v in enumerate(result[key][0])})
        elif bundle.task_type == 'multilabel':
            record.update({f'target_{i}': int(value) for i, value in enumerate(row['targets'])})
            record.update({f'p_{i}': float(value) for i, value in enumerate(probabilities(method, result)[0])})
        elif bundle.task_type == 'ordinal' and 'ordinal_logits' in result:
            record.update(target=row['target'], ordinal_score=float(result['ordinal_logits'].sigmoid().sum(1).item()),
                          prediction=int(result['ordinal_logits'].sigmoid().sum(1).round().item()))
            record.update({f'threshold_{i}': float(v) for i, v in enumerate(result['ordinal_logits'].sigmoid()[0])})
        else:
            record["target"] = row["target"]
            record.update({f"p_{i}": float(v) for i, v in enumerate(probabilities(method, result)[0])})
            for key in ('predictive_entropy', 'expected_entropy', 'mutual_information'):
                if key in result:
                    record[key] = float(result[key].squeeze())
        if hasattr(model, 'prediction_fields'):
            record.update(model.prediction_fields(result, row))
        if instance_output is not None:
            artifacts.append(model.export_instances(result, row, instance_output))
        predictions.append(record)
    if instance_output is not None:
        write_json(Path(instance_output)/'manifest.json', {'schema_version': 1, 'method': method, 'subset': subset,
            'synthetic': bool(bundle.config.get('synthetic', False)), 'bundle_sha256': bundle.fingerprint,
            'fm': bundle.fm, 'classes': bundle.classes, 'row_order': 'original feature/coordinate rows',
            'score_semantics': getattr(model, 'instance_score_semantics', 'class scores, normalized detection attention and their product are distinct; refinement order/background index are recorded in each asset'),
            'records': artifacts})
    return pd.DataFrame(predictions)


def score(bundle, predictions):
    if 'dspamil_decision' in predictions:
        from .dspamil_metrics import metrics
        return metrics(predictions, unit=bundle.config.get('evaluation_unit', 'slide'))
    if "mexd_stage" in predictions:
        from .mexd import source_and_uncertainty_metrics
        return {**score(bundle,predictions.drop(columns=["mexd_stage"])),**source_and_uncertainty_metrics(predictions)}
    if 'taxomil_fine_text' in predictions:
        from .taxomil import text_metrics
        return text_metrics(predictions)
    if bundle.task_type == 'multitask':
        from .multitask import multitask_metrics
        return multitask_metrics(bundle, predictions)
    if bundle.task_type == 'ordinal' and 'ordinal_score' in predictions:
        from .metrics import ordinal_metrics
        return ordinal_metrics(predictions, len(bundle.classes), bundle.config.get('evaluation_unit', 'slide'))
    if bundle.task_type == 'multilabel':
        from .multilabel import multilabel_metrics
        return multilabel_metrics(predictions, len(bundle.classes), bundle.config.get('evaluation_unit', 'slide'))
    if bundle.task_type == 'regression':
        from .regression import regression_metrics
        return regression_metrics(predictions, bundle.regression_targets.columns,
                                  bundle.config.get('evaluation_unit', 'patient'))
    if bundle.task_type == "survival":
        return {"c_index": harrell_c_index(predictions.time, predictions.event, predictions.risk),
                "unit": "patient", "n": len(predictions)}
    return classification_metrics(predictions, len(bundle.classes), bundle.config.get("evaluation_unit", "slide"))


def survival_epoch(model, bundle, training_units, optimizer, device, cfg):
    # Two-pass exact full-cohort risk gradient: graph memory bounded by one patient's bag.
    # Replaying RNG reproduces dropout so the recomputed risk matches the first pass.
    states, risks = [], []
    with torch.no_grad():
        for rows in training_units:
            states.append(rng_state())
            x, coords = load_unit(bundle, rows, device, cfg["max_instances"], train=True)
            risks.append(model(x, coords=coords)["logits"].flatten())
    after_first_pass = rng_state()
    r = torch.cat(risks).detach().requires_grad_(True)
    times = torch.tensor([g[0]["time"] for g in training_units], device=device)
    events = torch.tensor([g[0]["event"] for g in training_units], device=device)
    loss = cox_breslow_loss(r, times, events)
    gradient = torch.autograd.grad(loss, r)[0]
    optimizer.zero_grad(set_to_none=True)
    for i, rows in enumerate(training_units):
        restore_rng(states[i])
        x, coords = load_unit(bundle, rows, device, cfg["max_instances"], train=True)
        prediction = model(x, coords=coords)["logits"].flatten()
        prediction.backward(gradient[i:i+1])
    restore_rng(after_first_pass)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def train(data, method, output, fm=None, split=None, seed=2026, device="cpu", config=None,
          resume=False, stop_after=None):
    from .registry import check_compatibility, require_runnable, resolve_for_bundle
    require_runnable(method)
    bundle = Bundle(data, fm, split)
    spec = resolve_for_bundle(method, bundle)
    method = spec["id"]
    check_compatibility(spec, bundle)
    initialize_cpu_vector_math()
    if method.startswith('fouriermil_author_'):
        from .fouriermil_driver import train as fourier_train
        return fourier_train(data, method, output, fm=fm, split=split, seed=seed,
                             device=device, config=config, resume=resume, stop_after=stop_after)
    if method.startswith('fcmil_original_benchmark100_'):
        from .fcmil_driver import train as fcmil_train
        return fcmil_train(data, method, output, fm=fm, split=split, seed=seed,
                           device=device, config=config, resume=resume, stop_after=stop_after)
    if method.startswith('nnmil_author_'):
        from .nnmil_driver import train as nnmil_train
        return nnmil_train(data, method, output, fm=fm, split=split, seed=seed,
                           device=device, config=config, resume=resume, stop_after=stop_after)
    if method.startswith('mrblock_author_'):
        from .mrblock_driver import train as mrblock_train
        return mrblock_train(data, method, output, fm=fm, split=split, seed=seed,
                             device=device, config=config, resume=resume, stop_after=stop_after)
    if method.startswith('carmil_author_'):
        from .carmil_driver import train as carmil_train
        return carmil_train(data, method, output, fm=fm, split=split, seed=seed,
                            device=device, config=config, resume=resume, stop_after=stop_after)
    if method.startswith('attrimil_author_'):
        from .attrimil_driver import train as attrimil_train
        return attrimil_train(data, method, output, fm=fm, split=split, seed=seed,
                              device=device, config=config, resume=resume, stop_after=stop_after)
    if method.startswith('asmil_author_'):
        from .asmil_driver import train as asmil_train
        return asmil_train(data, method, output, fm=fm, split=split, seed=seed,
                           device=device, config=config, resume=resume, stop_after=stop_after)
    if method.startswith('pneamil_author_'):
        from .pneamil_driver import train as pneamil_train
        return pneamil_train(data, method, output, fm=fm, split=split, seed=seed,
                             device=device, config=config, resume=resume, stop_after=stop_after)
    if method.startswith('lnplmil_'):
        from .lnplmil_driver import train as lnpl_train
        return lnpl_train(data,output,method,fm,split,seed,device,config,resume,stop_after)
    if method.startswith('hybrid_supervision_'):
        from .hybrid_supervision_driver import train as hybrid_train
        return hybrid_train(data,output,method,fm,split,seed,device,config,resume,stop_after)
    if method.startswith('macnet_'):
        from .macnet_driver import train as mac_train
        return mac_train(data,method,output,fm=fm,split=split,seed=seed,device_name=device,config=config,resume=resume,stop_after=stop_after)
    if method.startswith('protomil_'):
        from .protomil_driver import train as proto_train
        return proto_train(data,output,fm=fm,split=split,seed=seed,device=device,config=config,resume=resume,stop_after=stop_after,method=method)
    if method.startswith('kmpmil_'):
        from .kmpmil_driver import train as kmp_train
        return kmp_train(data,output,fm=fm,split=split,seed=seed,device=device,config=config,resume=resume,stop_after=stop_after,method=method)
    if method.startswith('qpmilvl_'):
        from .qpmilvl_driver import train as qpmil_train
        return qpmil_train(data,output,fm=fm,split=split,seed=seed,device=device,config=config,resume=resume,stop_after=stop_after,method=method)
    if method.startswith('akdpmp_'):
        from .akdpmp_driver import train as akd_train
        return akd_train(data,method,output,fm=fm,split=split,seed=seed,device=device,config=config,resume=resume,stop_after=stop_after)
    if method.startswith('comel_'):
        from .comel_driver import train as comel_train
        return comel_train(data,method,output,fm=fm,split=split,seed=seed,device=device,config=config,resume=resume,stop_after=stop_after)
    if method=='pmil_prototypical_raw_patient_reference':
        from .pmil_prototypical_driver import train as pmil_train
        return pmil_train(data,method,output,fm=fm,split=split,seed=seed,device=device,config=config,resume=resume,stop_after=stop_after)
    if method.startswith('icmil_coupled_'):
        from .icmil_coupled_driver import train as coupled_train
        return coupled_train(data,method,output,fm=fm,split=split,seed=seed,device=device,
            config=config,resume=resume,stop_after=stop_after)
    if method.startswith('icmil_context_'):
        from .icmil_context_driver import train as context_train
        return context_train(data,method,output,fm=fm,split=split,seed=seed,device=device,
            config=config,resume=resume,stop_after=stop_after)
    if method.startswith('dspamil_'):
        from .dspamil_driver import train as dspa_train
        return dspa_train(data,method,output,fm=fm,split=split,seed=seed,device=device,config=config,resume=resume,stop_after=stop_after)
    if method.startswith('feather_'):
        from .feather_driver import train as feather_train
        return feather_train(data,method,output,fm=fm,split=split,seed=seed,device=device,config=config,resume=resume,stop_after=stop_after)
    if method.startswith('dyko_'):
        from .dyko_driver import train as dyko_train
        return dyko_train(data,method,output,fm=fm,split=split,seed=seed,device=device,config=config,resume=resume,stop_after=stop_after)
    if method.startswith('rmdl_'):
        from .rmdl_driver import train as rmdl_train
        return rmdl_train(data,method,output,seed=seed,device=device,config=config,
            resume=resume,stop_after=stop_after,fm=fm,split=split)
    if method.startswith('attention2majority_'):
        from .attention2majority_driver import train as native_train
        return native_train(data,method,output,seed=seed,device=device,config=config,
            resume=resume,stop_after=stop_after,fm=fm,split=split)
    cfg = copy.deepcopy(DEFAULTS)
    supplied = config or {}
    if set(supplied) - set(cfg):
        raise ValueError(f"Unknown training options: {set(supplied) - set(cfg)}")
    cfg.update(supplied)
    if method.startswith("mexd_"):
        for key,value in {"epochs":cfg["model"].get("moe_epochs",100)+cfg["model"].get("diffusion_epochs",200),"lr":2e-4,"weight_decay":1e-5,"grad_clip":1.,"selection_metric":"log_loss"}.items():
            if key not in supplied:cfg[key]=value
        if "patience"not in supplied:cfg["patience"]=cfg["epochs"]
    if method.startswith("lpgmil_"):
        for key,value in {"epochs":50,"lr":1e-4,"weight_decay":1e-5,"grad_clip":None,"selection_metric":"log_loss"}.items():
            if key not in supplied:cfg[key]=value
        if "patience"not in supplied:cfg["patience"]=cfg["epochs"]
    if method.startswith('promptmil_'):
        from .promptmil import RECIPES
        recipe=RECIPES[method]
        for key,value in {'epochs':recipe['epochs'],'lr':recipe['lr'],'weight_decay':recipe['weight_decay'],
                          'grad_clip':None,'selection_metric':recipe['selection']}.items():
            if key not in supplied:cfg[key]=value
        if 'patience'not in supplied:cfg['patience']=cfg['epochs']
    if method.startswith('iibmil_'):
        from .iibmil import CONDITIONS
        from .iibmil_source import source_config
        native = source_config(CONDITIONS[method])['TRAIN']
        for key, value in {'epochs':native['EPOCHS'], 'lr':native['LR'],
                'weight_decay':native['WEIGHT_DECAY'], 'grad_clip':native['CLIP_MAX_NORM'],
                'selection_metric':'macro_auc'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('nciemil_'):
        for key,value in {'epochs':100,'lr':2e-4,'weight_decay':1e-5,'grad_clip':None,
                'selection_metric':'nciemil_val_harmonic'}.items():
            if key not in supplied:cfg[key]=value
        if 'patience' not in supplied:cfg['patience']=cfg['epochs']
    if method.startswith('taxomil_'):
        for key,value in {'epochs':100,'lr':1e-4 if method=='taxomil_gpt2_paper_lr_key_repaired' else 1e-5,'weight_decay':.01,'patience':20,
                'grad_clip':None,'selection_metric':'taxomil_score'}.items():
            if key not in supplied:cfg[key]=value
    if method.startswith('mggmmil_'):
        for key,value in {'epochs':200,'lr':3e-5,'weight_decay':1e-5,'patience':20,
                          'grad_clip':None,'selection_metric':'mggmmil_val_negative_ce'}.items():
            if key not in supplied:cfg[key]=value
    if method.startswith('longmil_'):
        for key,value in {'epochs':200,'lr':.0001,'weight_decay':.01,'patience':20,
                          'grad_clip':None,'selection_metric':'c_index' if bundle.task_type=='survival' else 'macro_auc'}.items():
            if key not in supplied:cfg[key]=value
    if method.startswith('gmmamba_'):
        for key,value in {'epochs':100,'lr':.0001,'weight_decay':.00001,'patience':20,
                          'grad_clip':None,'selection_metric':'gmmamba_val_auc_f1'}.items():
            if key not in supplied:
                cfg[key]=value
    if method.startswith('muse_'):
        for key,value in {'epochs':200,'lr':.0001,'weight_decay':.00001,'patience':20,
                          'grad_clip':None,'selection_metric':'muse_val_error'}.items():
            if key not in supplied:
                cfg[key]=value
    if method.startswith('libramil_'):
        for key, value in {'epochs':20, 'lr':.0001, 'weight_decay':.01,
                          'grad_clip':None, 'selection_metric':'libramil_val_accuracy'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('m4_'):
        for key, value in {'epochs':32,'lr':.0001,'weight_decay':.00004,
                'grad_clip':None,'selection_metric':'m4_val_mean_auc'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('convl_'):
        for key, value in {'epochs':200, 'lr':.0002, 'weight_decay':.00001,
                'patience':8, 'grad_clip':None, 'selection_metric':'convl_val_nll'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('center_embeddings_'):
        for key, value in {'epochs':20, 'lr':.0001, 'weight_decay':.0001,
                'grad_clip':None, 'selection_metric':'center_embeddings_val_balanced_accuracy'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('miladapter_'):
        for key, value in {'epochs': 20, 'lr': .001, 'weight_decay': 1e-5,
                'grad_clip': None, 'selection_metric': 'miladapter_completed_epoch'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
        if 'sampling_seed' in cfg['model']:
            raise ValueError('MIL-Adapter support sampling seed is the run seed; do not override it in model options')
    if method.startswith('conmil_'):
        for key, value in {'epochs': 50, 'lr': .0002, 'weight_decay': 1e-5,
                'grad_clip': None, 'selection_metric': 'conmil_val_accuracy'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('sparseconvmil_raw_'):
        for key, value in {'epochs': 200, 'lr': .002, 'weight_decay': 1e-6,
                'grad_clip': None, 'selection_metric': 'sparseconvmil_completed_epoch'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('c2c_cluster_'):
        for key, value in {'epochs': 25, 'lr': .0001, 'weight_decay': 0.,
                'grad_clip': None, 'selection_metric': 'c2c_val_accuracy'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('sammil_'):
        for key, value in {'epochs': 200, 'lr': .0002, 'weight_decay': 1e-5,
                'patience': 20 if cfg['model'].get('recipe', 'tcga') == 'tcga' else 30,
                'grad_clip': None, 'selection_metric': 'sammil_val_auc'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('lrmil_'):
        budgets = cfg['model'].get('stage_epochs', [200, 200, 200])
        if not isinstance(budgets, (list, tuple)) or len(budgets) != 3 or any(type(v) is not int or v < 1 for v in budgets):
            raise ValueError('LRMIL must declare three complete positive phase budgets')
        for key, value in {'epochs': sum(budgets), 'lr': .0001, 'weight_decay': 0.,
                'patience': 20, 'grad_clip': None, 'selection_metric': 'lrmil_stage_selection'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('kmgmamba_'):
        for key, value in {'epochs': 100, 'lr': .0002, 'weight_decay': 1e-5,
                'patience': 20, 'grad_clip': None, 'selection_metric': 'kmgmamba_val_auc'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('ahdmil_'):
        from .ahdmil_source import RECIPES
        for key, value in {'epochs': 400, 'lr': RECIPES[method]['lr'], 'weight_decay': 1e-5,
                'patience': 20, 'grad_clip': None, 'selection_metric': 'ahdmil_source_ce'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('hdmil_'):
        from .hdmil_source import RECIPES
        for key, value in {'epochs': 400, 'lr': RECIPES[method]['lr'], 'weight_decay': 1e-5,
                'patience': 20, 'grad_clip': None, 'selection_metric': 'hdmil_source_ce'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('mgtrans_'):
        for key, value in {'epochs': 200, 'lr': .0001, 'weight_decay': 1e-5,
                'patience': 20, 'grad_clip': None, 'selection_metric': 'mgtrans_val_error'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('zoommil_'):
        for key, value in {'epochs': 100, 'lr': .0001, 'weight_decay': 0., 'grad_clip': None,
                'selection_metric': 'zoommil_source_ce' if method == 'zoommil_crc_source_repaired' else 'zoommil_weighted_f1'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('augdiff_'):
        for key, value in {'epochs': 200, 'lr': 2e-4, 'weight_decay': 1e-5,
                'patience': 10, 'grad_clip': None, 'selection_metric': 'augdiff_source_ce'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('vlsa_'):
        for key,value in {'epochs':10,'lr':2e-4,'weight_decay':1e-5,'grad_clip':None,
                          'selection_metric':'vlsa_final_epoch'}.items():
            if key not in supplied:cfg[key]=value
        if 'patience' not in supplied:cfg['patience']=cfg['epochs']
    if method.startswith('slip_'):
        for key,value in {'epochs':10,'lr':.002,'weight_decay':0.,'grad_clip':None,
                          'selection_metric':'slip_final_epoch'}.items():
            if key not in supplied:cfg[key]=value
        if 'patience' not in supplied:cfg['patience']=cfg['epochs']
    if method.startswith('dispro_'):
        budgets=cfg['model'].get('stage_epochs',[30,50,50])
        for key,value in {'epochs':sum(budgets),'lr':.0002,'weight_decay':.00001,
                'grad_clip':None,'selection_metric':'dispro_c_index'}.items():
            if key not in supplied:cfg[key]=value
        if 'patience' not in supplied:cfg['patience']=cfg['epochs']
    if method.startswith('amiml_'):
        for key, value in {'epochs':300, 'lr':.01 if method=='amiml_paper_reference' else .001,
                'weight_decay':0., 'patience':30, 'grad_clip':None,
                'selection_metric':'amiml_validation_generation'}.items():
            if key not in supplied: cfg[key]=value
    if method.startswith('dtmil_'):
        for key, value in {'epochs': 50, 'lr': .0002, 'weight_decay': .0001,
                           'grad_clip': .1, 'selection_metric': 'log_loss'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('pamil_policy_'):
        for key, value in {'epochs': 300, 'lr': .0001, 'weight_decay': 1e-5,
                           'grad_clip': None, 'selection_metric': 'macro_auc'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('focus_'):
        for key, value in {'epochs': 200, 'lr': .0001, 'weight_decay': 1e-5, 'patience': 20,
                           'grad_clip': None, 'selection_metric': 'focus_val_error'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('codmil_'):
        for key, value in {'epochs': 200, 'lr': .0001, 'weight_decay': 1e-5, 'patience': 20,
                           'grad_clip': None, 'selection_metric': 'codmil_val_loss'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('ins_frozen_'):
        for key, value in {'epochs': 150, 'lr': .01, 'weight_decay': .001,
                           'grad_clip': None, 'selection_metric': 'macro_auc'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('sclwc_'):
        for key, value in {'epochs': 100, 'lr': .0003 if method.endswith('source_control') else .001,
                           'weight_decay': 1e-5, 'grad_clip': None, 'patience': 20,
                           'selection_metric': 'macro_auc'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('deepgraphsurv_'):
        for key, value in {'epochs': 100, 'lr': .0002, 'weight_decay': 0., 'grad_clip': None,
                           'patience': 100, 'selection_metric': 'deepgraphsurv_val_patient_ci'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('triagemil_'):
        for key, value in {'epochs': 200, 'lr': 2e-4, 'weight_decay': 1e-4, 'grad_clip': None,
                           'patience': 40, 'selection_metric': 'triagemil_selection'}.items():
            if key not in supplied:cfg[key] = value
    if method.startswith('c2mil_'):
        for key, value in {'epochs': 100, 'lr': 1e-4, 'weight_decay': 5e-4, 'grad_clip': None,
                           'selection_metric': 'c2mil_val_c_index'}.items():
            if key not in supplied:cfg[key] = value
    if method.startswith('gatmamba_'):
        for key, value in {'epochs': 201, 'lr': 5e-5, 'weight_decay': 1e-4, 'grad_clip': None,
                           'patience': 5, 'selection_metric': 'gatmamba_val_loss'}.items():
            if key not in supplied:cfg[key] = value
    if method.startswith('protosurv_'):
        for key, value in {'epochs': 100, 'lr': 5e-5, 'weight_decay': 1e-5,
                           'grad_clip': None, 'patience': 15,
                           'selection_metric': 'protosurv_val_c_index'}.items():
            if key not in supplied:cfg[key] = value
    if method.startswith('pathrwkv_'):
        for key, value in {'epochs': 100, 'lr': 1e-4, 'weight_decay': .01,
                           'grad_clip': None, 'patience': 10,
                           'selection_metric': 'pathrwkv_val_loss'}.items():
            if key not in supplied:cfg[key] = value
    if method.startswith('c2aug_'):
        for key, value in {'epochs': 200, 'lr': 1e-4, 'weight_decay': 0., 'grad_clip': None,
                           'selection_metric': 'c2aug_val_macro_accuracy'}.items():
            if key not in supplied:cfg[key] = value
        if 'patience' not in supplied:cfg['patience'] = cfg['epochs']
    if method.startswith('agdrmil_'):
        for key,value in {'epochs':200,'lr':2e-4,'weight_decay':1e-5,'grad_clip':None,
            'patience':20,'selection_metric':'agdr_val_loss'}.items():
            if key not in supplied:cfg[key]=value
    if method.startswith('h2mil_'):
        for key,value in {'epochs':60,'lr':5e-5,'weight_decay':5e-4,'grad_clip':None,
            'selection_metric':'h2_val_auc'}.items():
            if key not in supplied:cfg[key]=value
        if 'patience' not in supplied:cfg['patience']=cfg['epochs']
    if method.startswith('hipt_lgp_'):
        for key,value in {'epochs':20,'lr':2e-4,'weight_decay':1e-5,'grad_clip':None,
                          'selection_metric':'hipt_val_loss'}.items():
            if key not in supplied:cfg[key]=value
        if 'patience' not in supplied:cfg['patience']=20 if cfg['model'].get('early_stopping',False) else cfg['epochs']
    if method.startswith('simil_'):
        for key,value in {'epochs':39,'lr':2e-4,'weight_decay':.005,'grad_clip':None,
                           'selection_metric':'simil_joint'}.items():
            if key not in supplied:cfg[key]=value
        if 'patience' not in supplied:cfg['patience']=cfg['epochs']
    if method.startswith('mamba2dmil_'):
        for key, value in {'epochs': 20, 'lr': 1e-4, 'weight_decay': 1e-5, 'grad_clip': None,
                           'selection_metric': 'c_index' if bundle.task_type == 'survival' else 'mamba2d_bag_auc'}.items():
            if key not in supplied:cfg[key] = value
        if 'patience' not in supplied:cfg['patience'] = cfg['epochs']
    if method.startswith('ttis_'):
        for key, value in {'epochs': 50, 'lr': 1e-4, 'weight_decay': .001, 'grad_clip': None,
                           'patience': 10, 'selection_metric': 'ttis_val_loss'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('msattention_'):
        for key, value in {'epochs': 10, 'lr': 1e-4, 'weight_decay': 0., 'grad_clip': None,
                           'selection_metric': 'log_loss'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('pmil_shapley_'):
        for key, value in {'epochs': cfg['model'].get('rounds', 10)*cfg['model'].get('round_epochs', 200),
                           'lr': 1e-3, 'weight_decay': 1e-5, 'grad_clip': None,
                           'patience': cfg['model'].get('patience', 20), 'selection_metric': 'pmil_selection'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('mammoth_'):
        for key, value in {'epochs': 20, 'lr': 1e-4, 'weight_decay': 1e-5, 'grad_clip': None,
                           'patience': 5, 'selection_metric': 'log_loss'}.items():
            if key not in supplied:
                cfg[key] = value
    if method == 'campanella_rnn':
        durations = [cfg['model'].get(k, 100) for k in ['cnn_epochs', 'rnn_epochs']]
        if any(type(v) is not int or v < 1 for v in durations):
            raise ValueError('Campanella phase durations must be positive integers')
        for key, value in {'epochs': sum(durations), 'lr': 1e-4, 'weight_decay': 1e-4, 'grad_clip': None,
                           'patience': sum(durations), 'selection_metric': 'campanella_balanced_accuracy'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('monai_'):
        for key, value in {'epochs': 50, 'lr': 3e-5, 'weight_decay': 0., 'grad_clip': None,
                           'patience': 50, 'selection_metric': 'qwk'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('stablemil_'):
        for key, value in {'epochs': 200, 'lr': 1e-4, 'weight_decay': 0., 'grad_clip': None,
                           'patience': 20, 'selection_metric': 'stablemil_loss'}.items():
            if key not in supplied:
                cfg[key] = value
    if method in {'setmil_source', 'setmil_repaired'}:
        for key, value in {'epochs': 100, 'lr': 2e-4, 'weight_decay': .05, 'grad_clip': .01,
                           'patience': 20, 'selection_metric': 'setmil_bag_auc'}.items():
            if key not in supplied:
                cfg[key] = value
    if method in {'gtp_source', 'gtp_repaired'}:
        for key, value in {'epochs': 120, 'lr': 1e-3, 'weight_decay': 5e-4, 'grad_clip': None,
                           'selection_metric': 'gtp_bag_accuracy'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('probsa_'):
        for key, value in {'epochs': 100, 'lr': 1e-4, 'weight_decay': 0., 'grad_clip': None,
                           'selection_metric': 'probsa_bag_auc'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method == 'micromil_frozen':
        for key, value in {'epochs': 50, 'lr': 1e-4, 'weight_decay': 1e-6, 'grad_clip': None,
                           'patience': 5, 'selection_metric': 'micromil_f1'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('loss_attention_'):
        classical = method == 'loss_attention_classical'
        temporal = method == 'loss_attention_temporal'
        for key, value in {'epochs': 200 if temporal else 100 if classical else 50,
                           'lr': .003 if temporal else .0001 if classical else .0005,
                           'weight_decay': 0. if temporal else .0005 if classical else .0001,
                           'grad_clip': None, 'selection_metric': 'loss_attention_error'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('pamil_prototype_'):
        for key, value in {'epochs': 350, 'lr': 1e-4, 'weight_decay': 1e-5, 'grad_clip': None,
                           'patience': 350, 'selection_metric': 'pamil_loss'}.items():
            if key not in supplied:
                cfg[key] = value
    if method in {'momil', 'momil_survival_extension'}:
        for key, value in {'epochs': 200, 'lr': 1e-4, 'weight_decay': 1e-5, 'grad_clip': None,
                           'selection_metric': 'c_index' if bundle.task_type == 'survival' else 'log_loss'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = 20 if cfg['model'].get('early_stopping', bundle.task_type == 'survival') else cfg['epochs']
    if method in {'smmile', 'smmile_binary', 'smmile_multilabel'}:
        defaults = {'epochs': 60, 'lr': 2e-5, 'weight_decay': 1e-5, 'grad_clip': None,
                    'patience': 50, 'selection_metric': 'smmile_source_loss'}
        for key, value in defaults.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('recomil_'):
        for key, value in {'epochs': 50, 'lr': 1e-4, 'weight_decay': 5e-4, 'grad_clip': 1.,
                           'patience': 10, 'selection_metric': 'reco_joint'}.items():
            if key not in supplied:
                cfg[key] = value
    if method in {'mammil', 'mammil_source', 'mammil_patient_union'}:
        defaults = {'epochs': 250, 'lr': 1e-4, 'weight_decay': .05, 'grad_clip': None,
                    'patience': 20, 'selection_metric': 'c_index' if bundle.task_type == 'survival' else 'log_loss'}
        for key, value in defaults.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('mfemil_'):
        survival = bundle.task_type == 'survival'
        defaults = {'epochs': 20 if survival else 200, 'lr': 2e-4 if survival else 1e-4,
                    'weight_decay': 1e-5 if survival else 0., 'grad_clip': None,
                    'patience': 5 if survival else 10,
                    'selection_metric': 'c_index' if survival else 'macro_auc'}
        for key, value in defaults.items():
            if key not in supplied:
                cfg[key] = value
    if method == 'msmmil':
        for key, value in {'epochs': 200, 'lr': 2e-4, 'weight_decay': 1e-5, 'grad_clip': None,
                           'patience': 40, 'selection_metric': 'log_loss'}.items():
            if key not in supplied:
                cfg[key] = value
    if method in {'dgmil', 'dgmil_source_extremes'}:
        for key, value in {'epochs': 50, 'lr': .01, 'weight_decay': 0., 'grad_clip': None, 'patience': 50}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('mambamil_'):
        for key, value in {'epochs': 200, 'lr': 2e-4, 'weight_decay': 1e-5, 'grad_clip': None, 'patience': 20,
                           'selection_metric': 'c_index' if bundle.task_type == 'survival' else 'log_loss'}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('pamoe_'):
        for key, value in {'epochs': 100, 'lr': 2e-4, 'weight_decay': 1e-2, 'grad_clip': None, 'patience': 5}.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('pathomics_'):
        for key, value in {'epochs': 50, 'lr': 1e-4, 'weight_decay': 0., 'grad_clip': None}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('mmp_'):
        for key, value in {'epochs': 50, 'lr': 1e-4, 'weight_decay': 1e-5, 'grad_clip': None, 'selection_metric': 'survival_loss'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method == 'pibd_source':
        for key, value in {'epochs': 30, 'lr': 5e-4, 'weight_decay': 0., 'grad_clip': None}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method.startswith('advmil_'):
        for key, value in {'epochs': 300, 'lr': 8e-5, 'weight_decay': 5e-4, 'grad_clip': None, 'patience': 30, 'selection_metric': 'survival_loss'}.items():
            if key not in supplied:
                cfg[key] = value
    if method == 'hmil_source_repaired':
        for key, value in {'epochs': 200, 'lr': 1e-4, 'weight_decay': 0., 'grad_clip': None,
                           'selection_metric': 'macro_auc'}.items():
            if key not in supplied:
                cfg[key] = value
        if 'patience' not in supplied:
            cfg['patience'] = cfg['epochs']
    if method in {'tpmil', 'cdpmil_source'}:
        defaults = {'lr': .0002, 'weight_decay': 1e-5, 'grad_clip': None, 'selection_metric': 'log_loss', 'patience': 20} if method == 'tpmil' else {'lr': .01, 'weight_decay': .005, 'grad_clip': None}
        for key, value in defaults.items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('rankmix_'):
        for key, value in {'lr': .0002, 'weight_decay': .005, 'grad_clip': None}.items():
            if key not in supplied:
                cfg[key] = value
    if method in {'cate', 'vilamil', 'vilamil_source'}:
        defaults = {'lr': 1e-5, 'weight_decay': 0.} if method == 'cate' else {'lr': 1e-4, 'weight_decay': 1e-5, 'selection_metric': 'log_loss', 'patience': 20}
        for key, value in defaults.items():
            if key not in supplied:
                cfg[key] = value
    if method == 'he2rna':
        for key, value in {'lr': .0003, 'weight_decay': 0., 'selection_metric': 'mean_pearson', 'patience': 50}.items():
            if key not in supplied:
                cfg[key] = value
    if method == 'healnet':
        for key, value in {'lr': .008, 'weight_decay': 0., 'selection_metric': 'survival_loss', 'patience': 5}.items():
            if key not in supplied:
                cfg[key] = value
    if method == 'camil_context':
        for key, value in {'weight_decay': 0., 'selection_metric': 'log_loss', 'patience': 20}.items():
            if key not in supplied:
                cfg[key] = value
    if method in {'mfcmil_abmil', 'mfcmil_transmil', 'dyhg'} or method.startswith('sdfp_'):
        defaults = {'mfcmil_abmil': (.0005, .0001), 'mfcmil_transmil': (.0002, .00001),
                    'dyhg': (.0001, .00001)}.get(method, (.002, .00001))
        for key, value in zip(('lr', 'weight_decay'), defaults):
            if key not in supplied:
                cfg[key] = value
    if method.startswith('mil_dropout_'):
        if 'lr' not in supplied:
            cfg['lr'] = 1e-4
        if 'weight_decay' not in supplied:
            cfg['weight_decay'] = 1e-4 if method in {'mil_dropout_abmil', 'mil_dropout_gated'} else 5e-3
    if method.startswith('packmil_') and 'lr' not in supplied:
        cfg['lr'] = .001 if method == 'packmil_survival' else .0005
    if method in {'deepattnmisl', 'scmil'}:
        defaults = {'deepattnmisl': {'lr': .0001, 'weight_decay': .0005},
                    'scmil': {'lr': .0002, 'weight_decay': .0001}}
        for key, value in defaults[method].items():
            if key not in supplied:
                cfg[key] = value
    if method.startswith('smmil_') and 'weight_decay' not in supplied:
        cfg['weight_decay'] = 0.
    if cfg["epochs"] < 1 or cfg["patience"] < 1 or cfg["lr"] <= 0 or cfg["threads"] < 1:
        raise ValueError("epochs, patience, lr and threads must be positive")
    if cfg["max_instances"] is not None and cfg["max_instances"] < 1:
        raise ValueError("max_instances must be positive or null")
    if not 0 < cfg["bag_weight"] <= 1:
        raise ValueError("bag_weight must be in (0,1]")
    if method.startswith("clam_") and cfg["bag_weight"] == 1:
        raise ValueError("CLAM requires an active instance loss; label a no-instance-loss experiment separately")
    metric = cfg["selection_metric"] or ("survival_loss" if method == 'deepattnmisl' else "c_index" if bundle.task_type == "survival" else
                                           "qwk" if bundle.task_type == "ordinal" else "macro_auc")
    cfg["selection_metric"] = metric
    inventory = bundle.check(deep=True)
    request = {"method": method, "data": str(bundle.root), "fm": bundle.fm, "split": bundle.split,
               "seed": seed, "config": cfg, "bundle_sha256": bundle.fingerprint,
               "features_sha256": inventory["feature_inventory_sha256"], "version": __version__,
               "software": software_identity(), "device": str(device),
               "synthetic": bool(bundle.config.get("synthetic", False)),
               "method_spec": {k: v for k, v in spec.items() if k not in
                               {"status", "gpu_validation", "real_wsi_validation", "paper_reproduction"}}}
    signature = digest_object(request)
    out = Path(output).resolve()
    if resume:
        previous = read_json(out / "request.json")
        if previous != request:
            raise ValueError("Resume refused: configuration, data, feature inventory or method spec changed")
        if read_json(out / "status.json")["state"] == "completed":
            raise ValueError("Run already completed; use a new output for a new experiment")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "request.json", request)
        write_json(out / "inputs.json", inventory)
    lock = out / ".running.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    started = time.monotonic()
    write_json(out / "status.json", {"state": "running", "signature": signature})
    try:
        seed_everything(seed)
        torch.set_num_threads(cfg["threads"])
        dev = torch.device(device)
        if dev.type == "cuda":
            # A fresh CLI process may not have created a CUDA context yet.
            # reset_peak_memory_stats does not initialize one itself.
            torch.cuda.set_device(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        if method == "mean_linear":
            if resume:
                raise ValueError("mean_linear is a single deterministic fit; resume is not supported")
            from .linear import fit_linear
            fit_linear(bundle, out, cfg)
            write_json(out / "status.json", {"state": "completed", "signature": signature,
                                             "wall_seconds": time.monotonic() - started})
            return out
        model_config = {**cfg['model'], 'sampling_seed': seed} if method.startswith(('miladapter_','mexd_')) else cfg['model']
        model = build_bundle_model(method, bundle, model_config).to(dev)
        if hasattr(model, 'bind_output'):
            model.bind_output(out)
        if method in {"mhim_abmil", "ibmil", "dgrmil"} and cfg["epochs"] <= model.warmup_epochs:
            raise ValueError(f"{method} epochs must include supervised initialization AND main training")
        if hasattr(model, 'configure_training'):
            model.configure_training(cfg)
        if not resume and hasattr(model, "prepare"):
            model.prepare(bundle, seed, dev)
        optimizers = make_optimizers(model, method, cfg["lr"], cfg["weight_decay"])
        epoch0, best, bad, history = 0, -float("inf"), 0, []
        if resume:
            checkpoint = torch.load(out / "last.pt", map_location=dev, weights_only=False)
            if checkpoint["signature"] != signature:
                raise ValueError("Checkpoint signature mismatch")
            model.load_state_dict(checkpoint["model"])
            if method.startswith('center_embeddings_') and len(checkpoint['optimizers']) != 2:
                raise ValueError('Center Embeddings checkpoint must retain both actual Adam optimizers')
            if method.startswith(('lrmil_', 'dispro_', 'hdmil_', 'ahdmil_')) and len(checkpoint['optimizers'])!=len(optimizers):
                raise ValueError('DisPro checkpoint must contain all three actual stage optimizers')
            for o, s in zip(optimizers, checkpoint["optimizers"]):
                o.load_state_dict(s)
            if hasattr(model,'validate_optimizer_state'):
                model.validate_optimizer_state()
            epoch0, best, bad, history = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["bad"], checkpoint["history"]
            restore_rng(checkpoint["rng"])
        if hasattr(model, 'protocol'):
            write_json(out / ('survival_protocol.json' if bundle.task_type == 'survival' else 'method_protocol.json'), model.protocol())
        write_json(out / "environment.json", {"python_packages": {p: importlib.metadata.version(p) for p in
            ("torch", "numpy", "scikit-learn", "h5py")}, "device": str(dev),
            "cuda": torch.version.cuda, "parameters": sum(p.numel() for p in model.parameters())})
        for epoch in range(epoch0, cfg["epochs"]):
            model.train()
            if hasattr(model, 'start_epoch'):
                model.start_epoch(epoch)
            if method == "dgrmil":
                model.epoch = epoch
            groups = units(bundle, "train")
            if not getattr(model, 'owns_sampling', False):
                random.shuffle(groups)
            if method == "mhim_abmil":
                model.epoch, model.total_epochs, model.steps_per_epoch = epoch, cfg["epochs"], len(groups)
                if epoch == model.warmup_epochs and not model.teacher_initialized:
                    model.initialize_teacher()
                    optimizers[0].state.clear()  # new training stage, initialization provenance stays in checkpoint
            if method == "ibmil" and epoch == model.warmup_epochs and not model.prepared:
                model.initialize_confounders(bundle, seed, dev, cfg)
                optimizers[0].state.clear()
            losses = []
            if method.startswith('mexd_'):
                losses=model.epoch(bundle,optimizers,dev)
            elif method.startswith('lpgmil_'):
                losses=model.epoch(bundle,optimizers[0],dev)
            elif method.startswith('promptmil_'):
                losses=model.epoch(bundle,optimizers[0],dev)
            elif method.startswith('iibmil_'):
                losses = model.epoch(bundle, optimizers[0], dev)
            elif method.startswith('longmil_'):
                from .longmil import longmil_epoch
                losses = longmil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('sclwc_'):
                from .sclwc import sclwc_epoch
                losses = sclwc_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith(('center_embeddings_','lrmil_','ahdmil_','hdmil_','dispro_','amiml_')):
                losses=model.epoch(bundle,optimizers,dev)
            elif method.startswith(('mggmmil_','nciemil_','taxomil_','gmmamba_','muse_','libramil_','m4_','convl_','miladapter_','conmil_','sparseconvmil_raw_','c2c_cluster_','sammil_','kmgmamba_','mgtrans_','zoommil_','augdiff_','vlsa_','slip_','dtmil_','pamil_policy_','focus_','codmil_','ins_frozen_','deepgraphsurv_','triagemil_','c2mil_','gatmamba_','protosurv_','pathrwkv_','c2aug_','agdrmil_','h2mil_','hipt_lgp_','simil_')):
                losses=model.epoch(bundle,optimizers[0],dev)
            elif method.startswith('mamba2dmil_'):
                from .mamba2d import mamba2d_epoch
                losses = mamba2d_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif bundle.task_type == 'regression':
                from .regression import regression_epoch
                losses = regression_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method in {'smmile', 'smmile_binary', 'smmile_multilabel'}:
                from .smmile import smmile_epoch
                losses = smmile_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('pamil_prototype_'):
                from .pamil import pamil_epoch
                losses = pamil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('monai_'):
                from .monai_mil import monai_epoch
                losses = monai_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('ttis_'):
                from .ttis import ttis_epoch
                losses = ttis_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('msattention_'):
                from .msattention import msattention_epoch
                losses = msattention_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('pmil_shapley_'):
                from .pmil_shapley import pmil_epoch
                losses = pmil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('mammoth_'):
                from .mammoth import mammoth_epoch
                losses = mammoth_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == 'campanella_rnn':
                from .campanella import campanella_epoch
                losses = campanella_epoch(model, bundle, groups, optimizers, dev, cfg)
            elif method.startswith('stablemil_'):
                from .stablemil import stable_epoch
                losses = stable_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method in {'setmil_source', 'setmil_repaired'}:
                from .setmil import setmil_epoch
                losses = setmil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method in {'gtp_source', 'gtp_repaired'}:
                from .gtp import gtp_epoch
                losses = gtp_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('probsa_'):
                from .probsa import probsa_epoch
                losses = probsa_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == 'micromil_frozen':
                from .micromil import micromil_epoch
                losses = micromil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('loss_attention_'):
                from .loss_attention import loss_attention_epoch
                losses = loss_attention_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method in {'dgmil', 'dgmil_source_extremes'}:
                from .dgmil import dgmil_epoch
                losses = dgmil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('mambamil_'):
                from .mamba_mil import mamba_epoch
                losses = mamba_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == 'msmmil':
                from .msmmil import msmmil_epoch
                losses = msmmil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('mfemil_'):
                from .mfemil import mfemil_epoch
                losses = mfemil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method in {'momil', 'momil_survival_extension'}:
                from .momil import momil_epoch
                losses = momil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('recomil_'):
                from .recomil import recomil_epoch
                losses = recomil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method in {'mammil', 'mammil_source', 'mammil_patient_union'}:
                from .mammil import mammil_epoch
                losses = mammil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('pamoe_'):
                from .pamoe import pamoe_epoch
                losses = pamoe_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('pathomics_'):
                from .pathomics import pathomics_epoch
                losses = pathomics_epoch(model, bundle, groups, optimizers, dev, cfg)
            elif method.startswith('mmp_'):
                from .mmp import mmp_epoch
                losses = mmp_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == 'pibd_source':
                from .pibd import pibd_epoch
                losses = pibd_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == 'healnet':
                from .healnet import healnet_epoch
                losses = healnet_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == 'camil_channel':
                from .channel_camil import channel_epoch
                losses = channel_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('packmil_'):
                from .packing import packed_epoch
                losses = packed_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('advmil_'):
                from .advmil import adversarial_epoch
                losses = adversarial_epoch(model, bundle, groups, optimizers, dev, cfg)
            elif bundle.task_type == "survival":
                if getattr(model, 'survival_objective', None) == 'batch_cox_patient_mean':
                    from .cluster_survival import deepattn_epoch
                    loss = deepattn_epoch(model, bundle, groups, optimizers[0], dev, cfg)
                elif getattr(model, 'survival_objective', None) in {'discrete_nll', 'continuous_mdn'}:
                    from .discrete_survival import discrete_epoch
                    loss = discrete_epoch(model, bundle, groups, optimizers[0], dev, cfg)
                else:
                    loss = survival_epoch(model, bundle, groups, optimizers[0], dev, cfg)
                losses.append({'loss': loss})
            elif method.startswith('bayesmil_'):
                from .bayesian import bayes_epoch
                losses = bayes_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('smmil_') or method == 'psamil':
                from .spatial_attention import spatial_epoch
                losses = spatial_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == 'slidegraph':
                from .slidegraph import slidegraph_epoch
                losses = slidegraph_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method in {'csmil', 'hagmil_ce'}:
                from .multiscale import csmil_epoch, hagmil_epoch
                losses = {'csmil': csmil_epoch, 'hagmil_ce': hagmil_epoch}[method](
                    model, bundle, groups, optimizers[0], dev, cfg)
            elif method in {'vilamil', 'vilamil_source'}:
                from .vila import vila_epoch
                losses = vila_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == "cbmil":
                from .cluster_contrastive import cbmil_epoch
                losses = cbmil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == "nnmil":
                from .batched import nnmil_epoch
                losses = nnmil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == 'hmil_source_repaired':
                from .hmil import hmil_epoch
                losses = hmil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == 'tpmil':
                from .tpmil import tpmil_epoch
                losses = tpmil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == 'cdpmil_source':
                from .cdpmil import cdpmil_epoch
                losses = cdpmil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method.startswith('rankmix_'):
                from .rankmix import rankmix_epoch
                losses = rankmix_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == "frmil":
                from .instance_models import frmil_epoch
                losses = frmil_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method in {"psemix_abmil", "prodiv_abmil"}:
                from .augmentation import pseudo_epoch
                losses = pseudo_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            elif method == "remix_abmil":
                from .augmentation import remix_epoch
                losses = remix_epoch(model, bundle, groups, optimizers[0], dev, cfg)
            else:
                for step, rows in enumerate(groups):
                    if method == "mhim_abmil":
                        model.step_in_epoch = step
                    x, coords = load_unit(bundle, rows, dev, cfg["max_instances"], train=True)
                    target = torch.tensor([rows[0]["target"]], device=dev)
                    losses.append(classification_step(model, method, x, coords, target, optimizers,
                                                      cfg["grad_clip"], cfg["bag_weight"]))
            if hasattr(model, 'finish_epoch'):
                model.finish_epoch(bundle, dev, cfg)
            alignment_stage = method.startswith('pathomics_') and model.pretraining
            epoch_metric = 'alignment_loss' if alignment_stage else metric
            validation_due = not hasattr(model, 'should_validate') or model.should_validate(epoch)
            if alignment_stage:
                val = None
                metrics = model.alignment_validation(bundle, dev)
            elif not validation_due:
                val, metrics = None, {'validation_performed': False}
            else:
                val = predict(model, bundle, method, "val", dev, cfg)
                metrics = score(bundle, val)
                if hasattr(model, 'validation_metrics'):
                    metrics.update(model.validation_metrics(val))
            if validation_due and hasattr(model, 'after_validation'):
                model.after_validation(metrics)
            value = metrics.get(epoch_metric)
            if validation_due and (value is None or not np.isfinite(value)):
                raise ValueError(f"Validation selection metric {epoch_metric} is undefined")
            objective = (-value if epoch_metric in {"log_loss", "survival_loss", "mse", "mae", "alignment_loss", "smmile_source_loss", "pamil_loss", "loss_attention_error", "ttis_val_loss"} else value) if validation_due else best
            if validation_due and hasattr(model, 'selection_objective'):
                objective = model.selection_objective(metrics, epoch, objective)
                if not np.isfinite(objective):
                    raise ValueError('Method selection objective is undefined')
            selecting = method not in {"mhim_abmil", "ibmil", "dgrmil"} or epoch >= model.warmup_epochs
            selecting = selecting and validation_due and epoch >= getattr(model, 'minimum_selection_epoch', 0)
            improved = selecting and objective > best
            if hasattr(model, 'selection_improved'):
                improved = selecting and model.selection_improved(objective, best)
            best, bad = (objective, 0) if improved else (best, bad + int(selecting))
            record = {"epoch": epoch, "train": {k: float(np.mean([l[k] for l in losses if k in l]))
                       for k in sorted({k for loss in losses for k in loss})}, "validation": metrics}
            history.append(record)
            state = {"epoch": epoch, "model": model.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
                     "rng": rng_state(), "best": best, "bad": bad, "history": history, "signature": signature}
            atomic_torch_save(state, out / "last.pt")
            if improved:
                selected_state=model.selection_checkpoint(state) if hasattr(model,'selection_checkpoint') else state
                atomic_torch_save(selected_state, out / "best.pt")
                val.to_csv(out / "validation_predictions.csv", index=False)
                write_json(out / "validation_metrics.json", metrics)
            write_json(out / "history.json", history)
            if method.startswith('pathomics_'):
                write_json(out / "survival_protocol.json", model.protocol())
            if method.startswith(('mambamil_', 'mamba2dmil_')):
                write_json(out / ('survival_protocol.json' if bundle.task_type == 'survival' else 'method_protocol.json'), model.protocol())
            if method.startswith('simil_'):
                write_json(out / 'method_protocol.json', model.protocol())
            if method.startswith('iibmil_'):
                write_json(out / 'method_protocol.json', model.protocol())
            if method.startswith(('mexd_','lpgmil_','promptmil_')):
                write_json(out / 'method_protocol.json', model.protocol())
            if method.startswith('hipt_lgp_'):
                write_json(out / 'method_protocol.json', model.protocol())
            if method.startswith('h2mil_'):
                write_json(out / 'method_protocol.json', model.protocol())
            if method.startswith('agdrmil_'):
                write_json(out / 'method_protocol.json', model.protocol())
            if method.startswith(('mggmmil_','nciemil_','taxomil_','gmmamba_','muse_','libramil_','m4_','center_embeddings_','miladapter_','conmil_','sparseconvmil_raw_','c2c_cluster_','sammil_','lrmil_','kmgmamba_','ahdmil_','hdmil_','mgtrans_','zoommil_','augdiff_', 'slip_', 'amiml_', 'dtmil_', 'pamil_policy_', 'focus_', 'codmil_', 'c2aug_', 'sclwc_', 'ins_frozen_')):
                write_json(out / 'method_protocol.json', model.protocol())
            if method.startswith(('convl_','vlsa_','dispro_')):
                write_json(out/'survival_protocol.json',model.protocol())
            if method.startswith(('deepgraphsurv_','triagemil_','c2mil_','gatmamba_','protosurv_','pathrwkv_')):
                write_json(out / 'method_protocol.json', model.protocol())
            if method.startswith('mfemil_'):
                write_json(out / ('survival_protocol.json' if bundle.task_type == 'survival' else 'method_protocol.json'), model.protocol())
            if method in {'mammil', 'mammil_source', 'mammil_patient_union'}:
                write_json(out / ('survival_protocol.json' if bundle.task_type == 'survival' else 'method_protocol.json'), model.protocol())
            if method in {'momil', 'momil_survival_extension', 'dgmil', 'dgmil_source_extremes', 'msmmil'} or method.startswith('recomil_'):
                write_json(out / ('survival_protocol.json' if bundle.task_type == 'survival' else 'method_protocol.json'), model.protocol())
            if method in {'smmile', 'smmile_binary', 'smmile_multilabel'} or method.startswith('pamil_prototype_'):
                write_json(out / 'method_protocol.json', model.protocol())
            if method.startswith(('ttis_', 'msattention_', 'pmil_shapley_', 'mammoth_', 'monai_', 'stablemil_', 'loss_attention_', 'probsa_')) or method in {'campanella_rnn', 'micromil_frozen', 'gtp_source', 'gtp_repaired', 'setmil_source', 'setmil_repaired'}:
                write_json(out / 'method_protocol.json', model.protocol())
            displayed = f'{value:.5f}' if validation_due else 'not_scheduled'
            print(f"epoch={epoch + 1} loss={record['train']['loss']:.5f} val_{epoch_metric}={displayed}", flush=True)
            if getattr(model, 'training_done', False):
                break
            if not getattr(model, 'handles_early_stopping', False) and bad >= cfg["patience"] and epoch + 1 > getattr(model, 'stop_after_epoch', 0):
                break
            if stop_after is not None and epoch + 1 >= stop_after and epoch + 1 < cfg["epochs"]:
                write_json(out / "status.json", {"state": "interrupted", "reason": "requested epoch boundary", "epoch": epoch})
                return out
        write_json(out / "status.json", {"state": "completed", "signature": signature,
                 "wall_seconds": time.monotonic() - started,
                 "peak_gpu_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
                 "completed_epochs": len(history), "test_evaluated": False})
        return out
    except BaseException as e:
        write_json(out / "status.json", {"state": "interrupted" if isinstance(e, KeyboardInterrupt) else "failed",
                   "error_type": type(e).__name__, "error": str(e), "traceback": traceback.format_exc()})
        raise
    finally:
        lock.unlink(missing_ok=True)


def evaluate(run, subset="test", device="cpu"):
    if subset not in {"val", "test"}:
        raise ValueError("Evaluation subset must be val or test")
    out = Path(run)
    req = read_json(out / "request.json")
    initialize_cpu_vector_math()
    if req.get('backend') == 'fouriermil_original_200_accumulation2_val_loss_locked_test':
        from .fouriermil_driver import evaluate as fourier_evaluate
        return fourier_evaluate(run, subset, device)
    if req.get('backend') == 'fcmil_original_mechanism_explicit_benchmark100_auc_writer_repaired':
        from .fcmil_driver import evaluate as fcmil_evaluate
        return fcmil_evaluate(run, subset, device)
    if req.get('backend') == 'nnmil_complete_author_classification_TRAIN_only_planner_repaired':
        from .nnmil_driver import evaluate as nnmil_evaluate
        return nnmil_evaluate(run, subset, device)
    if req.get('backend') == 'mrblock_complete_author_merge_aware_locked_test':
        from .mrblock_driver import evaluate as mrblock_evaluate
        return mrblock_evaluate(run, subset, device)
    if req.get('backend') == 'carmil_complete_author_pretrain_selector_locked_test':
        from .carmil_driver import evaluate as carmil_evaluate
        return carmil_evaluate(run, subset, device)
    if req.get('backend') == 'attrimil_complete_author_recipe_repaired':
        from .attrimil_driver import evaluate as attrimil_evaluate
        return attrimil_evaluate(run, subset, device)
    if req.get('backend') == 'asmil_complete_author_recipe_locked_test':
        from .asmil_driver import evaluate as asmil_evaluate
        return asmil_evaluate(run, subset, device)
    if req.get('backend') == 'pneamil_complete_author_recipe':
        from .pneamil_driver import evaluate as pneamil_evaluate
        return pneamil_evaluate(run, subset, device)
    if req.get('backend')=='lnplmil_full_raw_LPA_pipeline':
        from .lnplmil_driver import evaluate as lnpl_evaluate
        return lnpl_evaluate(run,subset,device)
    if req.get('backend')=='hybrid_supervision_full_paper_reference':
        from .hybrid_supervision_driver import evaluate as hybrid_evaluate
        return hybrid_evaluate(run,subset,device)
    if req.get('backend')=='macnet_complete_raw_reference':
        from .macnet_driver import evaluate as mac_evaluate
        return mac_evaluate(run,subset,device)
    if req.get('backend')=='protomil_complete_reference':
        from .protomil_driver import evaluate as proto_evaluate
        return proto_evaluate(run,subset,device)
    if req.get('backend')=='kmpmil_complete_continual_reference':
        from .kmpmil_driver import evaluate as kmp_evaluate
        return kmp_evaluate(run,subset,device)
    if req.get('backend')=='qpmilvl_complete_continual_reference':
        from .qpmilvl_driver import evaluate as qpmil_evaluate
        return qpmil_evaluate(run,subset,device)
    if req.get('backend')=='akdpmp_complete_continual_reference':
        from .akdpmp_driver import evaluate as akd_evaluate
        return akd_evaluate(run,subset,device)
    if req.get('backend')=='comel_complete_continual_reference':
        from .comel_driver import evaluate as comel_evaluate
        return comel_evaluate(run,subset,device)
    if req.get('backend')=='pmil_prototypical_complete_reference':
        from .pmil_prototypical_driver import evaluate as pmil_evaluate
        return pmil_evaluate(run,subset,device)
    if req.get('backend')=='icmil_coupled_complete_reference':
        from .icmil_coupled_driver import evaluate as coupled_evaluate
        return coupled_evaluate(run,subset,device)
    if req.get('backend')=='icmil_context_original_pretrained':
        from .icmil_context_driver import evaluate as context_evaluate
        return context_evaluate(run,subset,device)
    if req.get('backend')=='dspamil2025_full_PAT_reference':
        from .dspamil_driver import evaluate as dspa_evaluate
        return dspa_evaluate(run,subset,device)
    if req.get('backend')=='feather24k_full_pretrained_reference':
        from .feather_driver import evaluate as feather_evaluate
        return feather_evaluate(run,subset,device)
    if req.get('backend')=='dyko_full_original_TITAN':
        from .dyko_driver import evaluate as dyko_evaluate
        return dyko_evaluate(run,subset,device)
    if req.get('backend')=='rmdl_complete_native_tensorflow':
        from .rmdl_driver import evaluate as rmdl_evaluate
        return rmdl_evaluate(run,subset,device)
    if req.get('backend')=='attention2majority_native_tensorflow':
        from .attention2majority_driver import evaluate as native_evaluate
        return native_evaluate(run,subset,device)
    if read_json(out / "status.json")["state"] != "completed":
        raise ValueError("Evaluation requires a completed run")
    bundle = Bundle(req["data"], req["fm"], req["split"])
    if req["software"] != software_identity():
        raise ValueError("Evaluation software differs from the recorded run; use the locked environment/source")
    inventory = bundle.check(deep=True)
    if bundle.fingerprint != req["bundle_sha256"] or inventory["feature_inventory_sha256"] != req["features_sha256"]:
        raise ValueError("Evaluation inputs differ from training inputs")
    seed_everything(req["seed"])
    torch.set_num_threads(req["config"]["threads"])
    if req["method"] == "mean_linear":
        from .linear import predict_linear
        predictions = predict_linear(bundle, out, subset)
    else:
        model_config = {**req['config']['model'], 'sampling_seed': req['seed']} if req['method'].startswith(('miladapter_','mexd_')) else req['config']['model']
        model = build_bundle_model(req['method'], bundle, model_config).to(device)
        if hasattr(model, 'bind_output'):
            model.bind_output(out)
        state = torch.load(out / "best.pt", map_location=device, weights_only=False)
        if state["signature"] != digest_object(req):
            raise ValueError("Best checkpoint does not belong to this run")
        model.load_state_dict(state["model"])
        if getattr(model, 'supports_instance_outputs', False):
            predictions = predict(model, bundle, req['method'], subset, device, req['config'],
                                  instance_output=out/f'{subset}_instances')
        else:
            predictions = predict(model, bundle, req["method"], subset, device, req["config"])
    metrics = score(bundle, predictions)
    if req['method'] != 'mean_linear' and hasattr(model, 'validation_metrics'):
        metrics.update(model.validation_metrics(predictions))
    predictions.to_csv(out / f"{subset}_predictions.csv", index=False)
    write_json(out / f"{subset}_metrics.json", metrics)
    return metrics
