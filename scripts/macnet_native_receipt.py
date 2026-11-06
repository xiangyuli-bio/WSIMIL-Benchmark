"""Strict actual current-code installed-wheel full CUDA evidence for a CPU matrix."""
from pathlib import Path
import json
from milplus.io import digest_file
from milplus.macnet_driver import PROFILES
from milplus.macnet_source import SWIN22K_SHA256


def verify_native_receipt(path,software):
    path=Path(path);root=json.loads(path.read_text())
    if root.get('state')!='passed_all6_installed_wheel_complete_native' or any(root.get('software',{}).get(k)!=software[k] for k in ['source_sha256','sources_lock_sha256']):raise ValueError('MacNet CPU matrix requires actual passed current-code independent-wheel native100epoch CUDA evidence')
    rows=root.get('conditions',[])
    if len(rows)!=6 or {r.get('profile') for r in rows}!=set(PROFILES):raise ValueError('Complete all six actual MacNet native conditions; no reduced profile substitute')
    for row in rows:
        p=Path(row['evidence_file'])
        if not p.is_absolute():p=path.parent/p
        if digest_file(p)!=row.get('evidence_sha256'):raise ValueError('Actual MacNet native evidence bytes changed')
        v=json.loads(p.read_text())
        if v.get('state')!='passed_complete_full_native' or v.get('profile')!=row['profile'] or v.get('actual_epochs')!=100 or v.get('full_Swin_feature_parameters')!=27499482 or v.get('full_feature_dimension')!=768 or v.get('pretrained_weight_sha256')!=SWIN22K_SHA256 or v.get('fresh_selected_public_CLI_reloads')!=2:
            raise ValueError('MacNet native receipt changed original100epoch/full pretrained raw architecture or actual CLI reload requirements')
        if v.get('actual_guide_count')!={'public':5,'lscc81':81,'cscc64':64}[PROFILES[row['profile']]['condition']] or v.get('all_nonterminal_model_gradients_RMS_scaler_orders_RNG_exact') is not True or v.get('actual_epoch_interruption') is not True:
            raise ValueError('MacNet native receipt lacks actual complete guidance or every-state recovery proof')
        if v.get('device_type')!='cuda' or v.get('formal_admission') is not False:raise ValueError('MacNet software/native scope or hardware declaration changed')
        validator=Path(__file__).with_name('validate_macnet_native.py')
        if not validator.is_file() or v.get('validator_sha256')!=digest_file(validator):raise ValueError('Actual native MacNet validator differs from the current preserved full-budget program')
        from milplus.io import read_json,digest_object
        from milplus.icmil_coupled_training import load
        artifacts=v.get('evidence_files',{})
        if not artifacts:raise ValueError('Native receipt needs actual retained learned-state/history/audit artifacts, not asserted success flags')
        for filename,sha in artifacts.items():
            if digest_file(filename)!=sha:raise ValueError('Retained complete native MacNet artifact bytes changed')
        audits=[]
        for key in ['continuous_run','resumed_run']:
            run=Path(v[key]);request=read_json(run/'request.json');history=read_json(run/'history.json');checkpoint=load(run/'last.pt');selected=read_json(run/'selected.json')
            if len(history)!=100 or checkpoint['next_epoch']!=100 or not checkpoint['finished'] or request['recipe']['epochs']!=100 or request['recipe']['use_amp'] is not False:
                raise ValueError('Original native100epoch fp32 MacNet trajectory is incomplete')
            if checkpoint['signature']!=digest_object(request) or request['software']['source_sha256']!=software['source_sha256'] or request['method']!=v['profile']:
                raise ValueError('Actual native raw training checkpoint is not bound to this source/profile/request')
            if any(str(run/name) not in artifacts for name in ['request.json','history.json','every_state_audit.jsonl','selected.json']) or str(Path(selected['model'])) not in artifacts:
                raise ValueError('Actual complete native model/selection/history evidence is missing from the byte inventory')
            model=load(selected['model'])
            feature_count=sum(t.numel() for k,t in model['model'].items() if k.startswith('feature.') and not k.endswith(('relative_position_index','attn_mask')))
            if feature_count!=27499482 or model['model']['initial_adaptive_diagonal'].shape!=(768,768):raise ValueError('Actual learned native Swin or retained adaptive matrix has changed size')
            trace=[json.loads(line) for line in (run/'every_state_audit.jsonl').read_text().splitlines()]
            if len(trace)!=100 or [r['epoch'] for r in trace]!=list(range(1,101)):raise ValueError('Every original native epoch must have its actual full-state audit')
            audits.append([r['full_state_sha256'] for r in trace])
        if audits[0]!=audits[1]:raise ValueError('Actual continuous and interrupted full native all-state trajectories differ')

    return {'state':'passed_current_independent_wheel_native_receipt','receipt':str(path),'receipt_sha256':digest_file(path),'profiles':sorted(PROFILES)}
