"""Verify actual current-wheel native CUDA evidence for the CUDA-only profile."""
from pathlib import Path

from milplus.io import digest_file, read_json


def verify_native_receipt(path, identity):
    path = Path(path).expanduser().resolve()
    report = read_json(path)
    if report.get('state') != 'passed' or report.get('synthetic') is not True:
        raise ValueError('ConVL requires a passed actual current-wheel CUDA native receipt')
    software = report.get('software', {})
    if any(software.get(key) != identity[key] for key in ['source_sha256', 'sources_lock_sha256']):
        raise ValueError('ConVL native receipt belongs to another source or source-lock version')
    records = report.get('profiles', [])
    expected = {(profile, cohort) for profile in ['convl_yaml_fp16_source_repaired', 'convl_yaml_fp32_reference']
                for cohort in ['BRCA', 'COADREAD']}
    if len(records) != 4 or {(r.get('profile'), r.get('cohort')) for r in records} != expected:
        raise ValueError('ConVL requires all four full native precision/cohort conditions')
    for row in records:
        parameters = 112353976 if row['cohort'] == 'BRCA' else 112353984
        if (row.get('state') != 'passed' or row.get('device_type') != 'cuda'
                or row.get('max_epochs') != 200 or row.get('patience') != 8
                or not 9 <= row.get('epochs', 0) <= 200 or row.get('patients') != 20
                or row.get('patches_per_slide') != 257 or row.get('feature_dimension') != 512
                or row.get('parameters') != parameters or row.get('text_layers') != 12
                or row.get('context_tokens') != 16 or row.get('gradient_accumulation') != 2
                or row.get('optimizer_updates', 0) < 1
                or not row.get('all_native_mechanism_markers_updated')
                or not row.get('source_full_fast_slow_and_scaler_state_verified')
                or not row.get('source_NLL_only_selection_and_stopping_verified')
                or not row.get('selected_checkpoint_reloaded_twice')
                or row.get('synthetic') is not True):
            raise ValueError('ConVL native receipt reduced or omitted a real CUDA model/budget/state condition')
    return {'path': str(path), 'sha256': digest_file(path), 'report': report,
            'scope': 'Four actual full current-wheel CUDA native conditions; original FP16 never substituted by CPU FP32'}
