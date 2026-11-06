"""Portable raw DAE weights bound to the full seven-view TRAIN preparation."""
from pathlib import Path

import torch

from .augdiff_data import AugDiffInputs
from .augdiff_pretrain import training_contract
from .augdiff_source import COMMIT, LAYOUT, LDM_COMMIT, build_diffusion
from .data import resolve
from .io import atomic_torch_save, digest_file, digest_object, read_json, write_json

KIND = 'augdiff_trained_raw_dae'


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest_file(path)}


def export_trained_dae(bundle, views_manifest, pretrain_run, output):
    """Export only a completed, matching pretraining run's selected RAW weights."""
    assets = AugDiffInputs(bundle, views_manifest)
    run = resolve(bundle.root, pretrain_run).resolve()
    output = Path(output).resolve()
    weights = output.with_suffix('.weights.pt')
    if output.exists() or weights.exists():
        raise FileExistsError('AugDiff export exists; use a new immutable asset path')
    report = read_json(run/'protocol.json')
    contract = training_contract(assets, report['max_epochs'], report['batch_size'], report['precision'],
                                 report['seed'], report['device_type'], report['torch_version'], report['backend_flags'])
    if (report.get('contract') != contract or report.get('train_assets_contract') != assets.contract
            or report.get('layout') != LAYOUT or report.get('source') != COMMIT or report.get('ldm') != LDM_COMMIT
            or report.get('completed_epochs') != report['max_epochs'] or report.get('optimizer_updates', 0) <= 0
            or report.get('diffusion_timesteps') != 1000 or report.get('full_dae_parameters', 0) < 120_000_000):
        raise ValueError('AugDiff requires a completed actual full-width DAE run bound to these TRAIN views')
    # Only the private artifact produced by this preparation pipeline uses
    # pickle metadata (RNG/NumPy). The exported downstream asset is pure tensors.
    selected = torch.load(run/'selected.pt', map_location='cpu', weights_only=False, mmap=True)
    if (selected['contract'] != contract or selected['best_epoch'] != report['selected_epoch']
            or selected['completed_epochs'] != report['selected_epoch']+1
            or selected['optimizer_updates'] <= 0 or len(selected['history']) != selected['completed_epochs']):
        raise ValueError('AugDiff selected DAE checkpoint does not match the completed pretraining receipt')
    raw = {key: value for key, value in selected['model'].items() if not key.startswith('model_ema.')}
    with torch.random.fork_rng(devices=[]):
        model = build_diffusion(bundle.dim, use_ema=False)
    model.load_state_dict(raw, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(raw, weights)
    metadata = {'schema_version': 1, 'kind': KIND, 'synthetic': bool(bundle.config.get('synthetic', False)),
        'source': COMMIT, 'ldm': LDM_COMMIT, 'layout': LAYOUT, 'dimension': bundle.dim, 'fm': bundle.fm,
        'views_manifest': file_record(assets.path), 'weights': file_record(weights),
        'train_assets_contract': assets.contract, 'pretrain_contract': contract,
        'pretrain_protocol': report, 'pretrain_protocol_file': file_record(run/'protocol.json'),
        'selected_checkpoint': file_record(run/'selected.pt'),
        'sampling_parameters': 'RAW selected model; EMA only selects checkpoint, not used for MIL inference'}
    metadata['contract'] = digest_object(metadata)
    write_json(output, metadata)
    return str(output)


class AugDiffAssets:
    def __init__(self, bundle):
        self.bundle = bundle
        self.path = resolve(bundle.root, bundle.encoder['augdiff']['manifest']).resolve()
        self.metadata = meta = read_json(self.path)
        body = {key: value for key, value in meta.items() if key != 'contract'}
        if (meta.get('schema_version') != 1 or meta.get('kind') != KIND
                or meta.get('contract') != digest_object(body)
                or meta.get('source') != COMMIT or meta.get('ldm') != LDM_COMMIT or meta.get('layout') != LAYOUT
                or type(meta.get('synthetic')) is not bool
                or meta['synthetic'] != bool(bundle.config.get('synthetic', False))
                or meta.get('dimension') != bundle.dim or meta.get('fm') != bundle.fm):
            raise ValueError('AugDiff trained DAE manifest/version/FM/full dimension differs')
        self.views = AugDiffInputs(bundle, meta['views_manifest']['path'])
        report = meta['pretrain_protocol']
        # Preparation environment is recorded, not replaced with the consumer's
        # Torch CPU/CUDA build. Frozen raw weights may cross devices; numerical
        # validation for each consumer device remains a separate evidence gate.
        expected = training_contract(self.views, report['max_epochs'], report['batch_size'], report['precision'],
                                     report['seed'], report['device_type'], report['torch_version'], report['backend_flags'])
        if (meta['train_assets_contract'] != self.views.contract or meta['pretrain_contract'] != expected
                or report['contract'] != expected or report['completed_epochs'] != report['max_epochs']
                or report['optimizer_updates'] <= 0):
            raise ValueError('AugDiff DAE preparation used different TRAIN views, budget or source runtime')
        for name in ['views_manifest', 'weights', 'pretrain_protocol_file', 'selected_checkpoint']:
            entry = meta[name]; path = Path(entry['path'])
            if path.stat().st_size != entry['bytes'] or digest_file(path) != entry['sha256']:
                raise ValueError('AugDiff frozen DAE preparation bytes changed: '+name)
        self.weights = Path(meta['weights']['path'])
        self.files = list(dict.fromkeys([self.path, *self.views.files, self.weights]))
        self.metadata_sha = digest_file(self.path)
        stat = self.weights.stat(); self.weights_stat = (stat.st_size, stat.st_mtime_ns)
        self.contract = digest_object({'metadata_sha256': self.metadata_sha, 'views': self.views.contract})

    def check_unchanged(self):
        if digest_file(self.path) != self.metadata_sha:
            raise ValueError('AugDiff trained DAE manifest changed')
        stat = self.weights.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.weights_stat:
            raise ValueError('AugDiff immutable raw DAE weights changed')

    def load_dae(self, device):
        self.check_unchanged()
        # The source keeps the DAE outside the classifier/optimizer. Omit only
        # unused EMA cloning during inference; all RAW weights strictly load.
        with torch.random.fork_rng(devices=[]):
            model = build_diffusion(self.bundle.dim, use_ema=False)
        state = torch.load(self.weights, map_location='cpu', weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True)
        return model.to(device).requires_grad_(False).eval()

    def load_slide(self, row):
        self.check_unchanged()
        return self.bundle.load(row)[0]

    def inventory(self, row, deep=False):
        self.check_unchanged()
        return {'raw_weights_sha256': self.metadata['weights']['sha256'],
                'prepared_train_views_sha256': self.views.contract, 'layout': LAYOUT,
                'full_feature_dimension': self.bundle.dim}
