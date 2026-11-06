"""Verify user-supplied FEATHER24K checkpoints against public object identities."""
from pathlib import Path

from .feather_source import COMMIT, ENCODERS, original_configuration
from .io import digest_file,digest_object,read_json,write_json
from .registry import resource


def native_lock():
    value = resource('feather24k.lock.json')
    if (value.get('schema_version') != 1 or value.get('original_source_commit') != COMMIT
            or value.get('pretraining_task') != 'PC-108' or value.get('pretraining_release_scale_label') != '24K'
            or set(value.get('checkpoints',{})) != set(ENCODERS)):
        raise ValueError('Original FEATHER24K model identities changed')
    for encoder,dimension in ENCODERS.items():
        row = value['checkpoints'][encoder]
        if (row['model_id'] != f'MahmoodLab/abmil.base.{encoder}.pc108-24k'
                or row['native_dim'] != dimension or len(row['revision']) != 40
                or set(row['files']) != {'pytorch_model.bin','model.safetensors'}):
            raise ValueError('Original FEATHER24K checkpoint/encoder binding changed')
    return value


def actual_checkpoint(encoder, weights):
    if encoder not in ENCODERS:
        raise ValueError('Choose one of the three original FEATHER24K native encoders')
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('Provide your locally available original FEATHER24K checkpoint')
    digest = digest_file(path);stat = path.stat()
    row = native_lock()['checkpoints'][encoder]
    match = [name for name,expected in row['files'].items()
             if expected['sha256'] == digest and expected['bytes'] == stat.st_size]
    if len(match) != 1:
        raise ValueError('Local checkpoint bytes do not match this original pretrained FEATHER24K variant')
    if match[0] == 'model.safetensors' and path.suffix != '.safetensors':
        raise ValueError('Keep the original safetensors suffix for the author checkpoint loader')
    if match[0] == 'pytorch_model.bin' and path.suffix == '.safetensors':
        raise ValueError('The original PyTorch checkpoint cannot be labelled as a safetensors file')
    return {'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
            'sha256':digest,'source_filename':match[0]}


def prepare(encoder, weights, output):
    checkpoint = actual_checkpoint(encoder,weights)
    lock = native_lock();variant = lock['checkpoints'][encoder]
    config = original_configuration(encoder,2)
    config.pop('num_classes')
    value = {'schema_version':1,'kind':'verified_original_FEATHER24K_local_checkpoint',
        'encoder':encoder,'model_id':variant['model_id'],'revision':variant['revision'],
        'checkpoint':checkpoint,'native_feature_dim':ENCODERS[encoder],
        'source_commit':COMMIT,'source_base_architecture':config,
        'public_identity_lock_sha256':digest_object(lock),
        'pretraining_task':'PC-108','pretraining_release_scale_label':'24K',
        'exact_pretraining_cohort_size_verified':False,
        'model_loaded':False,'pretrained_model_validation':False,'full_method_validation':False,
        'historical_3K_paper_reproduction':False,'remote_files_downloaded':False}
    value['contract'] = digest_object(value)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    write_json(output,value)
    return output


class OriginalFEATHERAsset:
    def __init__(self, manifest):
        self.path = Path(manifest).expanduser().resolve()
        original = read_json(self.path);value = dict(original)
        self.contract = value.pop('contract',None)
        lock = native_lock()
        if (self.contract != digest_object(value) or value.get('schema_version') != 1
                or value.get('kind') != 'verified_original_FEATHER24K_local_checkpoint'
                or value.get('encoder') not in ENCODERS
                or value.get('source_commit') != COMMIT
                or value.get('public_identity_lock_sha256') != digest_object(lock)):
            raise ValueError('Original FEATHER asset binding changed')
        self.encoder = value['encoder'];variant = lock['checkpoints'][self.encoder]
        config = original_configuration(self.encoder,2);config.pop('num_classes')
        if (value.get('source_base_architecture') != config
                or value.get('model_id') != variant['model_id'] or value.get('revision') != variant['revision']
                or value.get('native_feature_dim') != ENCODERS[self.encoder]
                or value.get('pretraining_task') != 'PC-108' or value.get('pretraining_release_scale_label') != '24K'):
            raise ValueError('Original FEATHER checkpoint, complete graph or native encoder changed')
        self.weights = Path(value['checkpoint']['path'])
        if actual_checkpoint(self.encoder,self.weights) != value['checkpoint']:
            raise ValueError('Actual original FEATHER pretrained checkpoint changed')
        self.value = original

    def verify(self, deep=False):
        if read_json(self.path) != self.value:
            raise ValueError('Original FEATHER checkpoint binding changed')
        expected = self.value['checkpoint'];stat = self.weights.stat()
        if (stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']
                or (deep and digest_file(self.weights) != expected['sha256'])):
            raise ValueError('Actual original FEATHER checkpoint bytes changed')
