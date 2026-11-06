"""Own immutable batch checkpoints and process locks, without unrelated MIL dependencies."""
import os,json,socket,uuid
from pathlib import Path
import torch
from .io import atomic_torch_save,digest_file,read_json,write_json

def exists(path):
    return Path(path).with_suffix('.checkpoint.json').is_file()

def checkpoint_path(path):
    path = Path(path)
    if path.is_file():
        return path
    record = read_json(path.with_suffix('.checkpoint.json'))
    name = record['payload']
    if Path(name).name != name or not name.startswith(path.stem + '.'):
        raise ValueError('hybrid checkpoint generation escapes its owned stage')
    actual = path.parent / name
    if actual.is_symlink() or digest_file(actual) != record['sha256']:
        raise ValueError('hybrid checkpoint generation or pointer checksum changed')
    return actual

def load(path):
    path = checkpoint_path(path)
    if read_json(path.with_suffix(path.suffix + '.sha256.json'))['sha256'] != digest_file(path):
        raise ValueError('Owned hybrid learned checkpoint bytes changed')
    return torch.load(path, map_location='cpu', weights_only=False)

def save(value, path):
    path = Path(path)
    pointer = path.with_suffix('.checkpoint.json')
    previous = read_json(pointer) if pointer.exists() else None
    actual = path.with_name(path.stem + '.' + uuid.uuid4().hex + path.suffix)
    atomic_torch_save(value, actual)
    sha = digest_file(actual)
    write_json(actual.with_suffix(actual.suffix + '.sha256.json'), {'sha256': sha})
    write_json(pointer, {'payload': actual.name, 'sha256': sha})
    if previous is not None:
        old = path.parent / previous['payload']
        if old.name != previous['payload'] or not old.name.startswith(path.stem + '.') or old.is_symlink():
            raise ValueError('Previous hybrid checkpoint generation is not owned by this stage')
        old.unlink()
        old.with_suffix(old.suffix + '.sha256.json').unlink()
    return actual

def cpu_state(model):
    return {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}

class Interrupted(Exception):
    pass

class Progress:

    def __init__(self, output, stop_after):
        self.root = Path(output)
        self.stop_after = stop_after
        self.completed = 0

    def record(self, phase, epoch, history, finished):
        self.completed += 1
        write_json(self.root / 'status.json', {'state': 'running', 'phase': phase, 'phase_completed_epochs': epoch, 'phase_finished': finished, 'epochs_this_invocation': self.completed, 'last_epoch': history[-1]})
        if self.stop_after is not None and self.completed >= self.stop_after:
            raise Interrupted()

def _lock(out, resume):
    path = out / 'RUNNING.lock'
    if path.exists() and resume:
        prior = read_json(path)
        if prior['hostname'] != socket.gethostname():
            raise ValueError('Verify the previous host process is terminal before reclaiming its hybrid run lock')
        proc = Path('/proc') / str(prior['pid'])
        if proc.exists() and (proc / 'stat').read_text().split()[21] == prior['start_ticks']:
            raise ValueError('Original hybrid run process is still live')
        path.unlink()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 384)
    with os.fdopen(fd, 'w') as f:
        import json
        json.dump({'pid': os.getpid(), 'hostname': socket.gethostname(), 'start_ticks': Path('/proc/self/stat').read_text().split()[21]}, f)
    return path
