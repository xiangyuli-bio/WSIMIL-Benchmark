import importlib.metadata
from pathlib import Path

from .io import digest_file, digest_object


def software_identity():
    root = Path(__file__).parent
    versions = {p: importlib.metadata.version(p) for p in
                ('torch', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'h5py', 'einops')}
    for optional in ('torch-geometric', 'torch-scatter', 'torch-sparse', 'gpytorch', 'linear-operator', 'nystrom-attention', 'POT', 'admin-torch', 'faiss-cpu', 'ftfy', 'regex', 'nmslib', 'networkx', 'scikit-image', 'dgl', 'monai', 'torchvision', 'openslide-python', 'openslide-bin', 'natsort', 'transformers', 'tokenizers', 'huggingface-hub', 'safetensors', 'xformers'):
        try:
            versions[optional] = importlib.metadata.version(optional)
        except importlib.metadata.PackageNotFoundError:
            versions[optional] = None
    return {"source_sha256": digest_object({str(p.relative_to(root)): digest_file(p)
             for p in sorted(root.rglob("*.py"))}),
            "sources_lock_sha256": digest_file(root / "resources/sources.lock.json"),
            "versions": versions}
