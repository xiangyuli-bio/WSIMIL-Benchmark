"""Task bundles, patient identity checks and safe feature loading."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from .io import digest_file, digest_object


def load_array(path, key="features"):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as f:
            if key not in f:
                raise ValueError(f"{path}: missing HDF5 dataset {key}")
            link=f.get(key,getlink=True)
            if not isinstance(link,h5py.HardLink):
                raise ValueError(f"{path}: HDF5 aliases/external links must be materialized before feature hashing")
            dataset=f[key]
            if not isinstance(dataset,h5py.Dataset) or dataset.is_virtual or dataset.external:
                raise ValueError(f"{path}: virtual/external-storage HDF5 datasets must be materialized before feature hashing")
            return dataset[:]
    if path.suffix in {".pt", ".pth"}:
        x = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(x, dict):
            x = x[key]
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"{path}: expected a tensor or tensor dictionary")
        return x.numpy()
    raise ValueError(f"Unsupported feature format: {path}")


def resolve(root, path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else root / p


class Bundle:
    def __init__(self, path, fm=None, split=None):
        self.root = Path(path).expanduser().resolve()
        self.slip = None
        self.vlsa = None
        self.convl = None
        self.dispro = None
        self.config = yaml.safe_load((self.root / "task.yaml").read_text())
        c = self.config
        if c.get("schema_version") != 1:
            raise ValueError("task.yaml requires schema_version: 1")
        self.task_type = c["task_type"]
        if self.task_type not in {"classification", "ordinal", "survival", "regression", "multilabel", "multitask"}:
            raise ValueError(f"Unsupported task type {self.task_type}")
        self.fm = fm or c["defaults"]["fm"]
        self.split = split or c["defaults"]["split"]
        self.encoder = c["encoders"][self.fm]
        self.dim = int(self.encoder["dimension"])
        if self.dim <= 0:
            raise ValueError("Feature dimension must be positive")
        self.classes = c.get("classes", [])
        if self.task_type in {"classification", "ordinal", "multilabel"} and (len(self.classes) < 2 or len(set(self.classes)) != len(self.classes)):
            raise ValueError("classes must contain at least two unique ordered names")
        files = [self.root / "task.yaml", resolve(self.root, c.get("slides", "slides.csv")),
                 resolve(self.root, c["splits"][self.split]), resolve(self.root, self.encoder["index"])]
        slides, splits, features = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in files[1:]]
        for table, required, name in [(slides, {"slide_id", "patient_id"}, "slides"),
                                      (splits, {"patient_id", "subset"}, "splits"),
                                      (features, {"slide_id", "path"}, "features")]:
            if not required <= set(table):
                raise ValueError(f"{name}: missing columns {required - set(table)}")
            if table[list(required)].eq("").any().any():
                raise ValueError(f"{name}: empty identity/path/subset")
        if slides.slide_id.duplicated().any() or features.slide_id.duplicated().any():
            raise ValueError("Duplicate slide_id in slides or feature index")
        if features.path.map(lambda p: str(resolve(self.root, p).resolve())).duplicated().any():
            raise ValueError("Multiple slides reference the same physical feature path")
        if splits.patient_id.duplicated().any():
            raise ValueError("Duplicate patient in split: possible patient leakage")
        if set(splits.subset) != {"train", "val", "test"}:
            raise ValueError("Split must contain train, val and test only")
        if set(slides.patient_id) != set(splits.patient_id):
            raise ValueError("Split and slide patient sets differ; filter explicitly before creating bundle")
        if set(slides.slide_id) != set(features.slide_id):
            raise ValueError("Feature and slide sets differ; missing/extra features cannot be silently dropped")
        duplicate_columns = (set(slides) & set(features)) - {"slide_id"}
        if duplicate_columns:
            raise ValueError(f"Feature index must not redefine slide metadata: {duplicate_columns}")
        self.rows = slides.merge(splits, on="patient_id", validate="many_to_one").merge(
            features, on="slide_id", validate="one_to_one").sort_values("slide_id").reset_index(drop=True)
        if "global_patient_id" not in self.rows:
            self.rows["global_patient_id"] = self.rows.patient_id
        if self.rows.global_patient_id.eq("").any():
            raise ValueError("Empty global_patient_id")
        if (self.rows.groupby("global_patient_id").subset.nunique() > 1).any():
            raise ValueError("Global patient identity crosses splits")
        if self.task_type == "survival":
            for key in ("time", "event"):
                self.rows[key] = pd.to_numeric(self.rows[key], errors="raise")
            if not np.isfinite(self.rows.time).all() or (self.rows.time <= 0).any():
                raise ValueError("Survival time must be finite and positive")
            if not set(self.rows.event) <= {0, 1}:
                raise ValueError("event must be 0=censored or 1=observed")
            if (self.rows.groupby("global_patient_id")[["time", "event"]].nunique() > 1).any().any():
                raise ValueError("Inconsistent survival outcome within patient")
        elif self.task_type in {"classification", "ordinal"}:
            if not set(self.rows.label) <= set(self.classes):
                raise ValueError("Unknown or missing class label")
            self.rows["target"] = self.rows.label.map({v: i for i, v in enumerate(self.classes)})
            if c.get("evaluation_unit", "slide") == "patient" and (self.rows.groupby("global_patient_id").target.nunique() > 1).any():
                raise ValueError("Patient-level evaluation requires consistent patient labels")
            for subset in ("train", "val", "test"):
                if set(self.rows[self.rows.subset == subset].label) != set(self.classes):
                    raise ValueError(f"{subset} is missing a class; AUROC protocol cannot be evaluated")
        self.multilabel = None
        self.multitask = None
        if self.task_type == 'multitask':
            from .multitask import MultiTaskTargets
            self.multitask = MultiTaskTargets(self)
            files.extend(self.multitask.files)
        if self.task_type == 'multilabel':
            from .multilabel import MultilabelTargets
            self.multilabel = MultilabelTargets(self)
        self.hierarchy = None
        if 'hierarchy' in c:
            from .hierarchy import HierarchyLabels
            self.hierarchy = HierarchyLabels(self)
        self.regression_targets = None
        if self.task_type == 'regression':
            from .regression import RegressionTargets
            self.regression_targets = RegressionTargets(self)
            files.extend(self.regression_targets.files)
        self.omics = None
        if 'omics' in c:
            from .omics import OmicsTable
            self.omics = OmicsTable(self)
            files.extend(self.omics.files)
        self.multiscale = None
        if 'multiscale' in self.encoder:
            from .multiscale_data import MultiscaleTable
            self.multiscale = MultiscaleTable(self)
            files.extend(self.multiscale.files)
        self.vision_language = None
        if 'vision_language' in c or 'vision_language' in self.encoder:
            from .vision_language_data import VisionLanguageData
            self.vision_language = VisionLanguageData(self)
            files.extend(self.vision_language.files)
        self.dual_scale = None
        if 'dual_scale' in self.encoder:
            from .vision_language_data import DualScaleData
            self.dual_scale = DualScaleData(self)
            files.extend(self.dual_scale.files)
        self.pamoe = None
        if 'pamoe' in self.encoder:
            from .pamoe_data import TissuePrototypes
            self.pamoe = TissuePrototypes(self)
            files.extend(self.pamoe.files)
        self.regions = None
        if 'regions' in self.encoder:
            from .region_data import RegionTable
            self.regions = RegionTable(self)
        self.smmile = None
        if 'smmile' in self.encoder:
            from .smmile_data import SuperpixelTable
            self.smmile = SuperpixelTable(self)
            files.extend(self.smmile.files)
        self.pamil = None
        if 'pamil' in self.encoder:
            from .pamil_data import PAMILAnnotations
            self.pamil = PAMILAnnotations(self)
            files.extend(self.pamil.files)
        self.raw_images = None
        if 'raw_images' in self.encoder:
            from .raw_images import RawImages
            self.raw_images = RawImages(self)
            files.extend(self.raw_images.files)
        self.raw_regions = None
        if 'raw_regions' in self.encoder:
            from .raw_regions import RawRegions
            self.raw_regions = RawRegions(self)
            files.extend(self.raw_regions.files)
        self.ttis = None
        if 'ttis' in self.encoder:
            from .ttis_data import TTISAssets
            self.ttis = TTISAssets(self)
            files.extend(self.ttis.files)
        self.simil = None
        if 'simil' in self.encoder:
            from .simil_data import PathExpertAssets
            self.simil = PathExpertAssets(self)
            files.extend(self.simil.files)
        self.hipt = None
        if 'hipt' in self.encoder:
            from .hipt_data import HIPTPyramid
            self.hipt = HIPTPyramid(self)
            files.extend(self.hipt.files)
        self.h2mil = None
        if 'h2mil' in self.encoder:
            from .h2mil_data import H2Hierarchy
            self.h2mil = H2Hierarchy(self)
            files.extend(self.h2mil.files)
        self.deepgraphsurv = None
        if 'deepgraphsurv' in self.encoder:
            from .deepgraphsurv_data import DeepGraphSurvAssets
            self.deepgraphsurv = DeepGraphSurvAssets(self)
            files.extend(self.deepgraphsurv.files)
        self.triagemil = None
        if 'triagemil' in self.encoder:
            from .triagemil_data import TRIAGEMILAssets
            self.triagemil = TRIAGEMILAssets(self)
            files.extend(self.triagemil.files)
        self.c2mil = None
        if 'c2mil' in self.encoder:
            from .c2mil_data import C2MILAssets
            self.c2mil = C2MILAssets(self)
            files.extend(self.c2mil.files)
        self.gatmamba = None
        if 'gatmamba' in self.encoder:
            from .gatmamba_data import GATMambaAssets
            self.gatmamba = GATMambaAssets(self)
            files.extend(self.gatmamba.files)
        self.protosurv = None
        if 'protosurv' in self.encoder:
            from .protosurv_data import ProtoSurvAssets
            self.protosurv = ProtoSurvAssets(self)
            files.extend(self.protosurv.files)
        self.hdmil = None
        if 'hdmil' in self.encoder:
            from .hdmil_data import HDMILAssets
            self.hdmil = HDMILAssets(self)
            files.extend(self.hdmil.files)
        self.sparseconvmil = None
        if 'sparseconvmil' in self.encoder:
            from .sparseconvmil_data import SparseConvMILAssets
            self.sparseconvmil = SparseConvMILAssets(self, self.encoder['sparseconvmil']['manifest'])
            files.extend(self.sparseconvmil.files)
        self.pmil_prototypical = None
        if 'pmil_prototypical' in self.encoder:
            from .pmil_prototypical_data import PMILRGB
            self.pmil_prototypical = PMILRGB(self, self.encoder['pmil_prototypical']['manifest'])
            files.extend(self.pmil_prototypical.files)
        self.lnplmil = None
        if 'lnplmil' in self.encoder:
            from .lnplmil_raw import BundleInputs
            self.lnplmil = BundleInputs(self, self.encoder['lnplmil']['manifest'])
            files.extend(self.lnplmil.files)
        self.hybrid_supervision = None
        if 'hybrid_supervision' in self.encoder:
            from .hybrid_supervision_data import HybridInputs
            self.hybrid_supervision = HybridInputs(self, self.encoder['hybrid_supervision']['manifest'])
            files.extend(self.hybrid_supervision.files)
        self.dyko = None
        if 'dyko' in self.encoder:
            from .dyko_data import DyKoInputs
            self.dyko = DyKoInputs(self)
            files.extend(self.dyko.files)
        self.feather = None
        if 'feather' in self.encoder:
            from .feather_data import FEATHERInputs
            self.feather = FEATHERInputs(self)
            files.extend(self.feather.files)
        self.rmdl = None
        if 'rmdl_raw' in self.encoder:
            from .rmdl_data import RMDLInputs
            self.rmdl = RMDLInputs(self)
            files.extend(self.rmdl.files)
        self.macnet = None
        if 'macnet' in self.encoder:
            from .macnet_data import MacNetInputs
            self.macnet = MacNetInputs(self)
            files.append(self.macnet.path)
            files.extend(Path(p) for p in self.macnet.files)
        self.icmil_coupled = None
        if 'icmil_coupled' in self.encoder:
            from .icmil_coupled_data import CoupledRGB
            self.icmil_coupled = CoupledRGB(self, self.encoder['icmil_coupled']['manifest'])
            files.extend(self.icmil_coupled.files)
        self.center_embeddings = None
        if 'center_embeddings' in self.encoder:
            from .center_embeddings_data import CenterEmbeddingAssets
            self.center_embeddings = CenterEmbeddingAssets(self, self.encoder['center_embeddings']['manifest'])
            files.extend(self.center_embeddings.files)
        self.c2c = None
        if 'c2c' in self.encoder:
            from .c2c_data import C2CAssets
            self.c2c = C2CAssets(self, self.encoder['c2c']['manifest'])
            files.extend(self.c2c.files)
        self.sammil = None
        if 'sammil' in self.encoder:
            from .sammil_data import SAMMILAssets
            self.sammil = SAMMILAssets(self, self.encoder['sammil']['manifest'])
            files.extend(self.sammil.files)
        self.lrmil = None
        if 'lrmil' in self.encoder:
            from .lrmil_data import LRMILAssets
            self.lrmil = LRMILAssets(self, self.encoder['lrmil']['manifest'])
            files.extend(self.lrmil.files)
        self.kmgmamba = None
        if 'kmgmamba' in self.encoder:
            from .kmgmamba_data import KMGMambaAssets
            self.kmgmamba = KMGMambaAssets(self)
            files.extend(self.kmgmamba.files)
        self.mgtrans = None
        if 'mgtrans' in self.encoder:
            from .mgtrans_data import MGTransAssets
            self.mgtrans = MGTransAssets(self)
            files.extend(self.mgtrans.files)
        self.zoommil = None
        if 'zoommil' in self.encoder:
            from .zoommil_data import ZoomMILAssets
            self.zoommil = ZoomMILAssets(self)
            files.extend(self.zoommil.files)
        self.augdiff = None
        if 'augdiff' in self.encoder:
            from .augdiff_assets import AugDiffAssets
            self.augdiff = AugDiffAssets(self)
            files.extend(self.augdiff.files)
        self.focus = None
        if 'focus' in self.encoder:
            from .focus_data import FOCUSAssets
            self.focus = FOCUSAssets(self)
            files.extend(self.focus.files)
        self.codmil = None
        if 'codmil' in self.encoder:
            from .codmil_data import CoDAssets
            self.codmil = CoDAssets(self)
            files.extend(self.codmil.files)
        self.dispro = None
        if 'dispro' in self.encoder:
            from .dispro_data import DisProInputs
            self.dispro = DisProInputs(self)
            files.extend(self.dispro.files)
        if 'slip' in self.encoder:
            from .slip_data import SLIPInputs
            self.slip = SLIPInputs(self)
            files.extend(self.slip.files)
        if 'vlsa' in self.encoder:
            from .vlsa_data import VLSAInputs
            self.vlsa = VLSAInputs(self)
            files.extend(self.vlsa.files)
        self.nciemil = None
        if 'nciemil_selected' in self.encoder:
            from .nciemil_cache import SelectionAssets
            self.nciemil = SelectionAssets(self)
            files.extend(self.nciemil.files)
        self.taxomil = None
        if 'taxomil' in self.encoder:
            from .taxomil_data import TaxoMILInputs
            self.taxomil = TaxoMILInputs(self)
            files.extend(self.taxomil.files)
        self.muse = None
        if 'muse' in self.encoder:
            from .muse_data import MUSEInputs
            self.muse = MUSEInputs(self, self.encoder['muse']['manifest'])
            files.extend(self.muse.files)
        self.libramil = None
        if 'libramil' in self.encoder:
            from .libramil_data import LibraInputs
            self.libramil = LibraInputs(self, self.encoder['libramil']['manifest'])
            files.extend(self.libramil.files)
        self.miladapter = None
        if 'miladapter' in self.encoder:
            from .miladapter_data import MILAdapterAssets
            self.miladapter = MILAdapterAssets(self, self.encoder['miladapter']['manifest'])
            files.extend(self.miladapter.files)
        self.conmil = None
        if 'convl' in self.encoder:
            from .convl_data import ConVLInputs
            self.convl = ConVLInputs(self, self.encoder['convl']['manifest'])
            files.extend(self.convl.files)
        if 'conmil' in self.encoder:
            from .conmil_data import ConMILAssets
            self.conmil = ConMILAssets(self, self.encoder['conmil']['manifest'])
            files.extend(self.conmil.files)
        self.comel = None
        if 'comel' in self.encoder:
            from .comel_data import CoMELInputs
            self.comel = CoMELInputs(self)
            files.append(self.comel.path)
        self.akdpmp = None
        if 'akdpmp' in self.encoder:
            from .akdpmp_data import AKDInputs
            self.akdpmp = AKDInputs(self)
            files.append(self.akdpmp.path)
        self.protomil = None
        if 'protomil' in self.encoder:
            from .protomil_data import ProtoMILInputs
            self.protomil = ProtoMILInputs(self)
            files.append(self.protomil.path)
        self.kmpmil = None
        if 'kmpmil' in self.encoder:
            from .kmpmil_data import KMPInputs
            self.kmpmil = KMPInputs(self)
            files.append(self.kmpmil.path)
        self.qpmilvl = None
        if 'qpmilvl' in self.encoder:
            from .qpmilvl_data import QPMILInputs
            self.qpmilvl = QPMILInputs(self)
            files.append(self.qpmilvl.path)
        self.promptmil = None
        if 'promptmil' in self.encoder:
            from .promptmil_data import PromptInputs
            self.promptmil = PromptInputs(self, self.encoder['promptmil']['manifest'])
            files.append(self.promptmil.path)
        self.lpgmil = None
        if "lpgmil" in self.encoder:
            from .lpgmil_data import PrototypeInputs
            self.lpgmil = PrototypeInputs(self,self.encoder["lpgmil"]["manifest"])
            files.extend(self.lpgmil.files)
        self.dspamil = None
        if 'dspamil' in self.encoder:
            from .dspamil_data import DSPAInputs
            self.dspamil = DSPAInputs(self)
            files.extend(self.dspamil.files)
        self.fingerprint = digest_object({str(p.relative_to(self.root)) if p.is_relative_to(self.root)
                                          else str(p): digest_file(p) for p in files})

    def records(self, subset):
        return self.rows[self.rows.subset == subset].to_dict("records")

    def load(self, row, require_coords=False):
        if self.rmdl is not None:raise ValueError('RMDL original WSI and pixel masks require complete localization/mining/NMS/learned-FConv/MIL training')
        if getattr(self,'promptmil',None)is not None:raise ValueError('Prompt-MIL actual RGB requires the complete prompt-tuning recipe, not frozen-vector loading')
        if self.lnplmil is not None:raise ValueError('LNPL actual RGB/TRAIN-LPA needs the full weak/scoring/SP/TOD protocol')
        if self.hybrid_supervision is not None:raise ValueError('Hybrid supervision actual512RGB/fine masks require its full segmentation/EM/image-classification protocol')
        if self.macnet is not None:raise ValueError('MacNet raw RGB and expert guidance require its full end-to-end protocol')
        if self.pmil_prototypical is not None:
            raise ValueError('Prototypical PMIL original RGB256 images require the full max-MIL, AP and dual-loss pipeline')
        if self.icmil_coupled is not None:
            raise ValueError('Coupled ICMIL actual RGB256 images require its full extraction, MIL and encoder-distillation pipeline')
        if 'attention2majority_raw' in self.encoder:
            raise ValueError('Attention2majority original RGB bags require its complete native Keras discriminator/selection/MIL pipeline')
        if self.dispro is not None and not self.dispro.observed(row['global_patient_id'],'WSI'):
            raise ValueError('DisPro missing WSI has no feature tensor; use its explicit incomplete-modality interface')
        if self.center_embeddings is not None:
            raise ValueError('Center Embeddings original RGB bags are not frozen FM features; use the explicit end-to-end route')
        if self.sparseconvmil is not None:
            raise ValueError('SparseConvMIL raw spatial RGB bags are not frozen FM features')
        if self.c2c is not None:
            raise ValueError('C2C raw RGB patch bags are not frozen FM feature tensors; select its explicit end-to-end route')
        if self.raw_images is not None or self.raw_regions is not None:
            raise ValueError('Raw RGB images are not frozen patch features; select an explicit raw-image method')
        path = resolve(self.root, row["path"])
        x = load_array(path)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.dim:
            raise ValueError(f"{row['slide_id']}: expected nonempty [N,{self.dim}], got {x.shape}")
        if not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)) or not np.isfinite(x).all():
            raise ValueError(f"{row['slide_id']}: nonnumeric, nonreal or nonfinite features")
        coords = None
        if row.get("coords_path"):
            coords = load_array(resolve(self.root, row["coords_path"]), "coords")
        elif path.suffix in {".h5", ".hdf5"}:
            with h5py.File(path, "r") as f:
                if "coords" in f:
                    coords = f["coords"][:]
        if require_coords and coords is None:
            raise ValueError(f"{row['slide_id']}: real coordinates required")
        if coords is not None and (coords.shape != (len(x), 2) or
                not (np.issubdtype(coords.dtype, np.integer) or np.issubdtype(coords.dtype, np.floating)) or
                not np.isfinite(coords).all()):
            raise ValueError(f"{row['slide_id']}: coordinate/feature alignment error")
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), (
            None if coords is None else torch.from_numpy(np.asarray(coords, dtype=np.float32)))

    def cluster_labels(self, row, instances):
        if not row.get('cluster_labels_path'):
            raise ValueError(f"{row['slide_id']}: cluster_labels_path is required")
        labels = load_array(resolve(self.root, row['cluster_labels_path']), 'cluster_labels')
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.shape != (instances,) or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
            raise ValueError(f"{row['slide_id']}: cluster labels must be aligned nonnegative integer [N]")
        return torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def check(self, deep=False, require_coords=False):
        records = []
        attention_inputs = None
        if 'attention2majority_raw' in self.encoder:
            if require_coords:raise ValueError('Attention2majority raw patch identities are not a frozen spatial-feature index')
            from .attention2majority_data import AttentionInputs
            attention_inputs = AttentionInputs(self)
            attention_inputs.check(deep=deep)
        if deep and self.raw_images is not None:
            self.raw_images.check_initialization()
        if deep and self.raw_regions is not None:
            self.raw_regions.check_initialization()
        for row in self.rows.to_dict("records"):
            if self.dispro is not None:
                if require_coords:raise ValueError('DisPro incomplete patient inputs do not promise physical spatial coordinates')
                records.append(self.dispro.inventory(row,deep))
                continue
            p = resolve(self.root, row["path"])
            if not p.is_file():
                raise FileNotFoundError(f"{row['slide_id']}: {p}")
            record = {"slide_id": row["slide_id"], "path": str(p),
                      "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            if deep:
                if getattr(self,'promptmil',None)is not None:
                    if require_coords:raise ValueError('Prompt-MIL actual RGB patch identities are not frozen spatial feature coordinates')
                    raw=self.promptmil.inventory(row,deep=True)
                    record.update(promptmil_rgb=raw,sha256=raw['sha256'],instances=raw['shape'][0])
                elif attention_inputs is not None:
                    raw=attention_inputs.records[row['slide_id']]
                    record.update(attention2majority_rgb=raw,sha256=raw['sha256'],instances=raw['count'])
                elif self.pmil_prototypical is not None:
                    if require_coords:raise ValueError('PMIL raw RGB coordinates are not frozen FM spatial features')
                    raw = self.pmil_prototypical.inventory(row, deep=True)
                    record.update(pmil_prototypical_rgb=raw, sha256=raw['sha256'])
                elif self.lnplmil is not None:
                    if require_coords:raise ValueError('LNPL raw RGB coordinates are not a frozen FM feature tensor')
                    raw=self.lnplmil.inventory(row,deep=True)
                    record.update(lnplmil_rgb=raw,sha256=raw['sha256'],instances=raw['shape'][0])
                elif self.hybrid_supervision is not None:
                    if require_coords:raise ValueError('Hybrid original512RGB is not a frozen spatial FM tensor')
                    raw=self.hybrid_supervision.inventory(row,deep=True)
                    record.update(hybrid_supervision_rgb=raw,sha256=raw['sha256'])
                elif self.rmdl is not None:
                    if require_coords:raise ValueError('RMDL actual WSI geometry is not a frozen FM feature tensor')
                    raw=self.rmdl.inventory(row,deep=True)
                    record.update(rmdl_raw=raw,sha256=raw['sha256'])
                elif self.macnet is not None:
                    if require_coords:raise ValueError('MacNet raw RGB is not frozen spatial features')
                    raw=self.macnet.inventory(row,deep=True)
                    record.update(macnet_rgb=raw,sha256=raw['sha256'])
                elif self.icmil_coupled is not None:
                    if require_coords:raise ValueError('Coupled ICMIL raw RGB patch coordinates are not frozen FM spatial features')
                    raw = self.icmil_coupled.inventory(row, deep=True)
                    record.update(icmil_coupled_rgb=raw, sha256=raw['sha256'])
                elif self.center_embeddings is not None:
                    if require_coords:
                        raise ValueError('Center Embeddings raw measured coordinates are not frozen feature coordinates')
                    raw = self.center_embeddings.inventory(row, deep=True)
                    record.update(center_embeddings_rgb=raw, sha256=raw['sha256'])
                elif self.sparseconvmil is not None:
                    if require_coords:
                        raise ValueError('SparseConvMIL raw spatial RGB coordinates are distinct from frozen features')
                    raw = self.sparseconvmil.inventory(row, deep=True)
                    record.update(sparseconvmil_rgb=raw, sha256=raw['sha256'])
                elif self.c2c is not None:
                    if require_coords:
                        raise ValueError('C2C raw patch identity is distinct from a frozen spatial-feature index')
                    raw = self.c2c.inventory(row, deep=True)
                    record.update(c2c_rgb=raw, sha256=raw['sha256'])
                elif self.raw_regions is not None:
                    if require_coords:
                        raise ValueError('Raw ROI coordinates are not frozen-feature coordinates')
                    raw = self.raw_regions.inventory(row, deep=True)
                    record.update(raw_regions=raw, sha256=raw['sha256'])
                elif self.raw_images is not None:
                    if require_coords:
                        raise ValueError('Raw RGB images do not provide a frozen patch coordinate index')
                    raw = self.raw_images.inventory(row, deep=True)
                    record.update(raw_image=raw, sha256=raw['sha256'])
                else:
                    x, coords = self.load(row, require_coords)
                    record.update(instances=len(x), coords=coords is not None, sha256=digest_file(p))
                if row.get("sha256") and row["sha256"] != record["sha256"]:
                    raise ValueError(f"Feature hash mismatch: {p}")
            if row.get("coords_path"):
                cp = resolve(self.root, row["coords_path"])
                record["coords_sha256"] = digest_file(cp)
                if row.get("coords_sha256") and row["coords_sha256"] != record["coords_sha256"]:
                    raise ValueError(f"Coordinate hash mismatch: {cp}")
            if row.get('cluster_labels_path'):
                cp = resolve(self.root, row['cluster_labels_path'])
                record['cluster_labels_sha256'] = digest_file(cp)
                if row.get('cluster_labels_sha256') and row['cluster_labels_sha256'] != record['cluster_labels_sha256']:
                    raise ValueError(f'Cluster label hash mismatch: {cp}')
                if deep:
                    self.cluster_labels(row, len(x))
            if self.multiscale is not None:
                record['multiscale'] = self.multiscale.inventory(row, deep)
            if self.dual_scale is not None:
                record['dual_scale'] = self.dual_scale.inventory(row, deep)
            if self.regions is not None:
                record['regions'] = self.regions.inventory(row, deep)
            if self.smmile is not None:
                record['smmile'] = self.smmile.inventory(row, deep)
            if self.pamil is not None:
                record['pamil'] = self.pamil.inventory(row, deep)
            if self.ttis is not None:
                record['ttis'] = self.ttis.inventory(row, deep)
            if self.simil is not None:
                record['simil'] = self.simil.inventory(row, deep)
            if self.hipt is not None:
                record['hipt'] = self.hipt.inventory(row, deep)
            if self.hdmil is not None:
                record['hdmil'] = self.hdmil.inventory(row, deep)
            if self.lrmil is not None:
                record['lrmil'] = self.lrmil.inventory(row, deep)
            if self.kmgmamba is not None:
                record['kmgmamba'] = self.kmgmamba.inventory(row, deep)
            if self.mgtrans is not None:
                record['mgtrans'] = self.mgtrans.inventory(row, deep)
            if self.zoommil is not None:
                record['zoommil'] = self.zoommil.inventory(row, deep)
            if self.h2mil is not None:
                record['h2mil'] = self.h2mil.inventory(row, deep)
            if self.deepgraphsurv is not None:
                record['deepgraphsurv'] = self.deepgraphsurv.inventory(row, deep)
            if self.triagemil is not None:
                record['triagemil'] = self.triagemil.inventory(row, deep)
            if self.c2mil is not None:
                record['c2mil'] = self.c2mil.inventory(row, deep)
            if self.gatmamba is not None:
                record['gatmamba'] = self.gatmamba.inventory(row, deep)
            if self.protosurv is not None:
                record['protosurv'] = self.protosurv.inventory(row, deep)
            if self.augdiff is not None:
                record['augdiff'] = self.augdiff.inventory(row, deep)
            if self.focus is not None:
                record['focus'] = self.focus.inventory(row, deep)
            if self.codmil is not None:
                record['codmil'] = self.codmil.inventory(row, deep)
            if self.miladapter is not None:
                record['miladapter'] = self.miladapter.inventory(row, deep)
            if self.conmil is not None:
                record['conmil'] = self.conmil.inventory(row, deep)
            if self.taxomil is not None:
                record['taxomil'] = self.taxomil.protocol()
            if self.muse is not None:
                record['muse'] = self.muse.inventory(row, deep)
            if self.libramil is not None:
                record['libramil'] = self.libramil.inventory(row, deep)
            if self.convl is not None:
                record['convl'] = self.convl.inventory(row, deep)
            if self.vlsa is not None:
                record['vlsa'] = {'contract':self.vlsa.contract,'ordered_feature':self.vlsa.records[row['slide_id']]}
            records.append(record)
        if deep and self.pamoe is not None:
            self.pamoe.check_inventory(records)
        result = {"task_id": self.config["task_id"], "fm": self.fm, "split": self.split,
                "slides": len(self.rows), "patients": self.rows.global_patient_id.nunique(),
                "subsets": self.rows.subset.value_counts().to_dict(), "bundle_sha256": self.fingerprint,
                "feature_inventory_sha256": digest_object(records), "deep": deep, "features": records}
        if self.dispro is not None:
            result.update(slides=sum(self.dispro.observed(r['global_patient_id'],'WSI')
                for r in self.rows.to_dict('records')), input_slots=len(self.rows),
                patient_cases=self.rows.global_patient_id.nunique(),
                observed_wsi_patients=int(self.dispro.presence.wsi_observed.sum()),
                observed_omics_patients=int(self.dispro.presence.omics_observed.sum()),
                absent_or_masked_wsi_patients=int((self.dispro.presence.wsi_observed==0).sum()),
                patient_subsets=self.rows.drop_duplicates('global_patient_id').subset.value_counts().to_dict(),
                slide_count_scope='observed actual WSI feature records only; absent/masked patient slots are not WSIs')
        return result
