"""Measured RGB256 bags and patient-safe partitions for the full coupled pipeline."""
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .data import resolve
from .icmil_coupled_source import IMAGENET_SHA256
from .io import digest_file,digest_object,read_json,write_json

KIND='icmil_coupled_original_rgb256'


def encoder_identity(bundle):
    return digest_object({'fm':bundle.fm,'dimension':bundle.dim,
        'encoder':{k:v for k,v in bundle.encoder.items() if k!='icmil_coupled'}})


def schema_contract(bundle,schema):
    if not isinstance(schema,dict):raise ValueError("ICMIL RGB schema must be a mapping")
    if (bundle.task_type!='classification' or len(bundle.classes)!=2 or bundle.dim!=3
            or bundle.config.get('training_unit','slide')!='slide' or bundle.config.get('evaluation_unit','slide')!='slide'
            or schema.get('schema_version')!=1 or schema.get('kind')!=KIND
            or schema.get('layout')!='NHWC' or schema.get('dtype')!='uint8' or schema.get('color')!='RGB'
            or schema.get('patch_pixels')!=256 or schema.get('normalization')!='torchvision_ToTensor_only_no_normalization'
            or any(not isinstance(schema.get(k),str) or not schema[k].strip() for k in ['pixel_key','patch_ids_key','image_provenance','patch_provenance','initial_weights'])
            or schema.get('coordinate_frame') not in {'path_identity_only','level0_top_left_pixels'}
            or any(k in bundle.encoder for k in ['raw_images','raw_regions','c2c','sparseconvmil','attention2majority_raw','multiscale','vision_language'])):
        raise ValueError('Coupled ICMIL requires binary slide labels, real RGB256 uint8 images, ToTensor-only preprocessing and actual original initialization')
    if schema['coordinate_frame']=='level0_top_left_pixels' and not isinstance(schema.get('coords_key'),str):
        raise ValueError('Declared measured ICMIL coordinates require an embedded aligned dataset')
    return schema


def describe(bundle,row,schema):
    path=resolve(bundle.root,row['path'])
    if path.suffix.lower() not in {'.h5','.hdf5'}:raise ValueError('Pack original RGB256 images with real ordered patch IDs in HDF5')
    with h5py.File(path,'r') as f:
        for key in [schema['pixel_key'],schema['patch_ids_key']]:
            if key not in f or isinstance(f.get(key,getlink=True),(h5py.ExternalLink,h5py.SoftLink)):
                raise ValueError('Original RGB pixels and IDs must be embedded, not linked to unregistered files')
            value=f[key]
            if not isinstance(value,h5py.Dataset) or value.is_virtual or value.external:
                raise ValueError('ICMIL RGB and IDs cannot use external storage')
        image=f[schema['pixel_key']]
        if image.ndim!=4 or image.shape[0]<1 or image.shape[1:]!=(256,256,3) or image.dtype!=np.uint8:
            raise ValueError('ICMIL requires complete actual RGB256 images; no feature tensor or silent resize')
        ids=f[schema['patch_ids_key']].asstr()[:].tolist()
        if len(ids)!=len(image) or len(set(ids))!=len(ids) or any(not isinstance(i,str) or not i.strip() for i in ids):
            raise ValueError('ICMIL ordered patch IDs must be unique, nonempty and row aligned')
        coordinates=None
        if schema['coordinate_frame']=='level0_top_left_pixels':
            key=schema['coords_key']
            if key not in f or isinstance(f.get(key,getlink=True),(h5py.ExternalLink,h5py.SoftLink)):
                raise ValueError('Measured ICMIL coordinates must be embedded')
            coordinates=f[key][:]
            if coordinates.shape!=(len(image),2) or not np.issubdtype(coordinates.dtype,np.integer) or (coordinates<0).any() or len(np.unique(coordinates,axis=0))!=len(coordinates):
                raise ValueError('Measured ICMIL coordinate/patch row alignment is invalid')
        return {'shape':list(image.shape),'patch_ids_sha256':digest_object(ids),
            'coordinates_sha256':None if coordinates is None else digest_object(coordinates.tolist())}


def prepare(bundle,schema,output):
    schema=schema_contract(bundle,schema);output=resolve(bundle.root,output)
    if output.exists():raise FileExistsError('Prepare a new ICMIL input manifest instead of overwriting evidence')
    weights=resolve(bundle.root,schema['initial_weights']).resolve()
    if digest_file(weights)!=IMAGENET_SHA256:raise ValueError('Original ICMIL ImageNet initialization bytes differ')
    records={}
    for row in bundle.rows.to_dict('records'):
        path=resolve(bundle.root,row['path']).resolve()
        records[row['slide_id']]={**describe(bundle,row,schema),'path':str(path),'sha256':digest_file(path),
            **{k:row[k] for k in ['global_patient_id','subset','target']}}
    write_json(output,{'schema_version':1,'kind':KIND,'schema':schema,'encoder_identity':encoder_identity(bundle),
        'initial_weights':str(weights),'initial_weights_sha256':IMAGENET_SHA256,'slides':records,
        'synthetic':bool(bundle.config.get('synthetic',False)),'preparer_sha256':digest_file(Path(__file__))})
    return output


class CoupledRGB:
    def __init__(self,bundle,manifest):
        self.bundle=bundle;self.path=resolve(bundle.root,manifest);self.meta=read_json(self.path)
        if (self.meta.get('schema_version')!=1 or self.meta.get('kind')!=KIND
                or self.meta.get('encoder_identity')!=encoder_identity(bundle)
                or self.meta.get('synthetic')!=bool(bundle.config.get('synthetic',False))
                or self.meta.get('preparer_sha256')!=digest_file(Path(__file__))
                or set(self.meta.get('slides',{}))!=set(bundle.rows.slide_id)
                or self.meta.get('initial_weights_sha256')!=IMAGENET_SHA256):
            raise ValueError('Coupled ICMIL source/preparer/identity contract changed')
        self.schema=schema_contract(bundle,self.meta['schema']);self.weights=Path(self.meta['initial_weights'])
        if digest_file(self.weights)!=IMAGENET_SHA256:raise ValueError('Original coupled ICMIL initial encoder changed')
        for row in bundle.rows.to_dict('records'):
            record=self.meta['slides'][row['slide_id']]
            if record['path']!=str(resolve(bundle.root,row['path']).resolve()) or any(record[k]!=row[k] for k in ['global_patient_id','subset','target']):
                raise ValueError('ICMIL actual images, patient identities, labels or partitions changed')
        self.files=[self.path,self.weights];self.contract=digest_file(self.path);self._verified={}

    def _check(self,row):
        path=resolve(self.bundle.root,row['path']);s=path.stat();signature=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if self._verified.get(str(path))!=signature:
            expected=self.meta['slides'][row['slide_id']]
            if digest_file(path)!=expected['sha256'] or any(expected[k]!=v for k,v in describe(self.bundle,row,self.schema).items()):
                raise ValueError('ICMIL actual RGB bytes, shape or ordered patch identities changed')
            self._verified[str(path)]=signature

    @contextmanager
    def open(self,row):
        self._check(row)
        with h5py.File(resolve(self.bundle.root,row['path']),'r') as f:yield f[self.schema['pixel_key']]

    def count(self,row):return self.meta['slides'][row['slide_id']]['shape'][0]

    def inventory(self,row,deep=False):
        if deep:self._check(row)
        return {k:self.meta['slides'][row['slide_id']][k] for k in ['shape','sha256','patch_ids_sha256','coordinates_sha256']}
