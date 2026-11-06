"""Measured RGB256 PMIL input, preserving patient identities and actual pixels."""
from pathlib import Path

from .data import resolve
from .icmil_coupled_data import describe, CoupledRGB
from .io import digest_file, digest_object, read_json, write_json
from .pmil_prototypical_source import IMAGENET_SHA256

KIND = 'pmil_prototypical_original_rgb256'
NORMALIZATION = 'torchvision_ToTensor_then_mean0.5_std0.1_all_channels'


def identity(bundle):
    return digest_object({'fm':bundle.fm,'dimension':bundle.dim,
        'encoder':{k:v for k,v in bundle.encoder.items() if k!='pmil_prototypical'}})


def check_schema(bundle, schema):
    if (not isinstance(schema,dict) or schema.get('schema_version')!=1 or schema.get('kind')!=KIND
            or bundle.task_type!='classification' or len(bundle.classes)!=2 or bundle.dim!=3
            or bundle.config.get('training_unit','slide')!='slide' or bundle.config.get('evaluation_unit','slide')!='slide'
            or schema.get('normalization')!=NORMALIZATION or schema.get('patch_pixels')!=256
            or schema.get('dtype')!='uint8' or schema.get('layout')!='NHWC' or schema.get('color')!='RGB'
            or schema.get('coordinate_frame') not in {'path_identity_only','level0_top_left_pixels'}
            or any(not isinstance(schema.get(k),str) or not schema[k].strip()
                for k in ['pixel_key','patch_ids_key','image_provenance','patch_provenance','initial_weights'])
            or any(k in bundle.encoder for k in ['raw_images','raw_regions','icmil_coupled','c2c','sparseconvmil',
                'attention2majority_raw','multiscale','vision_language'])):
        raise ValueError('PMIL requires binary slide-level real RGB256 uint8 bags, original normalization and ResNet34 weights')
    if schema['coordinate_frame']=='level0_top_left_pixels' and not isinstance(schema.get('coords_key'),str):
        raise ValueError('PMIL measured coordinates need an actual embedded aligned coordinate dataset')
    return schema


def prepare(bundle, schema, output):
    schema=check_schema(bundle,schema);output=resolve(bundle.root,output)
    if output.exists():raise FileExistsError('Prepare a new PMIL input manifest without replacing evidence')
    weights=resolve(bundle.root,schema['initial_weights']).resolve()
    if digest_file(weights)!=IMAGENET_SHA256:raise ValueError('PMIL original ImageNet weights changed')
    rows={}
    for row in bundle.rows.to_dict('records'):
        path=resolve(bundle.root,row['path']).resolve()
        measured=describe(bundle,row,schema)
        if measured['shape'][0]<5:raise ValueError('Original PMIL needs at least five real patches per bag')
        rows[row['slide_id']]={**measured,'path':str(path),'sha256':digest_file(path),
            **{k:row[k] for k in ['global_patient_id','subset','target']}}
    write_json(output,{'schema_version':1,'kind':KIND,'schema':schema,'encoder_identity':identity(bundle),
        'initial_weights':str(weights),'initial_weights_sha256':IMAGENET_SHA256,'slides':rows,
        'preparer_sha256':digest_file(Path(__file__)),'synthetic':bool(bundle.config.get('synthetic',False))})
    return output


class PMILRGB:
    # Reuse only the already checked RGB container reader/hash verification.
    # PMIL's schema, real initializer and normalization remain independent.
    _check=CoupledRGB._check
    open=CoupledRGB.open
    count=CoupledRGB.count
    inventory=CoupledRGB.inventory

    def __init__(self,bundle,manifest):
        self.bundle=bundle;self.path=resolve(bundle.root,manifest);self.meta=read_json(self.path)
        if (self.meta.get('schema_version')!=1 or self.meta.get('kind')!=KIND
                or self.meta.get('encoder_identity')!=identity(bundle)
                or self.meta.get('synthetic')!=bool(bundle.config.get('synthetic',False))
                or self.meta.get('preparer_sha256')!=digest_file(Path(__file__))
                or set(self.meta.get('slides',{}))!=set(bundle.rows.slide_id)
                or self.meta.get('initial_weights_sha256')!=IMAGENET_SHA256):
            raise ValueError('PMIL prepared input, source reader, patients or initialization contract changed')
        self.schema=check_schema(bundle,self.meta['schema']);self.weights=Path(self.meta['initial_weights'])
        if digest_file(self.weights)!=IMAGENET_SHA256:raise ValueError('PMIL original ResNet34 weights changed')
        for row in bundle.rows.to_dict('records'):
            record=self.meta['slides'][row['slide_id']]
            if (record['path']!=str(resolve(bundle.root,row['path']).resolve())
                    or any(record[k]!=row[k] for k in ['global_patient_id','subset','target'])
                    or record['shape'][0]<5):
                raise ValueError('PMIL actual pixels, patients, partitions, labels or patch counts changed')
        self.files=[self.path,self.weights];self.contract=digest_file(self.path);self._verified={}
