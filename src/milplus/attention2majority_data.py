"""Patient-safe raw RGB bags and explicit discriminator training-pool evidence."""
import copy
from pathlib import Path
import h5py
import numpy as np

from .data import Bundle,resolve
from .io import digest_file,digest_object


class AttentionInputs:
    def __init__(self,path,fm=None,split=None,*,require_pool=True):
        self.bundle=path if isinstance(path,Bundle) else Bundle(path,fm,split)
        b=self.bundle;s=b.encoder.get('attention2majority_raw')
        if (not isinstance(s,dict) or s.get('schema_version')!=1 or s.get('kind')!='attention2majority_uint8_rgb_hdf5'
                or b.task_type!='classification' or b.dim!=3 or len(b.classes) not in {2,3}
                or b.config.get('training_unit','slide')!='slide' or b.config.get('evaluation_unit','slide')!='slide'
                or any(not isinstance(s.get(k),str) or not s[k].strip() for k in ['image_provenance','patch_provenance','weights'])):
            raise ValueError('Attention2majority requires explicit full original RGB HDF5, native binary/three-grade task and pretrained Keras weights')
        self.mode=s.get('mode');self.classes=len(b.classes);pixels=224 if self.mode=='camelyon_binary' else 128
        if self.mode not in {'camelyon_binary','kidney_three_grade'} or self.classes!=(2 if pixels==224 else 3):
            raise ValueError('Attention2majority task mode must match native224 binary or128 three-grade source inputs')
        self.schema=copy.deepcopy(s);self.rows=b.rows.to_dict('records');self.files=[];records={}
        self.weights=resolve(b.root,s['weights']).resolve()
        import hashlib
        from .attention2majority_source import RESNET_MD5
        if hashlib.md5(self.weights.read_bytes()).hexdigest()!=RESNET_MD5:
            raise ValueError('Attention2majority requires the full original Keras ResNet50 pretrained ImageNet no-top weights')
        self.weight_sha256=digest_file(self.weights)
        for row in self.rows:
            path=resolve(b.root,row['path']).resolve()
            with h5py.File(path,'r') as f:
                x=f['pixels'];ids=f['patch_ids'].asstr()[:].tolist()
                if (x.ndim!=4 or x.shape[1:]!=(pixels,pixels,3) or x.dtype!=np.uint8 or not len(x)
                        or len(ids)!=len(x) or len(ids)!=len(set(ids)) or any(not x.strip() for x in ids)):
                    raise ValueError('Attention2majority needs complete native RGB128/224 bags and unique aligned patch identities')
            stat=path.stat();records[row['slide_id']]={'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                'sha256':digest_file(path),'count':len(ids),'patch_ids':ids,'patch_ids_sha256':digest_object(ids)}
        if len({x['path'] for x in records.values()})!=len(records):raise ValueError('Distinct Attention2majority slides cannot alias raw image files')
        self.records=records;train={r['slide_id']:r for r in b.records('train')}
        if self.mode=='kidney_three_grade':
            self.pool=list(train);self.pool_evidence={'kind':'all_TRAIN_slides','weak_patch_label':'literal task class; no held-out updates'}
        elif not require_pool:
            # This binding is restricted to the initial TRAIN-only ImageNet
            # feature extraction used to build the Appendix-I pool. It is
            # never accepted by the discriminator/MIL training driver.
            self.pool=[];self.pool_evidence={'kind':'unselected_initial_TRAIN_features_only'}
        else:
            pool=s.get('discriminator_train_pool')
            if (not isinstance(pool,list) or not pool or len(pool)!=len(set(pool))
                    or any(x not in train for x in pool) or {train[x]['target'] for x in pool}!={0,1}):
                raise ValueError('Camelyon variant requires an explicit positive/negative discriminator pool exclusively from TRAIN slides')
            evidence=s.get('discriminator_pool_evidence')
            if not isinstance(evidence,str) or not evidence.strip():
                raise ValueError('Selected large-tumor TRAIN slides need reviewed source/ICA provenance, not arbitrary instance scores')
            ep=resolve(b.root,evidence);self.files.append(ep)
            from .io import read_json
            proof=read_json(ep)
            if (proof.get('kind') not in {'author_published_slide_pool','reviewed_large_tumor_TRAIN_pool','ICA_TRAIN_only_reference'}
                    or proof.get('selected_train_slide_ids')!=pool or proof.get('heldout_labels_used') is not False
                    or not isinstance(proof.get('review_provenance'),str) or not proof['review_provenance'].strip()):
                raise ValueError('Attention2majority discriminator pool evidence must bind its actual TRAIN identities and selection provenance')
            self.pool=pool;self.pool_evidence=proof
        for p in [b.root/'task.yaml',resolve(b.root,b.config.get('slides','slides.csv')),
                  resolve(b.root,b.config['splits'][b.split]),resolve(b.root,b.encoder['index'])]:self.files.append(p)
        self.file_hashes={str(p):digest_file(p) for p in self.files}
        self.contract=digest_object({'bundle':b.fingerprint,'schema':s,'rows':self.rows,'records':records,
            'pool':self.pool,'pool_evidence':self.pool_evidence,'weights_sha256':self.weight_sha256,'files':self.file_hashes})

    def check(self,deep=False):
        if self.bundle.rows.to_dict('records')!=self.rows or any(digest_file(p)!=h for p,h in self.file_hashes.items()):
            raise ValueError('Attention2majority task/patients/labels/splits/source pool changed')
        for r in self.records.values():
            p=Path(r['path']);s=p.stat()
            if (s.st_size!=r['bytes'] or s.st_mtime_ns!=r['mtime_ns'] or (deep and digest_file(p)!=r['sha256'])):
                raise ValueError('Attention2majority original RGB/patch-identity bytes changed')

    def export(self):
        self.check(deep=True)
        return {'schema_version':1,'input_contract':self.contract,'data':str(self.bundle.root),'mode':self.mode,
            'classes':self.bundle.classes,'rows':self.rows,'records':self.records,'pool':self.pool,
            'pool_evidence':self.pool_evidence,'file_hashes':self.file_hashes,'weights':str(self.weights),
            'weights_sha256':self.weight_sha256,'synthetic':bool(self.bundle.config.get('synthetic',False))}
