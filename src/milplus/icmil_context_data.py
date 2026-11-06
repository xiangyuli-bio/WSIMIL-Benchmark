"""Binary patient-safe in-context support/query bundles, with no implicit FM truncation."""
from pathlib import Path

import numpy as np

from .data import Bundle,resolve
from .io import digest_file,digest_object,read_json


class ContextInputs:
    def __init__(self,data,fm=None,split=None):
        self.bundle=data if isinstance(data,Bundle) else Bundle(data,fm,split);b=self.bundle
        if (b.task_type!='classification' or len(b.classes)!=2 or not 1<=b.dim<=25
                or b.config.get('training_unit','slide')!='slide' or b.config.get('evaluation_unit','slide')!='slide'):
            raise ValueError('Original in-context ICMIL is binary with at most25 features; use explicit TRAIN-only PCA preparation, never silently take the first25 FM channels')
        self.inventory=b.check(deep=True);self.records=b.rows.to_dict('records');counts=[]
        for r in self.records:
            x,_=b.load(r);counts.append(len(x))
        if len(set(counts))!=1:raise ValueError('Original ICMIL requires fixed-size bags without padding masks; prepare an explicitly declared bag budget')
        self.bag_size=counts[0];self.preparation=None
        schema=b.encoder.get('icmil_context')
        if schema is not None:
            if not isinstance(schema,dict) or schema.get('schema_version')!=1 or schema.get('bag_size')!=self.bag_size:
                raise ValueError('ICMIL preparation schema does not match actual fixed original bag counts')
            p=resolve(b.root,schema['manifest']);proof=read_json(p)
            if (proof['schema_version']!=1 or proof['fit_subsets']!=['train'] or proof['heldout_labels_used'] is not False
                    or proof['output_bag_size']!=self.bag_size or proof['output_dimension']!=b.dim):
                raise ValueError('ICMIL preparation must retain explicit TRAIN-only projection/instance selection evidence')
            if any(digest_file(f)!=sha for f,sha in proof['artifacts'].items()):raise ValueError('Actual ICMIL preparation artifacts changed')
            self.preparation={'manifest':str(p),'sha256':digest_file(p),'proof':proof}
        self.contract=digest_object({'bundle':b.fingerprint,'features':self.inventory['feature_inventory_sha256'],
            'bag_size':self.bag_size,'preparation':self.preparation,'rows':self.records})

    def dense(self,subset):
        rows=self.bundle.records(subset)
        features=np.stack([self.bundle.load(row)[0].numpy() for row in rows]).astype(np.float32)
        return features,np.asarray([row['target'] for row in rows],dtype=np.int64),rows

    def check(self):
        current=self.bundle.check(deep=True)
        if current['feature_inventory_sha256']!=self.inventory['feature_inventory_sha256']:
            raise ValueError('Original ICMIL support/query feature bytes or identities changed')
