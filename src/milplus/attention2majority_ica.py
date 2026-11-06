"""Explicit Appendix-I reference selector using actual initial ImageNet features.

The paper does not release its ICA implementation, whitening convention or
normal-reference slide identity. This reference names those conditions and
retains all actual TRAIN features and fit parameters; it is not a claim that
the historical twelve-slide selection has been exactly reproduced.
"""
import warnings
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from .io import digest_object


def select_large_tumor_train_pool(features,rows,template_slide,normal_reference,*,seed=2026,
                                 component_space,threshold=.1,positive_limit=12):
    train={r['slide_id']:r for r in rows if r['subset']=='train'}
    if (len({r['slide_id'] for r in rows})!=len(rows) or set(features)!=set(train)
            or template_slide not in train or normal_reference not in train
            or train[template_slide]['target']!=1 or train[normal_reference]['target']!=0
            or component_space not in {'unmixing_rows','mixing_columns'}
            or type(seed) is not int or not 0<threshold<1 or type(positive_limit) is not int or positive_limit<1):
        raise ValueError('Appendix-I reference requires complete TRAIN-only features, explicit positive template/normal reference and ICA convention')
    for name,x in features.items():
        if x.ndim!=2 or x.shape[1]!=1024 or len(x)<2 or not np.isfinite(x).all():
            raise ValueError('ICA selection needs actual complete native1024D initial ImageNet features for every TRAIN slide')
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        ica=FastICA(n_components=2,whiten='unit-variance',random_state=seed,max_iter=1000,tol=1e-4)
        ica.fit(np.asarray(features[template_slide],dtype=np.float64))
    components=ica.components_ if component_space=='unmixing_rows' else ica.mixing_.T
    if components.shape!=(2,1024) or not np.isfinite(components).all():raise ValueError('ICA did not produce two finite full1024D components')
    def distances(x):
        sums=np.zeros(2,dtype=np.float64);normal_votes=0
        for start in range(0,len(x),512):
            chunk=np.asarray(x[start:start+512],dtype=np.float64)
            value=np.square(chunk[:,None,:]-components[None,:,:]).sum(2)
            sums+=np.sqrt(value).sum(0)
        return sums/len(x)
    normal_component=int(distances(features[normal_reference]).argmin());tumor_component=1-normal_component
    fractions={}
    for slide,row in train.items():
        if row['target']!=1:continue
        x=features[slide];votes=0
        for start in range(0,len(x),512):
            value=np.square(np.asarray(x[start:start+512],dtype=np.float64)[:,None,:]-components[None,:,:]).sum(2)
            votes+=int((value.argmin(1)==tumor_component).sum())
        fractions[slide]=votes/len(x)
    eligible=[name for name in fractions if fractions[name]>=threshold]
    selected=sorted(eligible,key=lambda n:(-fractions[n],n))[:positive_limit]
    if not selected:raise ValueError('No TRAIN-positive slide meets the declared ICA tumor-fraction threshold; no fabricated selection fallback')
    return {'kind':'ICA_TRAIN_only_reference','selected_positive_slide_ids':selected,
        'template_slide_id':template_slide,'normal_reference_slide_id':normal_reference,
        'heldout_labels_used':False,'component_space':component_space,'whiten':'unit-variance',
        'components':components.tolist(),'mixing':ica.mixing_.tolist(),'unmixing':ica.components_.tolist(),
        'mean':ica.mean_.tolist(),'iterations':ica.n_iter_,'seed':seed,'threshold':threshold,
        'positive_limit':positive_limit,'normal_component':normal_component,'tumor_component':tumor_component,
        'all_TRAIN_tumor_fractions':fractions,'eligible_positive_slide_ids':eligible,
        'TRAIN_slide_identity_sha256':digest_object(list(train)),
        'reference_choices':'Declared modernICA whiten/component space and top-fraction cap; absent author implementation must not be treated as historical exact reproduction'}
