"""DGMIL dynamic density refinement; frozen FM input, genuine instance recipe."""
import copy
import heapq

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

from .upstream import load_definitions

PROFILES = {'dgmil', 'dgmil_source_extremes'}


def author_definitions(dim):
    import faiss
    scope = {'nn': nn, 'F': F, 'np': np, 'faiss': faiss, 'heapq': heapq}
    model = load_definitions('dgmil', 'model.py', ['Linear_projection_MAE'], scope,
                             replacements=[('768', str(dim))]).Linear_projection_MAE
    clusters = load_definitions('dgmil', 'pos_score_calculator.py', ['get_clusters'], scope).get_clusters
    select = load_definitions('dgmil', 'preprocessor.py',
        ['sort_and_generate_feats_bank_forpos_dynamic', 'sort_and_generate_feats_bank_forneg_dynamic'], scope)
    return model, clusters, select


def fit_density(features, clusters):
    """Cache the author's row-normalization, negative statistics and cluster inverse covariances."""
    if (features.ndim != 2 or len(features) < max(2, clusters) or
            not np.isfinite(features).all()):
        raise ValueError('DGMIL needs enough finite negative instances for the requested clusters')
    x = np.array(features, dtype=np.float32, copy=True)
    x /= np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
    mean, std = x.mean(0, keepdims=True), x.std(0, keepdims=True)
    x = (x-mean)/(std+1e-10)
    import faiss
    faiss.omp_set_num_threads(torch.get_num_threads())
    _, get_clusters, _ = author_definitions(x.shape[1])
    labels = get_clusters(x, clusters) if clusters > 1 else np.zeros(len(x), dtype=np.int64)
    centers, inverse, counts = [], [], []
    for group in np.unique(labels):
        values = x[labels == group]
        if len(values) < 2:
            raise ValueError('DGMIL singleton density cluster; reduce clusters or provide more negative patches')
        cov = np.atleast_2d(np.cov(values.T, bias=True))
        if not np.isfinite(cov).all() or not np.any(cov):
            raise ValueError('DGMIL zero/nonfinite cluster covariance; feature distribution is degenerate')
        centers.append(values.mean(0, keepdims=True))
        inverse.append(np.linalg.pinv(cov))
        counts.append(len(values))
    return {'mean': mean, 'std': std, 'centers': centers, 'inverse': inverse,
            'counts': counts, 'labels': labels, 'fitted_instances': len(x)}


def density_scores(state, features, block_size=8192):
    if state is None:
        raise ValueError('DGMIL density is not prepared')
    if (features.ndim != 2 or features.shape[1] != state['mean'].shape[1] or
            not len(features) or not np.isfinite(features).all()):
        raise ValueError('DGMIL requires nonempty finite instances in the fitted feature space')
    result=[]
    for start in range(0,len(features),block_size):
        x=np.array(features[start:start+block_size],dtype=np.float32,copy=True)
        x /= np.linalg.norm(x,axis=-1,keepdims=True)+1e-10
        x=(x-state['mean'])/(state['std']+1e-10)
        distances=[np.sum((x-c)*(inv.dot((x-c).T)).T,axis=-1)
                   for c,inv in zip(state['centers'],state['inverse'])]
        result.append(np.min(distances,axis=0))
    scores=np.concatenate(result)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite DGMIL Mahalanobis scores')
    return scores


class DGMIL(nn.Module):
    def __init__(self,bundle,profile='dgmil',clusters=10,projection_batch=65536):
        super().__init__()
        if profile not in PROFILES or bundle.task_type!='classification' or len(bundle.classes)!=2:
            raise ValueError('DGMIL currently requires an explicitly binary classification task')
        if any(type(v) is not int or v<1 for v in (clusters,projection_batch)):
            raise ValueError('DGMIL clusters and projection_batch must be positive integers')
        self.dim,self.profile,self.clusters=bundle.dim,profile,clusters
        self.projection_batch=projection_batch
        model,_,self.selector=author_definitions(bundle.dim)
        self.network=model()
        self.density,self.selected,self.fit_info=None,None,{}
        self.calibration=None
        self.optimizer_steps,self.iteration=0,0
        self.seed=2026
        self._banks=None

    def configure_training(self,config):
        if config['max_instances'] is not None:
            raise ValueError('DGMIL density uses all training patches; global max_instances is not supported')

    def training_banks(self,bundle):
        if self._banks is None:
            arrays=[[],[]];ledger=[]
            for row in sorted(bundle.records('train'),key=lambda r:r['slide_id']):
                x,_=bundle.load(row)
                label=row['target']
                arrays[label].append(x)
                ledger.append({'slide_id':row['slide_id'],'global_patient_id':row['global_patient_id'],
                               'target':label,'instances':len(x)})
            if any(not a for a in arrays):
                raise ValueError('DGMIL requires both negative and positive training slides')
            if self.fit_info and self.fit_info['training_inventory']!=ledger:
                raise ValueError('DGMIL training instance order or identity changed')
            self._banks=[torch.cat(a) for a in arrays]
            if min(map(len,self._banks))<20:
                raise ValueError('DGMIL 10% extreme selection requires at least 20 patches in each class')
            self.fit_info={'training_inventory':ledger,'instances':[len(x) for x in self._banks]}
        return self._banks

    @torch.no_grad()
    def prepare(self,bundle,seed,device):
        self.seed=seed
        banks=self.training_banks(bundle)
        self.refit(banks,project=False,device=device)

    def refit(self,banks,project,device):
        # These resets mirror get_score_and_dis_feats. NumPy state continues between iterations.
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(self.seed)
        values=[torch.cat([self.project(part.to(device)).cpu() for part in x.split(self.projection_batch)]).numpy()
                if project else x.numpy() for x in banks]
        self.density=fit_density(values[0],self.clusters)
        distances=[density_scores(self.density,x) for x in values]
        selected=[]
        for label,(x,scores) in enumerate(zip(banks,distances)):
            n=int(.1*len(x))
            if self.profile=='dgmil_source_extremes':
                fn=(self.selector.sort_and_generate_feats_bank_forpos_dynamic if label else
                    self.selector.sort_and_generate_feats_bank_forneg_dynamic)
                indices=fn(scores,np.arange(len(x),dtype=np.int64))
            else:
                indices=np.argsort(-scores if label else scores,kind='stable')[:n]
            selected.append(indices)
        self.selected=selected
        self.iteration+=int(project)

    @torch.no_grad()
    def project(self,x):
        # Only the linear projection goes into density fitting, not BatchNorm/ReLU/classifier.
        if x.ndim != 2 or x.shape[1] != self.dim or not len(x) or not torch.isfinite(x).all():
            raise ValueError('DGMIL requires finite nonempty [patches, FM_dim] input')
        return torch.cat([self.network.projection_head(part) for part in x.split(self.projection_batch)])

    def selected_training_batch(self,bundle,device):
        banks=self.training_banks(bundle)
        if self.selected is None:raise ValueError('DGMIL must initialize extreme instances before training')
        data=torch.cat([x[torch.from_numpy(idx)] for x,idx in zip(banks,self.selected)])
        labels=torch.cat([torch.full((len(idx),),label,dtype=torch.long) for label,idx in enumerate(self.selected)])
        indices,_=train_test_split(np.arange(len(data)),test_size=.1,random_state=12345,shuffle=True)
        if len(indices)<2 or len(set(labels[indices].tolist()))!=2:
            raise ValueError('DGMIL selected training subset has too few instances or loses a class')
        return data[indices].to(device),labels[indices].to(device)

    def make_optimizer(self,lr,weight_decay):
        return torch.optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)

    @torch.no_grad()
    def finish_epoch(self,bundle,device,config):
        self.eval()
        self.refit(self.training_banks(bundle),project=True,device=device)
        # Auxiliary probability reporting; ranking metrics always use the untouched mean distance.
        from .engine import load_unit, units
        scores=[];targets=[]
        for rows in units(bundle,'train'):
            x,_=load_unit(bundle,rows,device)
            scores.append(self.raw_scores(x).mean())
            targets.append(rows[0]['target'])
        a=np.array(scores,dtype=np.float64)
        mean,scale=float(a.mean()),float(a.std())
        if scale==0:scale=1.
        fitted=LogisticRegression(C=1.,solver='lbfgs',random_state=0,max_iter=1000).fit(((a-mean)/scale)[:,None],targets)
        self.calibration={'mean':mean,'scale':scale,'coef':float(fitted.coef_[0,0]),
                          'intercept':float(fitted.intercept_[0])}
        if not all(np.isfinite(v) for v in self.calibration.values()):
            raise ValueError('Nonfinite training-only DGMIL probability calibration')

    @torch.no_grad()
    def raw_scores(self,x):
        return density_scores(self.density,self.project(x).cpu().numpy())

    def forward(self,x,coords=None):
        if self.calibration is None:raise ValueError('DGMIL prediction requires a completed refinement epoch')
        raw=self.raw_scores(x)
        score=torch.tensor([raw.mean()],dtype=torch.float64,device=x.device)
        c=self.calibration
        z=(score-c['mean'])/c['scale']*c['coef']+c['intercept']
        p=z.sigmoid()
        return {'logits':torch.stack([torch.zeros_like(z),z],-1),
                'probabilities':torch.stack([1-p,p],-1),'ranking_score':score,
                'instance_scores':torch.as_tensor(raw,device=x.device)}

    def prediction_fields(self,result,row):
        return {'ranking_score':float(result['ranking_score'].item())}

    def get_extra_state(self):
        return copy.deepcopy({'density':self.density,'selected':self.selected,'fit_info':self.fit_info,
                              'calibration':self.calibration,'optimizer_steps':self.optimizer_steps,
                              'iteration':self.iteration,'seed':self.seed})

    def set_extra_state(self,state):
        for key in ('density','selected','fit_info','calibration','optimizer_steps','iteration','seed'):
            setattr(self,key,copy.deepcopy(state[key]))
        self._banks=None

    def protocol(self):
        return {'profile':self.profile,'source_network':'Linear_projection_MAE','dimension':self.dim,
                'clusters':self.clusters,'feature_input':'frozen FM; MAE image pretraining is external',
                'selection':'top positive / bottom negative 10%; source heapq first match' if self.profile.endswith('_source_extremes') else 'top positive / bottom negative 10%; stable unique indices on ties',
                'recipe':'one full selected-instance CE Adam update each epoch; then rebuild negative density and next pseudo labels',
                'internal_holdout':'source 10% selected-instance random split seed12345; not patient validation',
                'density':'negative TRAIN row L2 normalization, mean/std, Faiss100iter, cluster covariance pinv; raw first, linear projection subsequently',
                'inference_repair':'fixed training density; never fit on test/val patches',
                'ranking_metrics':'mean raw Mahalanobis score, uncalibrated; patient aggregation means per-bag scores',
                'probability_reporting':'additional train-only standardized LogisticRegression C1 for threshold/log-loss metrics; does not enter AUROC/AUPRC',
                'limits':'all training patches retained in CPU memory; selected batch on device; no implicit patch subsampling',
                'training_inventory':self.fit_info,'optimizer_steps':self.optimizer_steps,'iteration':self.iteration}


def dgmil_epoch(model,bundle,groups,optimizer,device,config):
    x,y=model.selected_training_batch(bundle,device)
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=F.cross_entropy(model.network(x),y)
    if not torch.isfinite(loss):raise ValueError('Nonfinite DGMIL classifier loss')
    loss.backward()
    if config.get('grad_clip') is not None:
        nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
    elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('Nonfinite DGMIL classifier gradient')
    optimizer.step();model.optimizer_steps+=1
    return [{'loss':float(loss.detach()),'selected_instances':len(x)}]
