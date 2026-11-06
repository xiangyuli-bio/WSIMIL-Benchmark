"""Exact TRAIN-only PathExpert quantile bins and missing-aware population standardization."""
import copy
from itertools import pairwise

import numpy as np
import pandas as pd


class PathExpertNormalization:
    def fit(self,train,bins=10):
        train=np.asarray(train,dtype=np.float64)
        if train.ndim!=2 or not len(train) or np.isinf(train).any():
            raise ValueError('TRAIN PathExpert values require a nonempty finite-or-missing matrix')
        self.edges=[]
        for values in train.T:
            valid=values[np.isfinite(values)]
            if not len(valid):
                self.edges.append([])
                continue
            _,edges=pd.qcut(valid,bins,labels=False,retbins=True,duplicates='drop')
            self.edges.append(edges.tolist())
        binned=self.bin(train)
        self.means=[];self.stds=[]
        for values in binned.T:
            valid=values[values!=-1]
            self.means.append(float(valid.mean()) if len(valid) else 0.)
            self.stds.append(float(valid.std()) if len(valid) else 0.)
        return self

    def bin(self,x):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(self.edges) or np.isinf(x).any():
            raise ValueError('PathExpert schema or finite/missing value contract differs')
        out=np.full_like(x,-1.)
        for i,edges in enumerate(self.edges):
            if len(edges)<2:
                continue
            values=np.asarray(pd.cut(x[:,i],bins=edges,labels=False,include_lowest=True))
            valid=np.isfinite(values)
            out[valid,i]=values[valid]
        return out

    def transform(self,x):
        out=self.bin(x)
        for i,(mean,std) in enumerate(zip(self.means,self.stds)):
            valid=out[:,i]!=-1
            out[valid,i]=(out[valid,i]-mean)/(std+1e-6)
        return out

    def state(self):
        return copy.deepcopy({'edges':self.edges,'means':self.means,'stds':self.stds})

    @classmethod
    def restore(cls,state):
        if (not isinstance(state,dict) or set(state)!={'edges','means','stds'} or
                not all(isinstance(state[k],list) for k in state) or not len(state['edges']) or
                len(state['means'])!=len(state['edges']) or len(state['stds'])!=len(state['edges'])):
            raise ValueError('Invalid SI-MIL normalization checkpoint schema')
        for edges,mean,std in zip(state['edges'],state['means'],state['stds']):
            if (not isinstance(edges,list) or not np.isfinite(np.asarray(edges,dtype=float)).all() or
                    any(b<=a for a,b in pairwise(edges)) or not np.isfinite([mean,std]).all() or std<0):
                raise ValueError('Invalid SI-MIL normalization checkpoint bins or moments')
        obj=cls()
        restored=copy.deepcopy(state)
        obj.edges=restored['edges'];obj.means=restored['means'];obj.stds=restored['stds']
        return obj

