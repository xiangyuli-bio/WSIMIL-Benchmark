"""Complete paper Eq1--4/9 Dyn-MoE with original full TransMIL adapters.

Hard Argmax/ArgTopK is kept. Router parameters consequently have no task
gradient; no straight-through or soft-routing substitute is inserted.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .mexd_source import components

COHORTS={'camelyon16':{'classes':2,'experts':2,'observed_background':0},
         'tcga_nsclc':{'classes':2,'experts':3,'observed_background':None},
         'bracs':{'classes':3,'experts':4,'observed_background':None}}


class DynamicExperts(nn.Module):
    def __init__(self,feature_dim,cohort,classes=None,observed_background=None):
        super().__init__()
        if cohort=='class_port':
            if type(classes)is not int or classes<2 or observed_background not in {None,0}:
                raise ValueError('MExD class port needs at least2classes and an explicit optional class0background')
            spec={'classes':classes,'experts':classes if observed_background==0 else classes+1,
                  'observed_background':observed_background}
        elif cohort in COHORTS:spec=COHORTS[cohort]
        else:raise ValueError('Select an explicit MExD cohort/expert label contract')
        self.cohort=cohort;self.spec=spec
        self.classes=self.spec['classes'];self.experts=self.spec['experts']
        original=components()[0]
        self.trans1=original(n_classes=self.classes,input_size=feature_dim,confounder_path=None)
        # Original width512, two8head/256landmark layers and full7/5/3PPEG.
        self.agg1=original(n_classes=self.classes,input_size=512,confounder_path=None)
        self.routers=nn.ModuleList([nn.Linear(512,2,bias=False)for _ in range(self.experts)])
        self.classifier=nn.ModuleList([nn.Linear(512,self.experts)for _ in range(self.experts)])
        # Keep known source-defined, inactive tensors; no hidden model thinning.
        self.confounder_W_q=nn.Linear(512,128);self.confounder_W_k=nn.Linear(512,128)
        self._fc2=nn.Linear(1024,self.classes);self.norm2=nn.LayerNorm(1024)

    def forward(self,features):
        first=self.trans1(features)
        # Eq1 has one output per original instance. Source TransMIL internally
        # pads its pseudo-grid; these copies are not extra observed instances.
        transformed=first['patch_feature'][0,:len(features),:]
        means=[];logits=[];probabilities=[];selected=[];sparse=[]
        counts=[];router_scores=[]
        for index,router in enumerate(self.routers):
            probability=router(transformed).softmax(-1)
            eta=0 if index==0 else 1
            assigned=torch.where(probability.argmax(-1)==eta)[0]
            if len(assigned):
                scores=probability[assigned,eta]
                ratio=.25 if index==0 else .5
                count=max(1,int(ratio*len(assigned)))
                order=torch.argsort(scores,descending=True,stable=True)[:count]
                chosen=assigned[order]
                values=transformed.index_select(0,chosen)
                mean=values.mean(0);sparse.append(values)
            else:
                chosen=torch.empty(0,dtype=torch.long,device=features.device)
                mean=transformed.new_zeros(512)
                # Explicit source-style empty-expert sentinel, never counted or
                # exported as an observed patch. No balancing/forced reassignment.
                sparse.append(mean.unsqueeze(0))
            prediction=self.classifier[index](mean)
            means.append(mean);logits.append(prediction);probabilities.append(prediction.softmax(-1))
            selected.append(chosen);counts.append(len(chosen));router_scores.append(probability)
        means=torch.stack(means);expert_logits=torch.stack(logits)
        expert_prob=torch.stack(probabilities)
        confidence=expert_prob.diagonal()
        # Eq8 is a sum, with no confidence renormalization.
        condition=(means*confidence[:,None]).sum(0)
        sparse=torch.cat(sparse,0)
        prior=self.agg1(sparse)
        return {'logits':prior['logits'],'probabilities':prior['Y_prob'],
            'condition':condition,'expert_logits':expert_logits,'expert_probabilities':expert_prob,
            'expert_confidence':confidence,'expert_features':means,'selected_rows':selected,
            'selected_counts':counts,'router_probabilities':router_scores,
            'original_instances':len(features),'source_grid_instances':first['patch_feature'].shape[1],
            'observed_sparse_instances':sum(counts),'empty_expert_sentinels':sum(n==0 for n in counts)}

    def loss(self,result,target):
        label=int(target.item());bag=F.cross_entropy(result['logits'],target.reshape(1))
        selected=result['selected_counts'];logits=result['expert_logits']
        negative=bag.new_zeros(())
        if selected[0]:negative=F.cross_entropy(logits[0:1],target.new_tensor([0]))
        positive=bag.new_zeros(())
        expert=(label if self.spec['observed_background']is not None else label+1)
        if expert!=0 and selected[expert]:
            positive=F.cross_entropy(logits[expert:expert+1],target.new_tensor([expert]))
        return bag+negative+positive,{'prior_ce':bag,'negative_expert_ce':negative,'matched_positive_expert_ce':positive}
