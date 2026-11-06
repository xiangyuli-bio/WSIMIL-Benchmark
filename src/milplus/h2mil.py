"""Complete H2-MIL source-compatible and explicitly repaired classification recipes."""
import math
import types
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GlobalAttention, LayerNorm, LEConv, global_max_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_remaining_self_loops, scatter, softmax

from .h2mil_data import COMMIT, H2Hierarchy
from .io import digest_file, digest_object
from .upstream import load_definitions, pinned_text, verified_source


class SparseTensor:
    """Type-only API compatibility; the reviewed hierarchy supplies explicit COO edges."""
    def __init__(self,*a,**kw):raise TypeError('H2-MIL adapter accepts explicit tensor COO graph edges only')

def contract_edges(edge_index,cluster):
    width=int(cluster.max())+1
    encoded=torch.unique(cluster[edge_index[0]]*width+cluster[edge_index[1]],sorted=True)
    return torch.stack([encoded//width,encoded%width])

class WidthNorm(nn.Module):
    def __init__(self,original,widths):
        super().__init__();self.norms=nn.ModuleDict()
        for w in widths:
            if str(w) not in self.norms:self.norms[str(w)]=original if w==widths[0] else LayerNorm(w)
    def forward(self,x):return self.norms[str(x.shape[1])](x)

def author_network(dim,classes,hidden,dropout,ratio1,ratio2,corrected):
    root,sha=verified_source('h2mil')
    if sha!=COMMIT:raise ValueError('H2-MIL source commit differs from the reviewed implementation')
    scope={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'Linear':nn.Linear,'Parameter':nn.Parameter,
        'Union':Union,'Optional':Optional,'Tuple':tuple,'Callable':object,'Tensor':Tensor,'Adj':Adj,
        'OptPairTensor':OptPairTensor,'OptTensor':OptTensor,'Size':Size,'MessagePassing':MessagePassing,
        'softmax':softmax,'scatter':scatter,'SparseTensor':SparseTensor,'LEConv':LEConv,'LayerNorm':LayerNorm,
        'add_remaining_self_loops':add_remaining_self_loops,'contract_edges':contract_edges,
        'global_mean_pool':global_mean_pool,'global_max_pool':global_max_pool,'GlobalAttention':GlobalAttention}
    conv=load_definitions('h2mil','code/RAConv.py',['glorot','zeros','RAConv'],scope,
        replacements=[('.to(device)','.to(x.device)')])
    text=pinned_text(root,sha,'code/IHPool.py');start=text.index('        A = 0 * torch.ones')
    end=text.index('        batch = edge_index.new_zeros',start)
    pool=load_definitions('h2mil','code/IHPool.py',['uniform','euclidean_dist','IHPool'],scope,
        replacements=[(text[start:end],'        edge_index = contract_edges(edge_index,cluster)\n\n'),('.to(device)','.to(x.device)')])
    replacements=[]
    if corrected:
        replacements=[('out_classes,drop_out_ratio=','out_classes,n_classes=2,drop_out_ratio='),
            ('torch.nn.Linear(out_classes//2,4)','torch.nn.Linear(out_classes//2,n_classes)'),
            ('self.softmax = nn.Softmax(dim=-1)','self.softmax = nn.Identity()')]
    model=load_definitions('h2mil','code/main.py',['GCN'],{**scope,'RAConv':conv.RAConv,'IHPool':pool.IHPool},
        replacements=replacements).GCN
    net=model(in_feats=dim,n_hidden=hidden,out_classes=hidden,drop_out_ratio=dropout,
        pool1_ratio=ratio1,pool2_ratio=ratio2,mpool_method='global_mean_pool',**({'n_classes':classes} if corrected else {}))
    net.norm=WidthNorm(net.norm,[dim,hidden,hidden//2])
    for m in net.modules():
        if isinstance(m,MessagePassing):m.propagate=types.MethodType(MessagePassing.propagate,m)
    return net

class H2MIL(nn.Module):
    owns_sampling=True
    supports_instance_outputs=True
    instance_score_semantics='Original hierarchical hard memberships and resolution-conditioned fitness, not causal attention. Projection vectors have no direct loss gradient in the released hard-index implementation.'
    def __init__(self,bundle,manifest=None,profile='h2mil_repaired',hidden=256,dropout=.3,batch_size=32,ratio1=.1,ratio2=4):
        super().__init__()
        if profile not in {'h2mil_source_compat','h2mil_repaired'} or type(hidden) is not int or hidden<2 or hidden%2 or type(batch_size) is not int or batch_size<1:
            raise ValueError('H2-MIL needs a named source/repaired profile, positive batch size and even hidden width')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in [dropout,ratio1,ratio2]) or not 0<=dropout<1 or min(ratio1,ratio2)<=0:
            raise ValueError('H2-MIL dropout and pooling ratios invalid')
        if profile=='h2mil_source_compat' and len(bundle.classes)!=2:
            raise ValueError('Source H2-MIL metric path is binary despite its four-output head')
        self.bundle=bundle;self.assets=H2Hierarchy(bundle,manifest);self.profile=profile
        self.corrected=profile=='h2mil_repaired';self.batch_size=batch_size
        self.network=author_network(bundle.dim,len(bundle.classes),hidden,dropout,ratio1,ratio2,self.corrected)
        self.optimizer_steps=0;self.completed_epochs=0;self.train_auc=None;self.epochs=60
        self.contract=digest_object({'hierarchy':self.assets.contract,'profile':profile,'classes':bundle.classes,
            'dimension':bundle.dim,'hidden':hidden,'dropout':dropout,'batch_size':batch_size,'ratios':[ratio1,ratio2]})

    def configure_training(self,cfg):
        if cfg.get('max_instances') is not None or cfg.get('grad_clip') is not None or cfg['selection_metric']!='h2_val_auc' or cfg.get('patience',cfg['epochs'])<cfg['epochs']:
            raise ValueError('H2-MIL requires complete graphs, unclipped native gradients and its explicit validation AUC')
        self.epochs=cfg['epochs']
    def make_optimizer(self,lr,weight_decay):return torch.optim.Adam(self.network.parameters(),lr=lr,weight_decay=weight_decay)
    def forward_graph(self,graph):
        raw,*trace=self.network(graph)
        if not torch.isfinite(raw).all():raise ValueError('Nonfinite H2-MIL output')
        probability=raw.softmax(1) if self.corrected else torch.cat([1-raw[:,1:2],raw[:,1:2]],1)
        return {'logits':raw,'probabilities':probability,'source_output':raw,'source_prediction':raw.argmax(1),'graph_trace':trace,
            'input_node_type':graph.node_type,'input_tree':graph.node_tree,'physical_coordinates':graph.physical_coordinates}
    def forward_unit(self,bundle,rows,device,config,train=False):
        if len(rows)!=1:raise ValueError('H2-MIL does not combine unrelated WSI graph coordinate frames')
        return self.forward_graph(self.assets.load(rows[0]).to(device))
    def prediction_fields(self,result,row):
        target=torch.tensor([row['target']],device=result['source_output'].device)
        fields={'h2_native_prediction':int(result['source_prediction'].item()),
            'h2_source_loss':float(F.cross_entropy(result['source_output'],target))}
        if not self.corrected:
            fields.update({f'h2_original_p_{i}':float(v) for i,v in enumerate(result['source_output'][0])})
            fields.update(ranking_score=float(result['source_output'][0,1]),decision_rule='binary_positive_at_half')
        return fields
    def auc(self,targets,probability):
        a=np.asarray(probability)
        return float(roc_auc_score(targets,a[:,1]) if len(self.bundle.classes)==2 else roc_auc_score(targets,a,average='macro',multi_class='ovr'))
    def validation_metrics(self,frame):
        return {'h2_val_auc':self.auc(frame.target,frame[[f'p_{i}' for i in range(len(self.bundle.classes))]].to_numpy()),
            'h2_native_accuracy':float((frame.h2_native_prediction==frame.target).mean()),
            'h2_native_val_loss':float(frame.h2_source_loss.mean()),'h2_training_auc':self.train_auc}
    def selection_improved(self,objective,best):
        eligible=self.train_auc is not None and (self.train_auc>.8 or self.completed_epochs-1>self.epochs-10)
        return eligible and objective>=best
    def epoch(self,bundle,optimizer,device='cpu'):
        self.train();rows=bundle.records('train');losses=[];targets=[];probabilities=[]
        for start in range(0,len(rows),self.batch_size):
            batch=rows[start:start+self.batch_size];total=0.
            for row in batch:
                result=self.forward_unit(bundle,[row],device,{},train=True)
                total=total+F.cross_entropy(result['source_output'],torch.tensor([row['target']],device=device))
                targets.append(row['target']);probabilities.append(result['probabilities'][0].detach().cpu().numpy())
            optimizer.zero_grad();total.backward()
            if not torch.isfinite(total) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
                raise ValueError('Nonfinite native H2-MIL batch objective or gradient')
            optimizer.step();self.optimizer_steps+=1
            losses.append({'loss':float(total.detach())/len(batch),'optimization_loss':float(total.detach())})
        self.train_auc=self.auc(targets,probabilities)
        return losses
    def finish_epoch(self,bundle,device,cfg):self.completed_epochs+=1
    def get_extra_state(self):
        return {'contract':self.contract,'optimizer_steps':self.optimizer_steps,'completed_epochs':self.completed_epochs,'train_auc':self.train_auc}
    def set_extra_state(self,state):
        if (set(state)!={'contract','optimizer_steps','completed_epochs','train_auc'} or state['contract']!=self.contract or
            any(type(state[k]) is not int or state[k]<0 for k in ['optimizer_steps','completed_epochs']) or
            (state['train_auc'] is not None and (isinstance(state['train_auc'],bool) or not isinstance(state['train_auc'],(int,float)) or not math.isfinite(state['train_auc']) or not 0<=state['train_auc']<=1))):
            raise ValueError('H2-MIL input/profile/training contract differs')
        self.optimizer_steps=state['optimizer_steps'];self.completed_epochs=state['completed_epochs'];self.train_auc=state['train_auc']

    def export_instances(self,result,row,directory):
        if self.training:raise ValueError('H2-MIL structural export requires deterministic evaluation')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        path=directory/(digest_object(row['slide_id'])[:24]+'.h5')
        trace=result['graph_trace'];fine=result['input_node_type']==2
        final=trace[0][1][trace[0][0]]
        with h5py.File(path,'w') as h:
            h['coords']=result['physical_coordinates'][fine].cpu().numpy()
            h['all_node_coords']=result['physical_coordinates'].cpu().numpy()
            h['original_node_type']=result['input_node_type'].cpu().numpy();h['original_parent']=result['input_tree'].cpu().numpy()
            h['fine_final_cluster']=final[fine].cpu().numpy();h['all_nodes_final_cluster']=final.cpu().numpy()
            for stage in range(2):
                group=h.create_group(f'stage_{stage}')
                for name,value in zip(['membership','node_type_after','fitness_before','parent_after','coordinates_after','edges_after'],trace):
                    group[name]=value[stage].detach().cpu().numpy()
            h.attrs['semantics']=self.instance_score_semantics;h.attrs['profile']=self.profile
        return {'slide_id':row['slide_id'],'path':str(path),'sha256':digest_file(path),'instances':int(fine.sum()),
            'nodes':len(fine),'stages':2,'final_clusters':int(final.max())+1}

    def protocol(self):
        return {'profile':self.profile,'source_commit':COMMIT,'hierarchy_contract':self.assets.contract,
            'levels':'actual thumbnail, coarse and fine images in one encoder space, true twofold physical grids and parent/eight-neighbor edges',
            'model':'both original RAConv and IHPool stages, sum of global mean readouts, original MLP; explicit shape-specific PyG LayerNorm repair',
            'loss':'original CE on four softmax outputs, binary source metric path' if not self.corrected else 'explicit C-way classifier and CE on logits repair',
            'prediction':'native four outputs/argmax retained; standardized binary reports use exact column1 rank and complementary probability' if not self.corrected else 'standard C-way softmax',
            'optimizer':'native Adam5e-5/wd5e-4, summed per-graph CE, fixed contiguous batches, no shuffle/scheduler/clipping',
            'selection':'last tied validation slide AUC, eligible only when pre-update TRAIN AUC>0.8 or zero-based epoch>epochs-10; TEST sealed',
            'pool_gradient_boundary':'released fitness projection and unused pool layers have no direct loss gradient; no invented straight-through estimator',
            'compatibility':['source-defined modules from pinned Git objects; generic PyG propagate and builtin scatter API bridge',
                'algebraically exact sparse edge contraction replaces quadratic dense binary adjacency; device allocations follow input tensors'],
            'epochs':self.epochs,'optimizer_steps':self.optimizer_steps,'completed_epochs':self.completed_epochs,
            'paper_recipe_boundary':'source default60epochs, four-way softmax-before-CE and ratio2=4 differ from published settings; no paper-score claim',
            'gpu_validation':'not_run','real_wsi_validation':'not_run','paper_reproduction':'not_run'}
