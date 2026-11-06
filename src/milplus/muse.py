"""Full released MUSE20-view recipe and recoverable noisy-expert training."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, WeightedRandomSampler

from .io import digest_object
from .muse_source import COMMIT, build_original, diagnostic_rng_port
from .muse_training import author_collate, author_training


class Bags(Dataset):
    def __init__(self, inputs, rows):
        self.inputs, self.rows = inputs, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]; feature = self.inputs.load(row)
        # Literal source Generic_MIL_Dataset creates two unused random coordinate
        # tensors even for this non-spatial method. Retain their RNG draws in
        # the original CPU/main or GPU/4-worker loader, never call them geometry.
        small_coords = torch.rand((10, 512)); large_coords = torch.rand((10, 512))
        return feature, small_coords, feature, large_coords, int(row['target'])


class TrackedSampler:
    def __init__(self, sampler):
        self.sampler, self.indices = sampler, []

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        self.indices = []
        for index in self.sampler:
            self.indices.append(int(index)); yield index


class MUSEMIL(nn.Module):
    owns_sampling = True

    def __init__(self, bundle, profile='muse_released_source', optimizer='adam',
                 weighted_sampling=False, early_stopping=True):
        super().__init__()
        if profile != 'muse_released_source' or getattr(bundle, 'muse', None) is None:
            raise ValueError('MUSE needs bound complete knowledge banks and original feature bags')
        if optimizer not in {'adam', 'sgd'} or type(weighted_sampling) is not bool or type(early_stopping) is not bool:
            raise ValueError('Invalid original MUSE optimizer/sampler/stopping configuration')
        self.bundle, self.inputs, self.profile = bundle, bundle.muse, profile
        self.settings = {'optimizer':optimizer,'weighted_sampling':weighted_sampling,'early_stopping':early_stopping}
        self.network = build_original(self.inputs.weights, self.inputs.cohort, bundle.dim)
        self.contract = digest_object({'source':COMMIT,'profile':profile,'inputs':self.inputs.contract,
                                      'settings':self.settings,'classes':bundle.classes,'dimension':bundle.dim})
        self.stop_after_epoch = 81
        self.recipe = None
        self.epoch_index, self.completed_epochs, self.optimizer_updates = -1, 0, 0
        self.train_orders, self.retrieval_history = [], []
        self.gradient_updates = {name:0 for name,_ in self.network.named_parameters()}
        self._retrievals = []
        original = self.network.retrieval_k

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            self._retrievals.append(self.network.top_k_indices.detach().cpu().tolist())
            return result

        self.network.retrieval_k = capture

    def configure_training(self, cfg):
        if (cfg['max_instances'] is not None or cfg['grad_clip'] is not None
                or cfg['selection_metric'] != 'muse_val_error' or cfg['lr'] != .0001
                or cfg['weight_decay'] != .00001 or cfg['patience'] != 20
                or type(cfg['epochs']) is not int or cfg['epochs'] < 1):
            raise ValueError('MUSE requires full source20-view CE, lr1e-4/wd1e-5, patience20/epoch>80, no generic clipping/subsampling')
        self.recipe = {k:cfg[k] for k in ['epochs','lr','weight_decay','patience','threads','selection_metric']}

    def make_optimizer(self, lr, weight_decay):
        if self.recipe is None or lr != self.recipe['lr'] or weight_decay != self.recipe['weight_decay']:
            raise ValueError('Configure complete MUSE recipe before optimizer construction')
        parameters = filter(lambda p:p.requires_grad,self.network.parameters())
        optimizer = (torch.optim.Adam(parameters,lr=lr,weight_decay=weight_decay) if self.settings['optimizer']=='adam'
                     else torch.optim.SGD(parameters,lr=lr,momentum=.9,weight_decay=weight_decay))

        def count(_optimizer, _args, _kwargs):
            self.optimizer_updates += 1
            for name, parameter in self.network.named_parameters():
                self.gradient_updates[name] += int(parameter.grad is not None)

        optimizer.register_step_pre_hook(count)
        self._optimizer = optimizer
        return optimizer

    def loader(self, subset):
        dataset = Bags(self.inputs,self.bundle.records(subset))
        if subset != 'train': sampler = SequentialSampler(dataset)
        elif self.settings['weighted_sampling']:
            counts = np.bincount([r['target'] for r in dataset.rows],minlength=2)
            weights = torch.tensor([len(dataset)/counts[r['target']] for r in dataset.rows],dtype=torch.float64)
            sampler = WeightedRandomSampler(weights,len(weights))
        else: sampler = RandomSampler(dataset)
        sampler = TrackedSampler(sampler)
        workers = 4 if next(self.network.parameters()).device.type=='cuda' else 0
        return dataset,sampler,DataLoader(dataset,batch_size=1,sampler=sampler,
                                         num_workers=workers,collate_fn=author_collate())

    def start_epoch(self, epoch):
        if type(epoch) is not int or epoch != self.completed_epochs:
            raise ValueError('MUSE epoch does not continue complete source training history')
        self.epoch_index = epoch

    def epoch(self, bundle, optimizer, device):
        if bundle is not self.bundle or self.epoch_index != self.completed_epochs:
            raise ValueError('MUSE bound source epoch mismatch')
        dataset,sampler,loader = self.loader('train')
        self.network.is_training = True; self._retrievals = []; losses = []

        def capture(_module, _inputs, outputs):
            losses.append(float(outputs[2].detach()))

        hook = self.network.register_forward_hook(capture); before = self.optimizer_updates
        try:
            author_training().train_loop(SimpleNamespace(model_type='Text_retrevial'),self.epoch_index,
                                         self.network,loader,optimizer,2)
        finally:
            hook.remove()
        expected = 20*len(dataset)
        if (len(losses)!=expected or self.optimizer_updates-before!=expected
                or len(self._retrievals)!=len(dataset) or len(sampler.indices)!=len(dataset)
                or not np.isfinite(losses).all()
                or any(len(ids)!=20 or len(set(ids))!=20 or min(ids)<0 or max(ids)>=300 for ids in self._retrievals)):
            raise ValueError('MUSE did not execute exactly20 genuine retrieved-view updates for every original TRAIN visit')
        order = [dataset.rows[i]['slide_id'] for i in sampler.indices]
        if not self.settings['weighted_sampling'] and set(order)!={r['slide_id'] for r in dataset.rows}:
            raise ValueError('MUSE original full random sampler omitted or repeated a TRAIN bag')
        self.train_orders.append(order); self.retrieval_history.append(copy.deepcopy(self._retrievals))
        self.completed_epochs += 1
        # Source reports the SUM of20 view losses per bag, averaged over bags.
        return [{'loss':sum(losses)/len(dataset)}]

    @torch.no_grad()
    def predict_subset(self, bundle, subset, device, cfg):
        if bundle is not self.bundle or subset not in {'val','test'}:
            raise ValueError('MUSE needs its bound validation/test subset')
        self.eval(); self.network.is_training = False
        dataset,sampler,loader = self.loader(subset); predictions = []
        for batch in loader:
            small,small_coords,features,coords,label = [value.to(device) for value in batch]
            probabilities,predicted,loss = self.network(x=features,label=label)
            predictions.append((probabilities[0].detach().cpu().tolist(),int(predicted.item()),float(loss)))
        if len(predictions)!=len(dataset):raise ValueError('MUSE missing complete source predictions')
        records = []
        for index,(probabilities,predicted,loss) in zip(sampler.indices,predictions):
            row = dataset.rows[index]
            record = {k:row[k] for k in ['slide_id','patient_id','global_patient_id','subset','target']}
            record.update(source_slides=row['slide_id'],muse_predicted_class=predicted,muse_ce=loss,
                          **{f'p_{i}':float(v) for i,v in enumerate(probabilities)})
            records.append(record)
        if subset=='val' and self.settings['early_stopping']:
            diagnostic_rng_port(self.network)
        return pd.DataFrame(records)

    def validation_metrics(self, frame):
        return {'muse_val_error':float((frame.muse_predicted_class.to_numpy()!=frame.target.to_numpy()).mean()),
                'muse_val_ce':float(frame.muse_ce.mean())}

    def selection_objective(self, metrics, epoch, current):
        return -metrics['muse_val_error']

    def selection_improved(self, objective, best):
        return objective>best if self.settings['early_stopping'] else True

    def get_extra_state(self):
        caches = {key:(None if getattr(self.network,key) is None else getattr(self.network,key).detach().cpu().clone())
                  for key in ['k_vectors','anti_k_vectors','top_k_indices']}
        return copy.deepcopy({'contract':self.contract,'recipe':self.recipe,'epoch_index':self.epoch_index,
            'completed_epochs':self.completed_epochs,'optimizer_updates':self.optimizer_updates,
            'gradient_updates':self.gradient_updates,'train_orders':self.train_orders,
            'retrieval_history':self.retrieval_history,'source_caches':caches,
            'source_is_training':self.network.is_training})

    def set_extra_state(self, state):
        keys = {'contract','recipe','epoch_index','completed_epochs','optimizer_updates','gradient_updates',
                'train_orders','retrieval_history','source_caches','source_is_training'}
        if (not isinstance(state,dict) or set(state)!=keys or state['contract']!=self.contract
                or (self.recipe is not None and state['recipe']!=self.recipe)
                or any(type(state[k]) is not int for k in ['epoch_index','completed_epochs','optimizer_updates'])
                or state['completed_epochs']<0 or state['epoch_index']!=state['completed_epochs']-1):
            raise ValueError('MUSE checkpoint source/recipe/input/epoch contract changed')
        ids = {r['slide_id'] for r in self.bundle.records('train')}; epochs = state['completed_epochs']
        if (len(state['train_orders'])!=epochs or len(state['retrieval_history'])!=epochs
                or state['optimizer_updates']!=epochs*len(ids)*20
                or any(len(order)!=len(ids) or not set(order)<=ids for order in state['train_orders'])
                or (not self.settings['weighted_sampling'] and any(set(order)!=ids for order in state['train_orders']))):
            raise ValueError('MUSE checkpoint omitted original TRAIN visits or twenty-view updates')
        for history in state['retrieval_history']:
            if len(history)!=len(ids) or any(len(x)!=20 or len(set(x))!=20 or min(x)<0 or max(x)>=300 for x in history):
                raise ValueError('MUSE complete original20-view retrieval history changed')
        if (set(state['gradient_updates'])!=set(self.gradient_updates)
                or any(type(n) is not int or n<0 or n>state['optimizer_updates'] for n in state['gradient_updates'].values())):
            raise ValueError('MUSE actual sparse parameter-gradient update counts changed')
        if set(state['source_caches'])!={'k_vectors','anti_k_vectors','top_k_indices'} or type(state['source_is_training']) is not bool:
            raise ValueError('MUSE source retrieval cache or phase state missing')
        for key,value in state['source_caches'].items():
            shape = (20,) if key=='top_k_indices' else (20,512)
            if epochs and (not torch.is_tensor(value) or tuple(value.shape)!=shape or not torch.isfinite(value).all()):
                raise ValueError('MUSE source full20-view cache values changed')
        for key in ['recipe','epoch_index','completed_epochs','optimizer_updates','gradient_updates','train_orders','retrieval_history']:
            setattr(self,key,copy.deepcopy(state[key]))
        device = next(self.network.parameters()).device
        for key,value in state['source_caches'].items():setattr(self.network,key,None if value is None else value.to(device))
        self.network.is_training = state['source_is_training']

    def load_state_dict(self, state_dict, strict=True, assign=False):
        fixed = ['text_base_0','text_base_1','text_base','class_without_learnable',
                 'prompt_learner.token_prefix','prompt_learner.token_suffix']
        current = self.network.state_dict()
        for name in fixed:
            key = 'network.'+name
            if key not in state_dict or not torch.equal(state_dict[key].cpu(),current[name].cpu()):
                raise ValueError('MUSE checkpoint original fixed pretrained knowledge/prompt buffers changed')
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def validate_optimizer_state(self, optimizer=None):
        optimizer = optimizer if optimizer is not None else getattr(self,'_optimizer',None)
        if optimizer is None:
            raise ValueError('MUSE optimizer must exist before its sparse state is validated')
        if self.settings['optimizer']=='adam':
            for name,parameter in self.network.named_parameters():
                count = self.gradient_updates[name]
                if count and (parameter not in optimizer.state or int(optimizer.state[parameter]['step'])!=count):
                    raise ValueError('MUSE Adam moments disagree with actual sparse-gradient update count: '+name)
                if not count and parameter in optimizer.state:
                    raise ValueError('MUSE unused parameter acquired optimizer moments: '+name)

    def protocol(self):
        return {'source_commit':COMMIT,'profile':self.profile,'inputs':self.inputs.contract,'cohort':self.inputs.cohort,
            'settings':self.settings,'recipe':self.recipe,'feature_dimension':self.bundle.dim,
            'parameters':sum(p.numel() for p in self.parameters()),'knowledge_per_class':300,'retrieved_views':20,
            'network':'Full original12layer/768hidden CONCH text path,16prompt tokens,512D8headattention,8Gaussian-noisy experts/top2 routing,source20percent auxiliary patch attention and dual logits',
            'source_initialization':'Full actual pretrained fixed language caches computed before original recursive self.apply; source reinitializes text linear/LN modules afterward, kept literally',
            'source_semantics':'Least-cosine positive-class top20 retrieval, original sorted queue with20separate updates, summed main+auxiliary logits, original auxiliary head reshape/unscaled second attention; explicit released-code condition, not a corrected paper reconstruction',
            'unused':'Original q_proj/agg branches and negative-class cache retained; negative cache is computed but not used in the released objective',
            'noise':'Original Gaussian router remains stochastic in eval; repeat inference uses the actual engine seed, without silently disabling router noise',
            'diagnostics':'No private sample I/O or diagnostic export. Exact651router draw shapes and Python seed/sample effects preserved after VAL, independently checked against isolated literal source diagnostic',
            'loader':'Original batch1/RandomSampler or explicit source weighted sampler; source unused10x512 coordinate draws retained, CPUworkers0/GPUworkers4; no physical geometry fabricated',
            'optimization':'Original constant lr1e-4/wd1e-5 Adam or explicit source SGD.9;20updates per visited TRAINbag; source200max/patience20/epoch_index>80, strict first VAL error selection',
            'completed_epochs':self.completed_epochs,'optimizer_updates':self.optimizer_updates,
            'real_wsi_validation':'not_run','paper_reproduction':'not_run','formal_admission':False}
