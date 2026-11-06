"""Uncapped RGB extraction, stateful source bag shuffling and distillation data."""
import bisect
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader

from .icmil_coupled_source import source_views,views
from .io import digest_file,digest_object,read_json,write_json


def tensor_identity(model):
    return digest_object({k:{'shape':list(v.shape),'dtype':str(v.dtype),
        'sha256':hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
        for k,v in model.state_dict().items()})


def extract(bundle,assets,encoder,output,device,subsets=('train','val')):
    """Source448-image FP32 batches; all real patch rows, no fixed bag cap."""
    if not set(subsets)<={'train','val','test'} or not subsets:raise ValueError('Declare exact extraction partitions')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    contract={'schema_version':1,'kind':'icmil_coupled_complete_1024_features','raw_contract':assets.contract,
        'encoder_state_sha256':tensor_identity(encoder),'subsets':list(subsets),
        'source_batch_size':448,'output_dimension':1024,'normalization':'ToTensor_only_no_normalization',
        'extractor_sha256':digest_file(Path(__file__))}
    if (out/'contract.json').exists():
        if read_json(out/'contract.json')!=contract:raise ValueError('ICMIL extraction cache encoder/input/protocol changed')
    else:write_json(out/'contract.json',contract)
    encoder.eval();rows=[];convert=source_views()[0]
    for row in bundle.rows[bundle.rows.subset.isin(subsets)].to_dict('records'):
        name=hashlib.sha256(row['slide_id'].encode()).hexdigest();path=out/(name+'.npy');meta=out/(name+'.json')
        if meta.exists():
            record=read_json(meta)
            if record['raw']!=assets.inventory(row,deep=True) or record['feature_sha256']!=digest_file(path):
                raise ValueError('Previously extracted ICMIL features or original rows changed')
            x=np.load(path,mmap_mode='r',allow_pickle=False)
            if x.shape!=(assets.count(row),1024) or x.dtype!=np.float32:raise ValueError('ICMIL cache lost original patch count or1024D features')
            rows.append(record);continue
        features=[]
        with assets.open(row) as pixels:
            for start in range(0,len(pixels),448):
                images=torch.stack([convert(Image.fromarray(image)) for image in pixels[start:start+448]]).to(device)
                with torch.no_grad():values=encoder(images).cpu().numpy()
                if values.shape!=(len(images),1024) or values.dtype!=np.float32 or not np.isfinite(values).all():
                    raise ValueError('Actual full original encoder did not return complete finite1024D features')
                features.append(values)
        # Only one slide's feature matrix is assembled; RGB images and all
        # cohort features are never concatenated into one hidden memory budget.
        array=np.concatenate(features,axis=0)
        temporary=path.with_suffix('.npy.partial')
        with temporary.open('wb') as f:np.save(f,array,allow_pickle=False)
        os.replace(temporary,path)
        record={'slide_id':row['slide_id'],'global_patient_id':row['global_patient_id'],'subset':row['subset'],
            'target':row['target'],'path':str(path),'feature_sha256':digest_file(path),'shape':list(array.shape),
            'raw':assets.inventory(row,deep=True)}
        write_json(meta,record);rows.append(record)
        write_json(out/'progress.json',{'state':'running','extracted_slides':len(rows)})
    result={'state':'completed','contract':contract,'rows':rows,'all_registered_patches_used':True}
    write_json(out/'index.json',result);return result


class CachedBags(Dataset):
    """The original in-place feature shuffle, represented by a resumable row map."""
    def __init__(self,feature_index,subset,orders=None):
        self.rows=[r for r in feature_index['rows'] if r['subset']==subset]
        if not self.rows:raise ValueError('ICMIL bag partition is empty')
        self.orders={r['slide_id']:np.arange(r['shape'][0],dtype=np.int64) for r in self.rows}
        if orders is not None:
            if set(orders)!=set(self.orders):raise ValueError('Saved ICMIL mutable feature-order IDs changed')
            for k,value in orders.items():
                if value.dtype!=np.int64 or value.shape!=self.orders[k].shape or not np.array_equal(np.sort(value),self.orders[k]):
                    raise ValueError('Saved ICMIL order is not a complete patch permutation')
            self.orders={k:v.copy() for k,v in orders.items()}
        for row in self.rows:
            if digest_file(row['path'])!=row['feature_sha256']:raise ValueError('ICMIL source feature cache bytes changed')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];order=self.orders[row['slide_id']]
        np.random.shuffle(order)
        x=np.load(row['path'],mmap_mode='r',allow_pickle=False)
        return torch.from_numpy(x[order].copy()),int(row['target'])


class PatchViews(Dataset):
    """Unlabelled actual images only; TRAIN/VAL membership comes from patient splits."""
    def __init__(self,bundle,assets,subset):
        if subset not in {'train','val'}:raise ValueError('Distillation cannot read TEST images')
        self.assets=assets;self.rows=bundle.records(subset)
        self.cumulative=np.cumsum([assets.count(r) for r in self.rows]).tolist()
        if not self.cumulative or not self.cumulative[-1]:raise ValueError('ICMIL distillation partition is empty')

    def __len__(self):return self.cumulative[-1]

    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        slide=bisect.bisect_right(self.cumulative,index);previous=0 if slide==0 else self.cumulative[slide-1]
        with self.assets.open(self.rows[slide]) as pixels:image=pixels[index-previous]
        return views(image)


def bag_loader(dataset):return DataLoader(dataset,batch_size=1,shuffle=True,drop_last=False,num_workers=0)


def patch_loader(dataset):return DataLoader(dataset,batch_size=100,shuffle=True,drop_last=False,num_workers=0)


def bag_validation(model,loader,device):
    from sklearn.metrics import roc_auc_score
    model.eval();targets=[];scores=[]
    with torch.no_grad():
        for x,y in loader:
            p=torch.softmax(model(x.squeeze(0).to(device)),dim=1)
            targets.extend(y.tolist());scores.extend(p.cpu().numpy().tolist())
    return float(roc_auc_score(targets,[p[1] for p in scores]))


def distillation_validation(model,loader,device):
    # Original validation draws BOTH augmented views and collects Python-float
    # probabilities before a float64 NumPy MAE. No clinical patch labels enter.
    model.student_encoder.eval();model.student_reduction.eval();model.student_classifier.eval()
    truth=[];scores=[]
    with torch.no_grad():
        for original,view1,unused_view2 in loader:
            p=model.teacher_classifier(model.teacher_reduction(model.teacher_encoder(original.to(device))))
            z=model.student_reduction(model.student_encoder(view1.to(device)))
            q=model.student_classifier(z)
            truth.extend(torch.softmax(p,dim=-1).cpu().numpy().tolist())
            scores.extend(torch.softmax(q,dim=-1).cpu().numpy().tolist())
    # Keep native2D logits for singleton tails instead of the released
    # validation squeeze(0) that erases that batch dimension.
    return float(np.mean(np.abs(np.asarray(scores)-np.asarray(truth))))
