"""Task-trained full-image scores, both selection tails and auditable feature cache."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data import resolve
from .io import digest_file,digest_object,read_json,write_json
from .nciemil_filter import FilterInputs,SwinFilter,WEIGHTS_SHA256
from .nciemil_source import COMMIT,discriminative_selection
from .nciemil_capacity import native_selection_minimum,validate_selection_capacity


@torch.no_grad()
def extract_selected(inputs,filter_run,weights,output,k,device='cpu'):
    """Use only a completed TRAIN-only filter; retain all real768D patch features.

    VAL/TEST are transformed by that frozen filter. Their labels never enter
    scoring or selection. The complete unsampled raw bag and all probability
    scores remain identifiable; the selected array is exactly highK then lowK.
    """
    from .provenance import software_identity
    native_selection_minimum(k)
    inputs=inputs if isinstance(inputs,FilterInputs) else FilterInputs(inputs)
    validate_selection_capacity(inputs.records,k)
    inputs.check(deep=True);run=Path(filter_run).resolve();status=read_json(run/'status.json')
    state=torch.load(run/'last.pt',map_location='cpu',weights_only=False);cfg=state['config']
    train=inputs.bundle.records('train');train_ids=[r['slide_id'] for r in train]
    if (status['state']!='completed' or not status['only_TRAIN_used_for_gradients']
            or status['trained_slides']!=train_ids or cfg['input_contract']!=inputs.contract
            or state['completed_epochs']!=cfg['epochs'] or cfg['software']!=software_identity()
            or len(state['history'])!=cfg['epochs'] or status['epochs']!=cfg['epochs']
            or any(len(row['order'])!=len(train_ids) or set(row['order'])!=set(train_ids) for row in state['history'])):
        raise ValueError('NcIEMIL extraction requires complete actual TRAIN-only filter budget and unchanged inputs/software')
    model=SwinFilter(weights,WEIGHTS_SHA256);model.load_state_dict(state['model'],strict=True);model.to(device).eval()
    if len(state['optimizer']['state'])!=len(list(model.parameters())):
        raise ValueError('NcIEMIL filter checkpoint omitted full extractor/head AdamW moments')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'features').mkdir();(out/'all_patches').mkdir()
    meta={'schema_version':1,'kind':'nciemil_tasktrained_swin_selected_cache','source_commit':COMMIT,
        'k':k,'input_contract':inputs.contract,'original_bundle':str(inputs.bundle.root),
        'filter_checkpoint':str(run/'last.pt'),'filter_checkpoint_sha256':digest_file(run/'last.pt'),
        'filter_status':status,'filter_config':cfg,'filter_protocol':state['protocol'],
        'raw_records':inputs.records,'records':{},'selection':'stable descending positive probability; firstK then lastK',
        'heldout_labels_used_for_filter_updates':False,'synthetic':bool(inputs.bundle.config.get('synthetic',False)),
        'paper_reproduction':'not_run','real_wsi_validation':'not_run'}
    index=[]
    for row in inputs.rows:
        item=inputs.records[row['slide_id']];features=[];scores=[]
        with inputs.open(row) as read:
            for start in range(0,item['count'],128):
                feature,logits=model(read(start,min(start+128,item['count'])).to(device))
                features.append(feature.cpu());scores.append(logits.sigmoid().cpu())
        features=torch.cat(features);scores=torch.cat(scores)
        selected,indices=discriminative_selection(features,scores,k)
        name=digest_object(row['slide_id']);fp=out/'features'/(name+'.npy');allp=out/'all_patches'/(name+'.npz')
        np.save(fp,selected.numpy(),allow_pickle=False)
        np.savez(allp,features=features.numpy(),positive_probabilities=scores.numpy(),selected_indices=indices.numpy())
        index.append({'slide_id':row['slide_id'],'path':str(fp.relative_to(out))})
        meta['records'][row['slide_id']]={'features':str(fp.relative_to(out)),'feature_sha256':digest_file(fp),
            'all_patches':str(allp.relative_to(out)),'all_patches_sha256':digest_file(allp),
            'selected_indices':indices.tolist(),'selected_patch_ids':[item['patch_ids'][i] for i in indices.tolist()],
            'selected_patch_ids_sha256':digest_object([item['patch_ids'][i] for i in indices.tolist()]),
            'original_count':item['count'],'count':2*k,'dimension':768,'dtype':'float32'}
    pd.DataFrame(index).to_csv(out/'features.csv',index=False)
    # Preserve the original exact patient/slide metadata and chosen split only.
    bundle=inputs.bundle
    slides=pd.read_csv(resolve(bundle.root,bundle.config.get('slides','slides.csv')),dtype=str,keep_default_na=False)
    splits=pd.read_csv(resolve(bundle.root,bundle.config['splits'][bundle.split]),dtype=str,keep_default_na=False)
    slides.to_csv(out/'slides.csv',index=False);splits.to_csv(out/'split.csv',index=False)
    config=copy.deepcopy(bundle.config);config['slides']='slides.csv';config['splits']={bundle.split:'split.csv'}
    config['encoders']={'nciemil_tasktrained_swin':{'dimension':768,'index':'features.csv',
        'nciemil_selected':{'manifest':'nciemil_selection.json'}}}
    config['defaults']={'fm':'nciemil_tasktrained_swin','split':bundle.split}
    (out/'task.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    write_json(out/'nciemil_selection.json',meta);write_json(out/'status.json',{'state':'completed','slides':len(index),'k':k})
    return out


class SelectionAssets:
    """Verify cached selected arrays against every original patch score/identity."""
    def __init__(self,bundle):
        schema=bundle.encoder['nciemil_selected'];self.bundle=bundle
        self.path=resolve(bundle.root,schema['manifest']);self.meta=read_json(self.path);m=self.meta
        if (bundle.task_type!='classification' or bundle.dim!=768 or m.get('schema_version')!=1
                or m.get('kind')!='nciemil_tasktrained_swin_selected_cache' or m.get('source_commit')!=COMMIT
                or m.get('k') not in {128,512} or m.get('heldout_labels_used_for_filter_updates') is not False
                or set(m.get('records',{}))!=set(bundle.rows.slide_id)
                or set(m.get('raw_records',{}))!=set(bundle.rows.slide_id)):
            raise ValueError('NcIEMIL needs full original TRAIN-only Swin extraction and both native selection tails')
        original=FilterInputs(m['original_bundle']);original.check(deep=True)
        if original.contract!=m['input_contract'] or original.records!=m['raw_records']:
            raise ValueError('NcIEMIL original complete raw image/patch/weak-label provenance changed')
        cp=Path(m['filter_checkpoint']);cfg=m['filter_config'];status=m['filter_status']
        if (digest_file(cp)!=m['filter_checkpoint_sha256'] or status['state']!='completed'
                or cfg['epochs']!=status['epochs'] or cfg['input_contract']!=original.contract
                or status['trained_slides']!=[r['slide_id'] for r in original.bundle.records('train')]):
            raise ValueError('NcIEMIL actual task-trained extractor checkpoint or TRAIN-only stage proof changed')
        old={r['slide_id']:r for r in original.rows}
        self.files=[self.path,cp];self.checked={}
        for row in bundle.rows.to_dict('records'):
            before=old[row['slide_id']]
            if any(row[k]!=before[k] for k in ['patient_id','global_patient_id','label','subset','target']):
                raise ValueError('NcIEMIL selected feature patient/label/split differs from original RGB bag')
            rec=m['records'][row['slide_id']];raw=m['raw_records'][row['slide_id']]
            fp=resolve(bundle.root,rec['features']);ap=resolve(bundle.root,rec['all_patches'])
            if fp.resolve()!=resolve(bundle.root,row['path']).resolve():raise ValueError('NcIEMIL selected feature path misaligned')
            if digest_file(fp)!=rec['feature_sha256'] or digest_file(ap)!=rec['all_patches_sha256']:
                raise ValueError('NcIEMIL full features/probabilities or selected cache bytes changed')
            with np.load(ap,allow_pickle=False) as arrays:
                if set(arrays.files)!={'features','positive_probabilities','selected_indices'}:
                    raise ValueError('NcIEMIL complete extraction arrays missing')
                features=torch.from_numpy(arrays['features']);scores=torch.from_numpy(arrays['positive_probabilities'])
                selected,ids=discriminative_selection(features,scores,m['k'])
                if (features.shape!=(raw['count'],768) or rec['original_count']!=raw['count']
                        or rec['count']!=2*m['k'] or ids.tolist()!=rec['selected_indices']
                        or not np.array_equal(ids.numpy(),arrays['selected_indices'])
                        or not np.array_equal(selected.numpy(),np.load(fp,allow_pickle=False))
                        or rec['selected_patch_ids']!=[raw['patch_ids'][i] for i in ids.tolist()]
                        or rec['selected_patch_ids_sha256']!=digest_object(rec['selected_patch_ids'])):
                    raise ValueError('NcIEMIL highK/lowK instances, real trained features or original patch order changed')
            self.files.extend([fp,ap]);self.checked[str(fp)]={'bytes':fp.stat().st_size,'mtime_ns':fp.stat().st_mtime_ns}
        self.contract=digest_object(m);self.k=m['k']

    def check(self):
        if digest_object(read_json(self.path))!=self.contract:raise ValueError('NcIEMIL selected-cache provenance changed')
        for p,s in self.checked.items():
            stat=Path(p).stat()
            if stat.st_size!=s['bytes'] or stat.st_mtime_ns!=s['mtime_ns']:
                raise ValueError('NcIEMIL selected task-trained feature bytes changed')
