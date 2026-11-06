"""Public raw-image to task-trained selected-bag preparation, with resumable stage state."""
from pathlib import Path

from .io import read_json,write_json
from .nciemil_filter import FilterInputs,train_filter,WEIGHTS_SHA256
from .nciemil_cache import extract_selected
from .nciemil_capacity import native_selection_minimum,validate_selection_capacity


def prepare(data,weights,output,config,k,device='cpu',resume=False,stop_after=None):
    required={'epochs','lr','weight_decay','bag_batch_size','seed'}
    if not isinstance(config,dict) or set(config)!=required:
        raise ValueError('NcIEMIL must explicitly declare every missing pretraining setting: '+str(sorted(required)))
    # These are the very same native selection prerequisites used by cache
    # extraction. Reject an impossible cohort before creating a run, loading
    # pretrained Swin or performing any TRAIN updates, not after training it.
    native_selection_minimum(k)
    inputs=data if isinstance(data,FilterInputs) else FilterInputs(data)
    validate_selection_capacity(inputs.records,k)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=resume)
    filter_run=out/'filter';checkpoint=filter_run/'last.pt'
    done=filter_run.exists() and (filter_run/'status.json').exists() and read_json(filter_run/'status.json')['state']=='completed'
    if not done:
        train_filter(inputs,weights,WEIGHTS_SHA256,filter_run,device=device,resume=resume and checkpoint.exists(),
            stop_after=stop_after,**config)
    if read_json(filter_run/'status.json')['state']!='completed':
        write_json(out/'status.json',{'state':'interrupted','stage':'Swin_filter','formal_admission':False})
        return {'state':'interrupted','output':str(out)}
    if (out/'dataset').exists():
        raise ValueError('Existing NcIEMIL selected dataset must be reviewed explicitly; incomplete extraction is never silently reused')
    dataset=extract_selected(inputs,filter_run,weights,out/'dataset',k,device)
    write_json(out/'status.json',{'state':'completed','dataset':str(dataset),'formal_admission':False})
    return {'state':'completed','data':str(dataset),'filter_checkpoint':str(checkpoint),'k':k}
