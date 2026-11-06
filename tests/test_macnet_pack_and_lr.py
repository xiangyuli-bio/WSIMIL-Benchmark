from pathlib import Path
import h5py,numpy as np,pandas as pd,pytest,torch
from PIL import Image
from milplus.macnet_pack import pack


def test_pack_preserves_actual_RGB_pixels_order_origin_and_explicit_guidance_without_resize(tmp_path):
    rows=[];expected=[]
    for i,role in enumerate(['unlabelled','unlabelled','unlabelled','prior_tumor','prior_tumor']):
        p=tmp_path/f'patch_{i}.png';array=np.random.default_rng(i).integers(0,256,(128,128,3),dtype='uint8');Image.fromarray(array).save(p);expected.append(array)
        rows.append({'slide_id':'actual_bag','patch_id':f'actual_patch_{i}','source_slide_id':'actual_source','global_patient_id':'actual_patient','role':role,'image_path':p.name})
    table=tmp_path/'input.csv';pd.DataFrame(rows).to_csv(table,index=False);index=pack(table,tmp_path/'packed');record=pd.read_csv(index).iloc[0]
    with h5py.File(record.path) as f:
        assert f['rgb'].shape==(5,128,128,3) and np.array_equal(f['rgb'][:],np.stack(expected))
        assert list(f['patch_ids'][:])==[f'actual_patch_{i}'.encode() for i in range(5)]
        assert list(f['roles'][:])==[r['role'].encode() for r in rows]
    with pytest.raises(FileExistsError):pack(table,tmp_path/'packed')


def test_pack_rejects_implicit_channel_conversion_and_reordering(tmp_path):
    p=tmp_path/'rgba.png';Image.new('RGBA',(96,96)).save(p);row={'slide_id':'bag','patch_id':'p','source_slide_id':'s','global_patient_id':'patient','role':'unlabelled','image_path':p.name}
    table=tmp_path/'rgba.csv';pd.DataFrame([row]).to_csv(table,index=False)
    with pytest.raises(ValueError,match='actual uint8 RGB'):pack(table,tmp_path/'bad_channel')
    p=tmp_path/'rgb.png';Image.new('RGB',(96,96)).save(p);a={**row,'image_path':p.name,'role':'prior_tumor'};b={**row,'image_path':p.name};pd.DataFrame([a,b]).to_csv(table,index=False)
    with pytest.raises(ValueError,match='original annotated guides last'):pack(table,tmp_path/'bad_order')


def test_actual_source_optimizer_constructor_mutated_group_semantics_are_distinct_from_requested_schedule():
    from milplus.macnet_source import training_definitions
    schedule=training_definitions().vit_lr_schedule
    assert [schedule(i) for i in [0,49,50,74,75,99]]==[1e-5,1e-5,5e-6,5e-6,1e-6,1e-6]
    # This is an optimizer-construction semantic check, not a downsized model
    # experiment or a claim of native100epoch training.
    groups=[{'params':[torch.nn.Parameter(torch.zeros(()))]},{'params':[torch.nn.Parameter(torch.ones(()))]}]
    observed=[]
    for i in range(100):
        optimizer=torch.optim.RMSprop(groups,lr=schedule(i));observed.append([p['lr'] for p in optimizer.param_groups]);assert not optimizer.state
    assert observed==[[1e-5,1e-5]]*100
