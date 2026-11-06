"""Incomplete-modality contracts, full patient inputs and leakage-free RNA stats."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from milplus.data import Bundle
from milplus.demo import create_demo
from milplus.engine import units
from milplus.registry import check_compatibility, inspect_method


def fixture(path, normalization='train_standard'):
    root = create_demo(path, patients=30, dimension=16, survival=True)
    config = yaml.safe_load((root/'task.yaml').read_text())
    presence = []
    patients = pd.read_csv(root/'split.csv').patient_id.tolist()
    for i,gid in enumerate(patients):
        presence.append({'global_patient_id':gid,'wsi_observed':int(i%4!=2),'omics_observed':int(i%4!=1)})
    pd.DataFrame(presence).to_csv(root/'presence.csv',index=False)
    columns = ['gene_C','gene_A','gene_D','gene_B','gene_F','gene_E']
    values = pd.DataFrame(np.arange(len(patients)*len(columns)).reshape(len(patients),len(columns)),columns=columns)
    values.insert(0,'global_patient_id',patients)
    # Naturally missing assays need not have a CSV row at all.
    observed = {p['global_patient_id'] for p in presence if p['omics_observed']}
    values.loc[values.global_patient_id.isin(observed)].to_csv(root/'expression.csv',index=False)
    groups = [{'name':'pathway_first','features':['gene_D','gene_C']},
              {'name':'pathway_second','features':['gene_F','gene_B','gene_A']}]
    (root/'pathways.json').write_text(json.dumps(groups))
    meta = {'schema_version':1,'kind':'dispro_incomplete_patient_inputs','synthetic':True,
        'presence':'presence.csv','expression':'expression.csv','pathways':'pathways.json',
        'gene_columns':columns,'normalization':normalization,
        'missingness_provenance':'Deterministic synthetic presence flags; no biological claims',
        'assay_preprocessing':'Synthetic consecutive integers, not genuine RNA assays',
        'pathway_provenance':'Synthetic ordered test gene groups'}
    (root/'dispro_inputs.json').write_text(json.dumps(meta))
    config['encoders']['toy']['dispro'] = {'manifest':'dispro_inputs.json'}
    (root/'task.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    # Missing cases keep identity metadata only. There is no array at this URI.
    index = pd.read_csv(root/'features.csv')
    slides = pd.read_csv(root/'slides.csv').set_index('slide_id')
    absent = {p['global_patient_id'] for p in presence if not p['wsi_observed']}
    for i,row in index.iterrows():
        gid = slides.loc[row.slide_id,'patient_id']
        if gid in absent:index.loc[i,'path'] = 'missing://wsi/'+gid
    index.to_csv(root/'features.csv',index=False)
    return root


def test_public_demo_retains_all331_pathway_sizes_and_explicit_missingness(tmp_path):
    from milplus.dispro_demo import create_dispro_demo,public_pathway_sizes
    sizes=public_pathway_sizes()
    assert (len(sizes),sum(sizes),min(sizes),max(sizes))==(331,9745,1,199)
    root=create_dispro_demo(tmp_path/'standalone_demo',patients=30,dimension=16,pathway_sizes=[2,3],instances=9)
    b=Bundle(root);inventory=b.check(deep=True)
    assert len(b.dispro.pathways)==2 and [len(p['features']) for p in b.dispro.pathways]==[2,3]
    assert b.config['synthetic'] and inventory['patient_cases']==30
    assert inventory['absent_or_masked_wsi_patients']>0
    observed=next(r for r in b.rows.to_dict('records') if b.dispro.observed(r['global_patient_id'],'WSI'))
    assert len(b.load(observed)[0])==9 and int(observed['width_level0'])==9*256
    assert len(b.dispro.values)==sum(b.dispro.observed(p,'Omics') for p in b.dispro.presence.index)


def test_registered_DisPro_rejects_features_only_survival_and_mixed_demo_flags(tmp_path):
    from milplus.cli import dispatch,parser
    from milplus.engine import train
    root=create_demo(tmp_path/'complete',patients=30,survival=True)
    assert inspect_method('DisPro')['id']=='dispro_source_repaired'
    with pytest.raises(ValueError,match='explicit incomplete-modality'):
        train(root,'DisPro',tmp_path/'refused')
    assert not (tmp_path/'refused').exists()
    args=parser().parse_args(['demo','--dispro','--amiml','--output',str(tmp_path/'mixed')])
    with pytest.raises(ValueError,match='no mixed demos'):dispatch(args)


def test_public_dispro_CLI_demo_creates_all331_ordered_groups_without_BioBERT_load(tmp_path):
    from milplus.cli import dispatch,parser
    result=dispatch(parser().parse_args(['demo','--dispro','--patients','18','--output',str(tmp_path/'full')]))
    bundle=Bundle(result['created'])
    assert len(bundle.dispro.pathways)==331
    assert sum(len(p['features']) for p in bundle.dispro.pathways)==9745
    assert bundle.dim==1024 and bundle.task_type=='survival'
    assert bundle.config['synthetic'] and bundle.check(deep=True)['patient_cases']==18


def test_real_absence_inventory_does_not_create_or_load_phantom_features(tmp_path):
    b = Bundle(fixture(tmp_path/'data'))
    result = b.check(deep=True)
    missing = [r for r in result['features'] if not r['wsi_observed']]
    assert result['slides']==sum(bool(r['wsi_observed']) for r in result['features'])
    assert result['slides']+len(missing)==result['input_slots']==len(b.rows)
    assert result['patient_cases']==30 and sum(result['patient_subsets'].values())==30
    assert result['observed_wsi_patients']+result['absent_or_masked_wsi_patients']==30
    assert missing and all(r['feature_kind']=='explicit_missing_or_masked_WSI_no_array' for r in missing)
    assert all('sha256' not in r and r['path'].startswith('missing://wsi/') for r in missing)
    for subset in ['train','val','test']:
        for rows in units(b,subset):
            inp = b.dispro.inputs(rows)
            assert set(inp)=={'x_WSI','x_Omics'}
            gid = rows[0]['global_patient_id']
            assert (inp['x_WSI'] is not None)==b.dispro.observed(gid,'WSI')
            assert (inp['x_Omics'] is not None)==b.dispro.observed(gid,'Omics')
            if not b.dispro.observed(gid,'WSI'):
                with pytest.raises(ValueError,match='missing WSI'):b.load(rows[0])


def test_ordered_genes_and_complete_multislide_bags_not_coordinate_or_label_proxies(tmp_path):
    b = Bundle(fixture(tmp_path/'data',normalization='none'))
    rows = next(r for r in units(b,'train') if len(r)==2 and b.dispro.observed(r[0]['global_patient_id'],'Omics'))
    inp = b.dispro.inputs(rows)
    expected = torch.cat([b.load(r)[0] for r in rows])[None]
    assert torch.equal(inp['x_WSI'],expected)
    gid = rows[0]['global_patient_id']
    raw = b.dispro.values.loc[gid]
    assert torch.equal(inp['x_Omics']['x_omic1'],torch.tensor([[raw.gene_D,raw.gene_C]],dtype=torch.float32))
    assert torch.equal(inp['x_Omics']['x_omic2'],torch.tensor([[raw.gene_F,raw.gene_B,raw.gene_A]],dtype=torch.float32))
    with pytest.raises(ValueError,match='all registered slides'):b.dispro.inputs(rows[:1])
    altered = [{**r,'event':1-r['event'],'time':999999} for r in rows]
    other = b.dispro.inputs(altered)
    assert torch.equal(other['x_WSI'],inp['x_WSI'])
    assert all(torch.equal(other['x_Omics'][k],inp['x_Omics'][k]) for k in inp['x_Omics'])


def test_TRAIN_observed_RNA_stats_ignore_heldout_values_and_missing_patients(tmp_path):
    root = fixture(tmp_path/'data');b = Bundle(root)
    train = set(b.rows.loc[b.rows.subset=='train','global_patient_id'])
    visible = [g for g in train if b.dispro.observed(g,'Omics')]
    ref = b.dispro.values.loc[sorted(visible)].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(b.dispro.mean,ref.mean(0))
    np.testing.assert_array_equal(b.dispro.scale,ref.std(0))
    mean,scale = b.dispro.mean.copy(),b.dispro.scale.copy()
    frame = pd.read_csv(root/'expression.csv')
    frame.loc[~frame.global_patient_id.isin(train),b.dispro.columns] += 10**6
    frame.to_csv(root/'expression.csv',index=False)
    after = Bundle(root)
    np.testing.assert_array_equal(mean,after.dispro.mean);np.testing.assert_array_equal(scale,after.dispro.scale)
    assert after.fingerprint!=b.fingerprint and after.dispro.contract!=b.dispro.contract


def test_hidden_physical_WSI_is_not_opened_and_zero_valued_observed_assay_stays_present(tmp_path):
    root = fixture(tmp_path/'data');b = Bundle(root)
    missing = next(r for r in b.rows.to_dict('records') if not b.dispro.observed(r['global_patient_id'],'WSI'))
    index = pd.read_csv(root/'features.csv')
    index.loc[index.slide_id==missing['slide_id'],'path'] = 'absent_file_deliberately_not_created.h5'
    index.to_csv(root/'features.csv',index=False)
    expr = pd.read_csv(root/'expression.csv');expr[b.dispro.columns] = 0;expr.to_csv(root/'expression.csv',index=False)
    b = Bundle(root);b.check(deep=True)
    rows = next(rows for rows in units(b,missing['subset']) if rows[0]['global_patient_id']==missing['global_patient_id'])
    values = b.dispro.inputs(rows)
    assert values['x_WSI'] is None and values['x_Omics'] is not None
    assert all(torch.equal(v,torch.zeros_like(v)) for v in values['x_Omics'].values())


@pytest.mark.parametrize('kind',['both_missing','truth_flag','missing_patient','duplicate_patient','unknown_patient'])
def test_presence_rejects_ambiguous_or_unusable_missingness(tmp_path,kind):
    root = fixture(tmp_path/'data');p = root/'presence.csv';frame = pd.read_csv(p)
    if kind=='both_missing':frame.loc[0,['wsi_observed','omics_observed']]=0
    elif kind=='truth_flag':
        frame['wsi_observed']=frame['wsi_observed'].astype(object)
        frame.loc[0,'wsi_observed']='True'
    elif kind=='missing_patient':frame=frame.iloc[1:]
    elif kind=='duplicate_patient':frame=pd.concat([frame,frame.iloc[:1]])
    else:frame.loc[0,'global_patient_id']='outside_task'
    frame.to_csv(p,index=False)
    with pytest.raises(ValueError,match='DisPro'):Bundle(root)


@pytest.mark.parametrize('kind',['missing_gene','duplicate_gene','unknown_gene','duplicate_pathway','empty_pathway','nonfinite_observed','missing_assay','outcome_column'])
def test_assay_and_pathway_integrity_prevents_silent_gene_reordering_or_imputation(tmp_path,kind):
    root = fixture(tmp_path/'data');p = root/'pathways.json';groups = json.loads(p.read_text())
    if kind in {'missing_gene','nonfinite_observed','missing_assay'}:
        expr = pd.read_csv(root/'expression.csv')
        if kind=='missing_gene':expr=expr.drop(columns=['gene_A'])
        elif kind=='missing_assay':expr=expr.iloc[1:]
        else:expr.loc[0,'gene_A']=np.nan
        expr.to_csv(root/'expression.csv',index=False)
    elif kind=='duplicate_gene':groups[0]['features']*=2
    elif kind=='unknown_gene':groups[0]['features'][0]='not_in_assay'
    elif kind=='duplicate_pathway':groups[1]['name']=groups[0]['name']
    elif kind=='empty_pathway':groups[0]['features']=[]
    else:
        meta = json.loads((root/'dispro_inputs.json').read_text());meta['gene_columns'].append('event')
        (root/'dispro_inputs.json').write_text(json.dumps(meta))
    p.write_text(json.dumps(groups))
    with pytest.raises((ValueError,KeyError),match='DisPro|float'):Bundle(root)


def test_complete_input_methods_cannot_silently_consume_incomplete_DisPro_cases(tmp_path):
    b = Bundle(fixture(tmp_path/'data'))
    with pytest.raises(ValueError,match='incomplete-modality'):check_compatibility(inspect_method('ABMIL'),b)
    with pytest.raises(ValueError,match='spatial coordinates'):b.check(deep=True,require_coords=True)
