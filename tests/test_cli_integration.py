import json
import subprocess
import sys

import pandas as pd
import pytest

from milplus.demo import create_demo
from milplus.execution import hcluster_script, matrix, run_matrix
from milplus.io import read_json
from milplus.preparation import index_features, prepare_tcga


def cli(*args):
    return subprocess.run([sys.executable,'-m','milplus',*map(str,args)],capture_output=True,text=True,check=True)


def test_cli_end_to_end(tmp_path):
    task=tmp_path/'task'; run=tmp_path/'run'
    cli('demo','--output',task,'--patients','20')
    cli('check','--data',task,'--method','abmil','--deep')
    cli('run','--data',task,'--method','mean_linear','--output',run)
    result=json.loads(cli('evaluate','--run',run).stdout)
    assert result['macro_auc'] >= 0
    cli('report','--runs',tmp_path,'--output',tmp_path/'report.csv')
    assert len(pd.read_csv(tmp_path/'report.csv')) == 1


def test_matrix_keeps_blocked_and_runs(tmp_path):
    task=create_demo(tmp_path/'task',patients=20)
    plan=tmp_path/'plan.json'
    rows=matrix(task,['mean_linear','2DMambaMIL'],['toy'],[2026],plan)
    assert [r['state'] for r in rows] == ['ready','blocked']
    result=run_matrix(plan,tmp_path/'batch',['cpu'],epochs=1)
    assert result['completed'] == 1 and result['total'] == 2
    result=run_matrix(plan,tmp_path/'batch',['cpu'],epochs=1)
    assert result['completed'] == 0


def test_indexing_and_duplicate_protection(tmp_path):
    task=create_demo(tmp_path/'task',patients=20)
    result=index_features(task,task/'features','second',16,'synthetic-weights','synthetic test')
    assert result['indexed'] == 30
    with pytest.raises(ValueError,match='already registered'):
        index_features(task,task/'features','second',16,'weights','test')


def test_cluster_command_is_safe_and_explicit():
    script,cmd=hcluster_script('/mnt/lixiangyu/repo','/mnt/lixiangyu/task','abmil','/mnt/lixiangyu/results',
                               'registry/image:tag','ailab-medpeak','medai4s_gpu_pool')
    assert '--memory=32768' in cmd
    assert not any('private-machine' in s for s in cmd)
    assert 'sleep' not in script
    assert len([c for c in cmd if c.startswith('--mount=')]) == 4


def test_tcga_preserves_source_patient_split(tmp_path):
    rows=[]; paths={}
    for j,subset in enumerate(['train','val','test']):
        ids=[]
        for label in ['IDC','ILC']:
            patient=f'TCGA-{j}-{label}'
            ids.append(patient)
            rows.append({'project_id':'TCGA-BRCA','slide_type_code':'DX','OncoTreeCode':label,
                         'case_id':patient,'slide_id':patient+'-DX1','wsi_path':'/raw/'+patient+'.svs'})
        paths[subset]=tmp_path/(subset+'.csv')
        pd.DataFrame({'case_id':ids}).to_csv(paths[subset],index=False)
    inventory=tmp_path/'inventory.csv'; pd.DataFrame(rows).to_csv(inventory,index=False)
    prepare_tcga(inventory,paths,'TCGA-BRCA',['IDC','ILC'],tmp_path/'prepared')
    report=read_json(tmp_path/'prepared'/'preparation.json')
    assert report['patients'] == 6
    assert len(report['source_splits']) == 3


def test_mambamil_family_resolves_ready_with_all_variants(tmp_path):
    task=create_demo(tmp_path/'task',patients=20)
    rows=matrix(task,['MambaMIL','mambamil_bimamba','mambamil_unidirectional'],['toy'],[2026],tmp_path/'plan.json')
    assert len(rows)==3 and all(row['state']=='ready' for row in rows)
    assert rows[0]['method']=='mambamil_srmamba'
    cli('check','--data',task,'--method','MambaMIL','--deep')
    config=tmp_path/'config.json';config.write_text(json.dumps({'epochs':1,'threads':1,'model':{'hidden':16,'scan_chunk':16}}))
    cli('run','--data',task,'--method','MambaMIL','--config',config,'--output',tmp_path/'run')
    result=json.loads(cli('evaluate','--run',tmp_path/'run').stdout)
    assert result['n']>0
    assert read_json(tmp_path/'run/request.json')['method']=='mambamil_srmamba'


def test_matrix_never_marks_unresolved_family_ready(tmp_path,monkeypatch):
    from milplus import execution
    task=create_demo(tmp_path/'task',patients=20)
    monkeypatch.setattr(execution,'inspect_method',lambda method:{'id':'family','status':'adapted','runnable':False,'blocker':None})
    rows=matrix(task,['family'],['toy'],[2026],tmp_path/'plan.json')
    assert rows[0]['state']=='blocked' and 'explicit runnable' in rows[0]['reason']
