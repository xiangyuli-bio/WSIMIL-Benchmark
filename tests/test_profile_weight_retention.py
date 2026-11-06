"""Fresh synthetic validation may retain receipts after real selected-weight evaluation."""

import pytest

from milplus.demo import create_demo
from milplus.engine import evaluate, train
from milplus.io import digest_file, read_json, write_json
from scripts.validate_all_profiles import retain_metadata


def test_retention_after_actual_reload_and_prediction(tmp_path):
    data=create_demo(tmp_path/'data',patients=20)
    run=train(data,'abmil',tmp_path/'fresh'/'abmil',config={'epochs':1,'threads':1,'model':{'hidden':8}})
    with pytest.raises(ValueError,match='evaluation'):retain_metadata(tmp_path/'fresh',run)
    evaluate(run)
    weights={str(p.relative_to(run)):(digest_file(p),p.stat().st_size) for p in run.rglob('*.pt')}
    predictions=(run/'test_predictions.csv').read_bytes()
    receipt=retain_metadata(tmp_path/'fresh',run)
    report=read_json(run/'weight_receipts.json')
    assert report['evaluation_completed_before_pruning'] is True
    assert {w['path']:(w['sha256'],w['bytes']) for w in report['weights']}==weights
    assert receipt['pruned_weight_bytes']==sum(size for _,size in weights.values())
    assert not list(run.rglob('*.pt')) and (run/'test_predictions.csv').read_bytes()==predictions


@pytest.mark.parametrize('kind',['natural','running','outside','symlink'])
def test_retention_refuses_non_owned_or_unverified_data(tmp_path,kind):
    root=tmp_path/'fresh';root.mkdir();run=root/'run';run.mkdir()
    write_json(run/'request.json',{'synthetic':kind!='natural'})
    write_json(run/'status.json',{'state':'running' if kind=='running' else 'completed'})
    (run/'test_predictions.csv').write_text('target,p_1\n0,.1\n')
    write_json(run/'test_metrics.json',{})
    weight=run/'best.pt';weight.write_bytes(b'not_a_real_checkpoint')
    if kind=='outside':root=root/'other'
    if kind=='symlink':
        external=tmp_path/'external.pt';external.write_bytes(b'user-owned')
        (run/'link.pt').symlink_to(external)
    with pytest.raises(ValueError):retain_metadata(root,run)
    assert weight.exists()
