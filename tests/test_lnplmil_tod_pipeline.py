"""Real RGB/LPA -> weak -> frozen features -> source SP -> full TOD checks."""
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import torch
from torch import nn

from milplus.engine import seed_everything
from milplus.io import digest_file, read_json
from milplus.hybrid_supervision_checkpoint import load
from milplus.lnplmil_raw import RawInputs, prepare
from milplus.lnplmil_weak import fit as fit_weak, score_raw
from milplus.lnplmil_source import model
from milplus.lnplmil_optim import create_optimizer
from milplus.lnplmil_tod import prepare_selection, SelectedInputs, fit, predict
from milplus.upstream import load_definitions
from test_lnplmil_weak import raw_fixture
from test_pmil_prototypical_components import equal


@pytest.fixture(scope='module')
def selected_native_inputs(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp('lnpl_complete_source_inputs')
    # The source SP stage is undefined for an all-tied/constant ROI. Keep that
    # failure as a separate rejection check below, and test the complete path
    # on401 varied actual RGB patches per bag (larger than native Top200).
    data, rows, schema = raw_fixture(root, patch_count=401, varied=True)
    raw = RawInputs(prepare(data, rows, schema, 'manifest.json'))
    weak = fit_weak(raw, root / 'weak', epochs=2, batch_size=256)
    scored = score_raw(raw, weak, root / 'scored', batch_size=64)
    selected = prepare_selection(scored, root / 'selected', 'released_source_with_reference_weak')
    return root, selected


def test_actual_raw_scoring_source_HNSW_and_full_TOD_literal_accumulation_exact(selected_native_inputs, tmp_path):
    torch.set_num_threads(2)
    _, manifest = selected_native_inputs
    selected = SelectedInputs(manifest)
    run = fit(manifest, tmp_path / 'run', epochs=1)
    state = load(run / 'last.pt')
    assert state['updates'] == 1 and state['batches'] == 2
    seed_everything(2026)
    literal = model()
    opt = create_optimizer(literal)
    order = torch.randperm(len(selected.records('train'))).tolist()
    interface = load_definitions('lnplmil', 'models/model_interface.py', ['ModelInterface'],
        {'torch': torch, 'nn': nn, 'np': __import__('numpy'),
         'pl': SimpleNamespace(LightningModule=nn.Module)}, replacements=[('.cuda()', '.to(data.device)')])
    owner = SimpleNamespace(model=literal, loss=nn.CrossEntropyLoss(), topk=200, ratio=.001,
                            data=[{'count': 0, 'correct': 0} for _ in range(2)])
    opt.zero_grad(set_to_none=True)
    for i in order:
        row = selected.records('train')[i]
        x = selected.load(row)[None]
        y = torch.tensor([row['target']])
        value = interface.ModelInterface.training_step(owner, (x, y), 0)['loss'] / 2
        value.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    equal(literal.state_dict(), state['model'])
    equal(opt.state_dict(), state['optimizer'])
    assert state['gradients'] == {name: None for name, _ in literal.named_parameters()}
    assert len(literal.C_Trans.layers) == 4 and literal.position_layer.pe.shape == (256, 200, 200)


@pytest.mark.parametrize('cut', [1, 2, 3])
def test_full_TOD_recovery_keeps_actual_partial_gradients_optimizer_slow_state_and_RNG(selected_native_inputs, tmp_path, cut):
    torch.set_num_threads(2)
    _, manifest = selected_native_inputs
    baseline = fit(manifest, tmp_path / 'baseline', epochs=2)
    interrupted = fit(manifest, tmp_path / 'resumed', epochs=2, stop_after_batches=cut)
    prior = load(interrupted / 'last.pt')
    if cut % 2:
        assert any(v is not None and torch.count_nonzero(v) for v in prior['gradients'].values())
    else:
        assert all(v is None for v in prior['gradients'].values())
    assert read_json(interrupted / 'status.json')['state'] == 'interrupted'
    fit(manifest, interrupted, epochs=2, resume=True)
    equal(load(baseline / 'last.pt'), load(interrupted / 'last.pt'))
    equal(load(baseline / 'selected.pt'), load(interrupted / 'selected.pt'))
    a, b = predict(baseline), predict(interrupted)
    equal(torch.tensor(a[['p_0', 'p_1']].values), torch.tensor(b[['p_0', 'p_1']].values))
    assert a.global_patient_id.tolist() == b.global_patient_id.tolist()
    assert read_json(interrupted / 'status.json')['test_evaluated']


def test_no_feature_only_bypass_or_changed_actual_scores_can_enter_full_TOD(tmp_path):
    path = tmp_path / 'features_only.json'
    path.write_text('{"kind":"frozen_patch_features_only"}')
    with pytest.raises(ValueError, match='actual weak'):
        prepare_selection(path, tmp_path / 'reject', 'released_source_with_reference_weak')
    assert not (tmp_path / 'reject').exists()


def test_TOD_complete_family_not_counted_before_full_method_CLI_wheel_and_native_acceptance():
    # Profiles describe executable candidates; their existence does not certify
    # full native/wheel/clinical acceptance. Preserve the original gate/node ID.
    from milplus.registry import profiles, methods
    candidates=[p for p in profiles() if p['family']=='lnplmil']
    assert len(candidates)==2
    assert all(p['formal_admission'] is False and p['status']=='adapted_under_verification'
               and p['real_wsi_validation']=='not_run' for p in candidates)
    family=next(r for r in methods() if r['id']=='lnplmil')
    assert family['formal_admission'] is False and family['paper_reproduction']=='not_run'


def test_original_source_empty_ROI_fails_explicitly_instead_of_inventing_fallback():
    from milplus.lnplmil_source import filter_superpatches
    from test_lnplmil_components import fixture
    value = fixture(401)
    value[:, 0] = .5
    with pytest.raises(ValueError, match='original source fails before its minimum-size fallback'):
        filter_superpatches(value)


def test_canonical_HNSW_preserves_original_search_parameters_and_cross_process_patch_identity(tmp_path):
    import subprocess, sys, os, json
    from milplus.lnplmil_source import canonical_superpatch_definitions
    from test_lnplmil_components import fixture
    value = fixture(401)
    value[:201, 3:] += 100; value[201:, 3:] -= 100
    value[:201, 0] = torch.linspace(.6, .99, 201)
    value[201:, 0] = torch.linspace(.01, .4, 200)
    search = canonical_superpatch_definitions().Hnsw(space='l2').fit(value[:, 3:].numpy())
    assert search.index_params_ == {'M':16, 'post':0, 'efConstruction':400, 'indexThreadQty':1}
    assert search.query_params_ == {'ef':90}
    path = tmp_path / 'registered_scored_input.pt'; torch.save(value, path)
    code = '''import sys,torch,hashlib,json
from milplus.lnplmil_source import filter_superpatches,paper_disjoint_superpatches,select_top200
torch.set_num_threads(2);x=torch.load(sys.argv[1],weights_only=True)
a=select_top200(filter_superpatches(x,canonical=True))[0]
b=select_top200(paper_disjoint_superpatches(x,canonical=True)[0])[0]
print(json.dumps([hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in [a,b]]))'''
    signatures=[]
    for _ in range(4):
        result=subprocess.run([sys.executable,'-s','-c',code,str(path)],text=True,capture_output=True,env=os.environ.copy())
        assert result.returncode==0,result.stderr
        signatures.append(json.loads(result.stdout))
    assert all(v==signatures[0] for v in signatures)


def test_actual_retained_patch_multiset_and_original_TopK_have_canonical_physical_score_ties():
    from collections import Counter
    from milplus.lnplmil_tod import canonical_candidate_order
    from milplus.lnplmil_source import select_top200
    from test_lnplmil_components import fixture
    value=fixture(401)
    value[:,0]=(torch.arange(len(value))%4).float()/5+.1
    other=value[torch.randperm(len(value),generator=torch.Generator().manual_seed(25))]
    a,b=canonical_candidate_order(value),canonical_candidate_order(other)
    equal(a,b)
    equal(select_top200(a)[0],select_top200(b)[0])
    assert Counter(r.numpy().tobytes() for r in a)==Counter(r.numpy().tobytes() for r in value)
    assert len(a)==len(value)==401 and a.shape[1]==515
