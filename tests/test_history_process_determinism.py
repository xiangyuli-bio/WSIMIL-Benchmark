"""Public training histories must retain identical bytes across hash-randomized processes."""
import json
import os
import subprocess
import sys

from milplus.demo import create_demo


def test_public_multiterm_history_bytes_are_stable_across_python_hash_seeds(tmp_path):
    root = create_demo(tmp_path/'data', patients=18, dimension=32)
    cfg = tmp_path/'config.json'
    cfg.write_text(json.dumps({'epochs':2, 'threads':2, 'model':{'k_sample':4}}))
    histories = []
    # CLAM exercises the same generic aggregation with actual bag and instance loss
    # terms. Different processes keep their real salted set iteration; do not
    # force PYTHONHASHSEED globally or sort the serialized files in the test.
    for salt in ['1','392']:
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = salt
        output = tmp_path/('run_'+salt)
        cmd = [sys.executable,'-m','milplus','run','--data',str(root),'--method','CLAM',
               '--output',str(output),'--config',str(cfg),'--device','cpu','--seed','2026']
        result = subprocess.run(cmd,env=env,capture_output=True,text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads((output/'status.json').read_text())['state']=='completed'
        histories.append((output/'history.json').read_bytes())
        records=json.loads(histories[-1])
        assert len(records)==2 and len(records[0]['train'])>1
        assert list(records[0]['train'])==sorted(records[0]['train'])
    assert histories[0] == histories[1]
