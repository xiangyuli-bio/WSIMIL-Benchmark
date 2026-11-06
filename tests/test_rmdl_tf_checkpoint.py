"""Optional isolated-runtime regression for actual RMDL checkpoint graph identity.

This checks checkpoint naming/creation and recovery, not native training budgets.
Full256x300x1024 GPU pipeline gates are separately required for admission.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('pooling',['sum','mean'])
def test_independent_process_checkpoint_graph_and_all_tensors_match(tmp_path,pooling):
    if not os.environ.get('MILPLUS_TF_PYTHON'):
        pytest.skip('Set MILPLUS_TF_PYTHON to the isolated original-graph TF2.15.1 runtime')
    from milplus.rmdl_driver import environment
    executable,env=environment({'threads':2},'cpu')
    helper=tmp_path/'actual_graph_runner.py'
    helper.write_text('''import sys,json,hashlib\nfrom pathlib import Path\nimport numpy as np,tensorflow as tf\nfrom milplus.rmdl_source import build_aggregator,normalize_features,objective\nfrom milplus.rmdl_tf import _checkpoint,_restore\nfrom milplus.io import read_json\nrole,pooling,folder=sys.argv[1:];out=Path(folder);out.mkdir(exist_ok=True)\ntf.keras.utils.set_random_seed(73);tf.config.experimental.enable_op_determinism()\nm,opt,dropout=build_aggregator(seed=73,pooling=pooling);start=0\nif role=='resume':\n state=read_json(out/'state.json');_restore(m,opt,state);start=state['updates']\nfor step in range(start,1 if role=='stop'else 2):\n rng=np.random.RandomState(73+step);x=rng.normal(size=(2,300,1024)).astype('float32');x=np.stack([normalize_features(v)for v in x])\n opt.learning_rate.assign(.001)\n with tf.GradientTape()as tape:loss,ce,l2,p=objective(m,x,np.asarray([0,2]),True)\n gradients=tape.gradient(loss,m.trainable_variables);opt.apply_gradients(zip(gradients,m.trainable_variables))\n _checkpoint(m,opt,out,{'updates':step+1})\nif role!='stop':\n reader=tf.train.load_checkpoint(read_json(out/'state.json')['checkpoint_prefix'])\n values={}\n for key in sorted(reader.get_variable_to_shape_map()):\n  value=np.asarray(reader.get_tensor(key));values[key]={'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}\n (out/'all_tensors_including_object_graph.json').write_text(json.dumps(values,sort_keys=True,indent=2))\n''')
    for role,folder in [('baseline','baseline'),('stop','recovered'),('resume','recovered')]:
        result=subprocess.run([executable,'-s',str(helper),role,pooling,str(tmp_path/folder)],env=env,
                              capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    left=(tmp_path/'baseline/all_tensors_including_object_graph.json').read_bytes()
    right=(tmp_path/'recovered/all_tensors_including_object_graph.json').read_bytes()
    assert left==right
