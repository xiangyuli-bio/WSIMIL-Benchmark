"""Execute original2018AGCN graph, derivatives and Adam in TF1 compatibility mode."""
import argparse
import ast
import hashlib
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import tensorflow

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',required=True)
parser.add_argument('--fixture',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
ROOT=Path(args.source).resolve()
OUTPUT=Path(args.output).resolve()
OUTPUT.mkdir(parents=True,exist_ok=False)
SHA='3149752deba68067f1a39b9f3d212cb16ffaf560'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==SHA
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
tf=tensorflow.compat.v1;tf.disable_v2_behavior()
tf.contrib=types.SimpleNamespace(layers=types.SimpleNamespace(xavier_initializer=tf.glorot_uniform_initializer))
files={}
def source(filename,names,scope):
 code=subprocess.check_output(['git','show',SHA+':'+filename],cwd=ROOT,text=True)
 files[filename]=hashlib.sha256(code.encode()).hexdigest();tree=ast.parse(code)
 tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in tree.body}==set(names)
 module=types.ModuleType('deepgraphsurv_original_probe_'+filename.replace('/','_').replace('.','_'))
 module.__dict__.update(scope);exec(compile(tree,str(ROOT/filename),'exec'),module.__dict__)  # noqa: S102 -- selected definitions from the verified original Git blob
 return module
layers=source('utils/layers.py',['_variable_on_cpu','_variable_with_weight_decay','SGC_LL',
 'gather','fully_connected','batch_norm_template','batch_norm_for_conv1d','batch_norm_for_fc'],{'tf':tf,'np':np})
model=source('models/networks/AGCN.py',['attention_net','basic_agcn_wsi_survival2','survival_loss'],{'tf':tf,'np':np,'layers':layers})
report={'source_commit':SHA,'source_file_sha256':files,'tensorflow':tensorflow.__version__,
 'synthetic':True,'device':'cpu','input_shape':[2,12,128],
 'scope':'Original native256/256/128 adaptive graph CNN and32/1attention, literal tanh risk/Cox, all gradients and one TF1Adam step probe. Small synthetic shapes only; no integrated adapter/whole trainer/realWSI/paper-score claim.',
 'optimizer_comparison':'Actual TF1Adam apply_gradients fed the captured unchanged original loss derivatives; identical values used in the PyTorch optimizer proof, no concurrent graph mutation',
 'compatibility':['Actual tensorflow.compat.v1 graph/session with v2 behavior disabled','Removed tf.contrib Xavier import mapped to compat.v1.glorot_uniform_initializer; unchanged original function bodies']}
try:
 fixture=np.load(args.fixture);data=fixture['x'];sizes=fixture['sizes'];lap=fixture['laplacian'];dimension=int(data.shape[2]);report['input_shape']=list(data.shape)
 with tf.Graph().as_default():
  tf.set_random_seed(2345)
  X=tf.placeholder(tf.float32,[2,12,dimension]);L=tf.placeholder(tf.float32,[2,12,12]);N=tf.placeholder(tf.int32,[2])
  training=tf.placeholder(tf.bool,[]);events=tf.placeholder(tf.float32,[2])
  native=model.basic_agcn_wsi_survival2(X,L,N,12,1,training,use_attn=True,bn_decay=tf.constant(.5))
  loss=model.survival_loss(native['output'],events)
  variables=tf.trainable_variables();grads=tf.gradients(loss,variables)
  opt=tf.train.AdamOptimizer(.0002)
  gradient_inputs=[tf.placeholder(tf.float32,v.shape,name='controlled_gradient_'+str(i)) for i,v in enumerate(variables)]
  step=opt.apply_gradients(list(zip(gradient_inputs,variables)))
  config=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
  feed={X:data,L:lap,N:sizes,training:True,events:np.array([0,1],np.float32)}
  with tf.Session(config=config) as sess:
   sess.run(tf.global_variables_initializer())
   masks=[op.outputs[0] for op in tf.get_default_graph().get_operations() if op.type=='GreaterEqual' and 'dropout' in op.name]
   assert len(masks)==3,[(o.name,o.type) for o in tf.get_default_graph().get_operations() if 'dropout' in o.name]
   input_gradient=tf.gradients(loss,X)[0]
   before=sess.run(tf.global_variables());variable_names=[v.op.name for v in tf.global_variables()]
   # Fetch the complete derivative graph without concurrently mutating variables.
   # opt.minimize builds its own backward graph: fetching its apply op together
   # with a second tf.gradients graph can race variable reads against updates.
   value,pred,grad,dx,mask_values=sess.run([loss,native['output'],grads,input_gradient,masks],feed_dict=feed)
   forward_state=sess.run(tf.global_variables())
   assert np.isfinite(value) and np.isfinite(pred).all() and all(np.isfinite(g).all() for g in grad)
   # Isolate actual TF1Adam from separate backward graphs and FP32 differences:
   # both frameworks apply the exact same original loss gradient arrays.
   sess.run(step,feed_dict=dict(zip(gradient_inputs,grad)))
   after=sess.run(tf.global_variables());assert any(not np.array_equal(a,b) for a,b in zip(before,after))
   arrays={'input':data,'intrinsic_laplacian':lap,'sizes':sizes,'events':feed[events],
    'risk':pred,'loss':np.array(value),'input_gradient':dx}
   for name,v in zip(variable_names,before):arrays['before::'+name]=v
   for name,v in zip(variable_names,after):arrays['after::'+name]=v
   for name,v in zip(variable_names,forward_state):arrays['after_forward::'+name]=v
   for v,g in zip(variables,grad):arrays['gradient::'+v.op.name]=g
   for tensor,mask_value in zip(masks,mask_values):arrays['mask::'+tensor.op.name]=mask_value
   destination=OUTPUT/'reference.npz';np.savez(destination,**arrays)
   report.update(reference_arrays=str(destination),variables=variable_names,trainables=[v.op.name for v in variables],
    dropout_masks=[v.op.name for v in masks],loss=float(value),all_gradients_finite=True)
   feed[training]=False
   p1=sess.run(native['output'],feed_dict=feed);p2=sess.run(native['output'],feed_dict=feed)
   saved=[v.op.name for v in native['saved_vars']]
   report.update(state='original_TF1_native_network_all_gradients_and_Adam_step_passed',
    loss=float(value),risk=pred.tolist(),trainable_parameters=sum(int(np.prod(v.shape.as_list())) for v in variables),
    trainable_tensors=len(variables),finite_gradient_tensors=len(grad),
    stochastic_eval=not np.array_equal(p1,p2),original_saved_variables=saved,
    omitted_trainables=[v.op.name for v in variables if v.op.name not in saved])
except Exception as e:  # noqa: BLE001 -- persist explicit failed comparison before nonzero exit
 report.update(state='source_probe_failed',error_type=type(e).__name__,error=str(e))
p=OUTPUT/'reference.json';p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'state':report['state'],'report':str(p),'error':report.get('error')}))

if report['state'] != 'original_TF1_native_network_all_gradients_and_Adam_step_passed':
 raise SystemExit(1)
