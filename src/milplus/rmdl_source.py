"""Private original Keras RMDL graphs and explicit modern-runtime bridges.

No reduced encoder or generic attention aggregator replaces either full graph.
The unpublished localization trainer is a separate named reference protocol.
"""
import ast
from functools import lru_cache
from pathlib import Path
import textwrap
import types

import numpy as np

from .io import digest_file
from .upstream import verified_source, pinned_text, load_definitions

COMMIT='8f13238285b92e772e04d2abe25cc24bb8099829'
CONTRIB_COMMIT='3fc5ef709e061416f4bc8a92ca3750c824b5d2b0'
IMAGENET_MD5='d19885ff4a710c122648d3b5c3b684e4'
IMAGENET_SHA256='937fb087127afa1d6d09f5eed7e560fc9ef20d3c6583b54653141c922a6dee0f'
KERAS221_COMMIT='eb2c1fdc2d5f841e328993e3bd0e1a3de268fe20'


def checked_source():
    root,commit=verified_source('rmdl')
    if commit!=COMMIT:raise ValueError('Use the pinned actual RMDL source')
    return root


def _tf():
    import tensorflow as tf
    if tf.__version__!='2.15.1':raise ValueError('Full original RMDL graph port requires isolated TensorFlow2.15.1')
    return tf


def legacy_adam(**kwargs):
    """Literal Keras2.2.1 dense Adam arithmetic inside a checkpointable optimizer.

    TF2.15's fused ResourceApplyAdam changes the rounding of m/v updates. The
    original expression order is retained here rather than relaxing equality.
    Original non-AMSGrad unused zero-vhat placeholders have no update equation.
    """
    tf=_tf()
    class OriginalDenseAdam221(tf.keras.optimizers.legacy.Adam):
        def _resource_apply_dense(self,grad,var,apply_state=None):
            dtype=var.dtype.base_dtype
            one=tf.cast(1.,dtype)
            b1=self._get_hyper('beta_1',dtype);b2=self._get_hyper('beta_2',dtype)
            lr=self._get_hyper('learning_rate',dtype)
            if self._initial_decay>0:
                decay=self._get_hyper('decay',dtype)
                lr=lr*(one/(one+decay*tf.cast(self.iterations,dtype)))
            t=tf.cast(self.iterations,dtype)+one
            rate=lr*(tf.sqrt(one-tf.pow(b2,t))/(one-tf.pow(b1,t)))
            m=self.get_slot(var,'m');v=self.get_slot(var,'v')
            next_m=b1*m+(one-b1)*grad
            next_v=b2*v+(one-b2)*tf.square(grad)
            next_var=var-rate*next_m/(tf.sqrt(next_v)+tf.cast(self.epsilon,dtype))
            return tf.group(m.assign(next_m),v.assign(next_v),var.assign(next_var))
        def _resource_apply_sparse(self,*args,**kwargs):
            raise ValueError('Original full RMDL dense graphs do not define a sparse Adam fallback')
    if kwargs.get('amsgrad',False):
        raise ValueError('Original RMDL uses non-AMSGrad Adam')
    return OriginalDenseAdam221(**kwargs)


@lru_cache(maxsize=1)
def instance_normalization():
    tf=_tf();root,commit=verified_source('keras_contrib_legacy')
    if commit!=CONTRIB_COMMIT:raise ValueError('Use the pinned full keras-contrib InstanceNormalization')
    return load_definitions('keras_contrib_legacy',
        'keras_contrib/layers/normalization/instancenormalization.py',['InstanceNormalization'],
        {'Layer':tf.keras.layers.Layer,'InputSpec':tf.keras.layers.InputSpec,'K':tf.keras.backend,
         'initializers':tf.keras.initializers,'regularizers':tf.keras.regularizers,
         'constraints':tf.keras.constraints}).InstanceNormalization


def _shape(shape,default_size,min_size,data_format,require_flatten,weights=None):
    # Validation-only port of the removed keras_applications helper.
    if data_format!='channels_last':raise ValueError('RMDL original RGB uses channels_last')
    shape=(default_size,default_size,3)if shape is None else tuple(shape)
    if len(shape)!=3 or shape[-1]!=3 or any(n is not None and n<min_size for n in shape[:2]):
        raise ValueError('RMDL requires the complete original spatial RGB dimensions')
    if require_flatten and any(n is None for n in shape):raise ValueError('Top classifier requires fixed spatial size')
    return shape


def build_localizer(weights):
    """Actual full author 10/20/10-block FCN, with verified ImageNet initialization."""
    import hashlib
    tf=_tf();root=checked_source();weights=Path(weights)
    h=hashlib.md5()
    with weights.open('rb')as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    if h.hexdigest()!=IMAGENET_MD5 or digest_file(weights)!=IMAGENET_SHA256:
        raise ValueError('Complete official Keras ImageNet InceptionResNetV2 weights required')
    ns={'warnings':__import__('warnings'),'np':np,'K':tf.keras.backend,'Model':tf.keras.Model,
        '_obtain_input_shape':_shape,'get_source_inputs':tf.keras.utils.get_source_inputs,'roi_size':1067}
    for name in ['Activation','AveragePooling2D','BatchNormalization','Concatenate','Conv2D','Dense',
                 'GlobalAveragePooling2D','GlobalMaxPooling2D','Input','Lambda','MaxPooling2D']:
        ns[name]=getattr(tf.keras.layers,name)
    original=load_definitions('rmdl','weights/inception_resnet_v2.py',
        ['conv2d_bn','inception_resnet_block','BuildModel'],ns)
    model=original.BuildModel(include_top=False,weights=None,input_shape=(None,None,3),classes=3,isextract=False)
    # The released BuildModel's weights argument never loads ImageNet. Load the
    # actual official full same-graph no-top weights, then copy every backbone
    # tensor by weighted-layer topology and verify every value, not just shape.
    with tf.device('/CPU:0'):
        imagenet=tf.keras.applications.InceptionResNetV2(include_top=False,weights=None,input_shape=(None,None,3))
        imagenet.load_weights(str(weights))
    expected=[layer for layer in imagenet.layers if layer.weights]
    actual=[layer for layer in model.layers if layer.weights and layer.name not in {'conv7','cls'}]
    if len(actual)!=len(expected):raise ValueError('Full author/ImageNet backbone topology differs')
    records=[]
    for dst,src in zip(actual,expected):
        values=src.get_weights()
        if [tuple(v.shape)for v in dst.weights]!=[v.shape for v in values]:
            raise ValueError('Full ImageNet backbone layer mismatch: '+dst.name+'/'+src.name)
        dst.set_weights(values)
        if any(not np.array_equal(a,b)for a,b in zip(dst.get_weights(),values)):
            raise ValueError('Actual pretrained backbone values changed: '+dst.name)
        records.append({'author_layer':dst.name,'official_layer':src.name,'shapes':[list(v.shape)for v in values]})
    encoder=tf.keras.Model(model.inputs,model.get_layer('conv7').output,name='rmdl_actual_pre_relu_FConv1024')
    return model,encoder,{'ImageNet_file_sha256':digest_file(weights),'full_weighted_backbone_layers':records,
        'author_source_commit':COMMIT,'full_original_graph':True,'native_FConv':1024,'native_CConv':3,
        'feature_is_before_ReLU':True,'random_encoder_fallback':False}


def _dropout_factory(seed):
    tf=_tf();created=[]
    class ReplayableDropout(tf.keras.layers.Layer):
        def __init__(self,rate,**kwargs):
            index=len(created);kwargs.setdefault('name','dropout_'+str(index+1))
            super().__init__(**kwargs);self.rate=float(rate);self.index=index
            self.counter=self.add_weight(name='replay_counter',shape=(),dtype=tf.int64,initializer='zeros',trainable=False)
            self.base_seed=np.asarray([int(seed)%(2**31-1),index+1],dtype=np.int32);created.append(self)
        def call(self,x,training=None):
            if training is None:training=tf.keras.backend.learning_phase()
            def dropped():
                key=tf.random.experimental.stateless_fold_in(tf.constant(self.base_seed),self.counter)
                value=tf.nn.experimental.stateless_dropout(x,rate=self.rate,seed=key)
                self.counter.assign_add(1)
                return value
            return tf.__internal__.smart_cond.smart_cond(training,dropped,lambda:tf.identity(x))
        def get_config(self):return {**super().get_config(),'rate':self.rate}
    return ReplayableDropout,created


def build_aggregator(classes=3,pooling='sum',seed=2026):
    """Execute the actual get_chowder body, original ADD and normalization class."""
    tf=_tf();root=checked_source()
    if type(classes)is not int or classes<2 or pooling not in {'sum','mean'}:
        raise ValueError('Declare the native sum or explicitly named paper-average reference')
    pooling_defs=load_definitions('rmdl','2detector/Code/mil_nets/pooling_method.py',
        ['max_pooling','mean_pooling','LSE_pooling','softmax','relu_activation','sigmoid_activation','add_pooling','choice_pooling'],
        {'K':tf.keras.backend})
    layer_defs=load_definitions('rmdl','2detector/Code/mil_nets/layer.py',['ADD','Expand_dims'],
        {'Layer':tf.keras.layers.Layer,'K':tf.keras.backend,'pooling':pooling_defs})
    original=pinned_text(root,COMMIT,'2detector/Code/RMDL.py')
    begin=original.index('    def get_chowder(self):');end=original.index('    def predict(self):',begin)
    tree=ast.parse(textwrap.dedent(original[begin:end]))
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):raise ValueError('Original complete RMDL graph body changed')
    dropout,dropouts=_dropout_factory(seed)
    ns={'local_feature_size':[300,1024],'InstanceNormalization':instance_normalization(),
        'Expand_dims':layer_defs.Expand_dims,'ADD':lambda **kwargs:layer_defs.ADD(pooling_mode='add'if pooling=='sum'else'ave',**kwargs),
        'l2':tf.keras.regularizers.l2,'optimizers':types.SimpleNamespace(Adam=legacy_adam),
        'Model':tf.keras.Model,'Dropout':dropout}
    for name in ['Input','Dense','Concatenate','MaxPooling1D','LeakyReLU','Softmax','Multiply','Flatten']:
        ns[name]=getattr(tf.keras.layers,name)
    exec(compile(tree,str(root/'2detector/Code/RMDL.py')+':original_get_chowder','exec'),ns)
    holder=types.SimpleNamespace(local_feature_size=[300,1024],n_class=classes,lr=1e-3)
    ns['get_chowder'](holder)
    model=holder.model
    if model.input_shape!=(None,300,1024)or model.output_shape!=(None,classes):
        raise ValueError('Original complete RMDL widths/instance count changed')
    return model,model.optimizer,dropouts


def normalize_features(x):
    x=np.asarray(x,dtype=np.float32)
    if x.shape!=(300,1024)or not np.isfinite(x).all():raise ValueError('RMDL requires300 actual selected1024D features')
    # Literal generate_testdata per-bag all-element min/max rule.
    return np.float32(2.)*(x-x.min())/(x.max()-x.min()+np.float32(1e-8))-np.float32(1.)


def objective(model,inputs,targets,training=True):
    """Keras2.2.1 probability-clipped CE plus actual model kernel regularizers."""
    tf=_tf();probabilities=model(inputs,training=training)
    p=probabilities/tf.reduce_sum(probabilities,axis=-1,keepdims=True)
    p=tf.clip_by_value(p,tf.cast(1e-7,p.dtype),tf.cast(1.-1e-7,p.dtype))
    labels=tf.one_hot(targets,probabilities.shape[-1],dtype=probabilities.dtype)
    ce=-tf.reduce_mean(tf.reduce_sum(labels*tf.math.log(p),axis=-1))
    regularization=tf.add_n(model.losses)if model.losses else tf.zeros((),dtype=ce.dtype)
    return ce+regularization,ce,regularization,probabilities


def source_review():
    root=checked_source()
    return {'source_commit':COMMIT,'files':{name:digest_file(root/name)for name in [
        'weights/inception_resnet_v2.py','weights/inception_resnet_v2_test.py','2detector/Code/RMDL.py',
        '2detector/Code/load_data.py','2detector/Code/mil_nets/layer.py','2detector/Code/mil_nets/pooling_method.py']},
        'runtime_port':'full original graphs privately executed in TensorFlow2.15.1/Keras2.15; original TF1.10 runtime not claimed',
        'dropout_port':'same rates/shapes with explicit checkpointed stateless per-layer counters for process recovery',
        'optimizer':'literal Keras2.2.1 non-fused dense Adam expression order, inverse-time decay0.001; fused TF2 legacy Adam is not substituted',
        'optimizer_equation_source_commit':KERAS221_COMMIT,
        'loss':'literal Keras2.2.1 clipped probability CE plus original Dense-kernelL2_0.005',
        'normalization':'actual original keras-contrib InstanceNormalization std+0.001, not LayerNorm',
        'feature_scaling':'literal per-bag source minmax to[-1,1]',
        'native_localizer':'complete author InceptionResNetV2/AvgPool8/FConv1024/ReLU/CConv3',
        'native_selection':'paper FCN1899,stride1632,NMS0.68,100perlocalizerclass,300total',
        'native_aggregation':'original Dense128/64/64,globalmax1/2,recalibration;sum and namedmean separately',
        'native_aggregation_budget':'7000iterations,batch256,Adam1e-3,LR*.8every140',
        'missing_original_trainers':'localizer supervision/sampling/hard-negative trainer and MIL train loop not released',
        'complete_method_counted':False,'formal_admission':False}
