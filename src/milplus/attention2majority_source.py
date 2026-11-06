"""Pinned complete Attention2majority TensorFlow components, no Torch substitute.

TensorFlow is imported only in the isolated method environment. The binary
graph and selectors execute the actual author definitions. The three-head
graph is reconstructed from the full commented author class and named as such.
"""
import ast
import types

from .upstream import verified_source,pinned_text,load_definitions
from .io import digest_file

COMMIT='17c9614e2192ddf899a7a449081256bc9542b5d0'
RESNET_MD5='4d473c1dd8becc155b73f8504c6f6626'


def checked_source():
    root,sha=verified_source('attention2majority')
    if sha!=COMMIT:raise ValueError('Attention2majority author commit changed')
    return root,sha


def binary_classes():
    import tensorflow as tf
    checked_source()
    return load_definitions('attention2majority','milmodels.py',['gatedattention','AttMILbinary'],
        {'tf':tf,'layers':tf.keras.layers,'models':tf.keras.models})


def commented_multiclass_class():
    import tensorflow as tf
    root,sha=checked_source();original=pinned_text(root,sha,'milmodels.py')
    start=original.index('# class AttMIL2(models.Model):');end=original.index('class gatedattention')
    lines=[]
    for line in original[start:end].splitlines():
        i=line.find('#')
        if i>=0:
            rest=line[i+1:];rest=rest[1:] if rest.startswith(' ') else rest
            lines.append(line[:i]+rest)
        else:lines.append(line)
    text='\n'.join(lines);tree=ast.parse(text)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.ClassDef) or tree.body[0].name!='AttMIL2':
        raise ValueError('Full commented author three-head graph is missing')
    namespace={'tf':tf,'layers':tf.keras.layers,'models':tf.keras.models}
    exec(compile(tree,str(root/'milmodels.py')+':commented_AttMIL2','exec'),namespace)
    return namespace['AttMIL2']


def binary_selection(features,scores,ratio=.3,zero_tail_repaired=False):
    """Execute the actual author positive/negative-tail selector, keeping -0 explicit."""
    import numpy as np
    checked_source()
    x=np.asarray(features);p=np.asarray(scores)
    if (x.ndim!=2 or x.shape[1]!=1024 or not len(x) or p.shape!=(len(x),1)
            or not np.isfinite(x).all() or not np.isfinite(p).all() or (p<0).any() or (p>1).any()
            or type(ratio) not in {float,int} or not 0<ratio<=1):
        raise ValueError('Attention2majority selection needs complete native1024D aligned actual discriminator probabilities')
    # Extra identity column is not passed through the network: it allows the
    # literal selector to expose original indices without rewriting its order.
    identity=np.concatenate([x,np.arange(len(x),dtype=x.dtype)[:,None]],axis=1)
    replacements=[]
    if zero_tail_repaired:
        replacements=[('embedding0 = embedding0[sortidxs0[-round(K*scores0.shape[0]):]]',
            'embedding0 = embedding0[sortidxs0[-round(K*scores0.shape[0]):]] if round(K*scores0.shape[0]) else embedding0[:0]')]
    source=load_definitions('attention2majority','intelsampling.py',['createbags_final'],
        {'np':np},replacements)
    bags,labels=source.createbags_final(['slide'],{'slide':p},{'slide':identity},{'slide':0},ratio,1025)
    indices=bags[0][:,-1].astype(np.int64)
    if not len(indices) or len(indices)!=len(np.unique(indices)):
        raise ValueError('Attention2majority selected an empty/nonunique real bag; no duplicated or padded patches')
    return x[indices],indices


def majority_selection(features,probabilities,k=2000):
    """Paper/source majority ordering: class frequency, within-class confidence, topK."""
    import numpy as np
    x=np.asarray(features);p=np.asarray(probabilities)
    if (x.ndim!=2 or x.shape[1]!=1024 or not len(x) or p.shape!=(len(x),3)
            or type(k) is not int or k<1 or not np.isfinite(x).all() or not np.isfinite(p).all()
            or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1.,rtol=1e-6,atol=1e-7)):
        raise ValueError('Attention2majority majority queue needs aligned native1024D features and three-class probabilities')
    predictions=p.argmax(1);confidence=p.max(1)
    # Preserve released NumPy argsort tie behavior; do not silently introduce
    # a new stable tie order into the original majority queue.
    groups=np.argsort(-np.bincount(predictions,minlength=3));rank=np.argsort(-confidence)
    ranked_predictions=predictions[rank]
    indices=np.concatenate([rank[np.where(ranked_predictions==c)[0]] for c in groups])[:k]
    return x[indices],indices


def preprocess(pixels,mode):
    import numpy as np
    import tensorflow as tf
    x=np.asarray(pixels)
    if x.ndim!=4 or x.shape[-1]!=3 or x.dtype!=np.uint8 or not len(x):
        raise ValueError('Attention2majority needs original nonempty uint8 NHWC RGB patches')
    if mode not in {'source_signed_int8_caffe_then_rescale','paper_uint8_caffe'}:
        raise ValueError('Name the source signed-int8 or paper-reference uint8 preprocessing condition')
    # The author tf.data generator declares int8, then its load_discimage
    # casts to float and performs exact Keras Caffe preprocessing. A uint8
    # input therefore wraps >=128 before casting. Keep this difference explicit.
    value=x.astype(np.int8).astype(np.float32) if mode.startswith('source_') else x.astype(np.float32)
    return tf.keras.applications.resnet.preprocess_input(tf.convert_to_tensor(value))


def build_discriminator(weights,patch_pixels=224,classes=2,preprocessing='source_signed_int8_caffe_then_rescale'):
    import hashlib
    import tensorflow as tf
    if tf.__version__!='2.15.1' or patch_pixels not in {128,224} or classes not in {2,3}:
        raise ValueError('Attention2majority reference requires exact TensorFlow2.15.1 and native RGB128/224, binary/three-class outputs')
    if hashlib.md5(open(weights,'rb').read()).hexdigest()!=RESNET_MD5:
        raise ValueError('Full original Keras ImageNet ResNet50 no-top weights are required')
    base=tf.keras.applications.ResNet50(include_top=False,weights=str(weights),input_shape=(patch_pixels,patch_pixels,3))
    encoder=tf.keras.Model(base.input,base.get_layer('conv4_block6_out').output,name='resnet_block3')
    layers=[]
    if preprocessing=='source_signed_int8_caffe_then_rescale':layers.append(tf.keras.layers.Rescaling(1./255,input_shape=(patch_pixels,patch_pixels,3),name='enrescale'))
    elif preprocessing!='paper_uint8_caffe':raise ValueError('Explicit source/paper normalization condition is required')
    layers += [encoder,tf.keras.layers.GlobalAveragePooling2D(name='globalpooling'),
        tf.keras.layers.Dropout(.25,name='endrop'),
        tf.keras.layers.Dense(1 if classes==2 else 3,activation='sigmoid' if classes==2 else 'softmax',name='enhead')]
    model=tf.keras.Sequential(layers,name='discriminator')
    model(tf.zeros([1,patch_pixels,patch_pixels,3]),training=False)
    features=tf.keras.Model(model.input,model.get_layer('globalpooling').output,name='task_trained_1024_features')
    if features.output_shape!=(None,1024) or not encoder.trainable or model.get_layer('endrop').rate!=.25:
        raise ValueError('Complete source truncated ResNet50/GAP1024/dropout/.25 classifier changed')
    return model,features


def build_aggregator(classes):
    import tensorflow as tf
    if classes==2:
        model=binary_classes().AttMILbinary();model.build(inputshape=(None,None,1024))
    elif classes==3:
        model=commented_multiclass_class()();model.build((None,None,1024),3)
    else:raise ValueError('Original Attention2majority supports binary and the source three-head three-class graph')
    model(tf.zeros([1,7,1024]),training=False)
    return model


def objective(model,features,target,class_weights,classes,training=True):
    import tensorflow as tf
    output=model(features,training=training)
    if classes==2:
        loss=tf.keras.losses.binary_crossentropy(tf.reshape(tf.cast(target,tf.float32),[1,1]),output)
    else:
        loss=tf.keras.losses.categorical_crossentropy(tf.one_hot([target],3),output,from_logits=True)
    data=tf.reduce_mean(loss)*tf.cast(class_weights[int(target)],tf.float32)
    regularization=tf.add_n(model.losses) if model.losses else tf.constant(0.,tf.float32)
    return data+regularization,output,regularization


def protocol():
    root,_=checked_source()
    return {'commit':COMMIT,'files':{n:digest_file(root/n) for n in ['milmodels.py','MIL.py','discriminator.py','intelsampling.py','data.py']},
        'binary_graph':'literal original1024/512 gated attention, all V/U/W L2 plus sigmoid classifier; inline source Dropout.2',
        'three_class_graph':'complete author commented AttMIL2 restored verbatim: three independent1024/512 gates, concat3072, classifier3/L2, inline Dropout.1',
        'discriminator':'actual full Keras ResNet50 truncated at conv4_block6_out, GAP1024, Dropout.25 and learned binary/three-class head',
        'known_source_failures':'discriminator.py indentation error and undefined data/encoder variables; full trainer must be explicitly reconstructed',
        'preprocessing_difference':'source int8 wrap then Caffe preprocessing and Rescaling1/255; paper-reference uint8 Caffe without extra rescale is named separately',
        'paper_reproduction':'not_run','real_wsi_validation':'not_run'}
