"""Actual full TensorFlow author graphs, signed pixels, selectors, gradients and Adam."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from milplus.attention2majority_source import (binary_classes,commented_multiclass_class,
    build_aggregator,build_discriminator,binary_selection,majority_selection,objective,preprocess,protocol)
from milplus.io import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    p.add_argument('--scope',choices=['aggregation','all'],default='all')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.threading.set_intra_op_parallelism_threads(2)
    report={'state':'running','checks':[],'formal_admission':False,'protocol':protocol()}
    def passed(name):report['checks'].append(name);write_json(out/'validation.json',report);print('PASS',name,flush=True)
    # Native1024D and a genuine257-instance bag, no hidden-width reduction.
    features=np.random.default_rng(901).normal(size=(1,257,1024)).astype(np.float32)
    for classes in [2,3]:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(902)
        port=build_aggregator(classes)
        initialized=[v.numpy().copy() for v in port.weights]
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(902)
        if classes==2:
            reference=binary_classes().AttMILbinary();reference.build(inputshape=(None,None,1024))
        else:
            reference=commented_multiclass_class()();reference.build((None,None,1024),3)
        reference(tf.zeros([1,7,1024]),training=False)
        assert len(initialized)==len(reference.weights)
        for x,v in zip(initialized,reference.weights):np.testing.assert_array_equal(x,v.numpy())
        target=classes-1;class_weights={i:1.+i*.25 for i in range(classes)}
        # Independent Keras compiled-loss oracle includes all actual source L2.
        reference.compile(optimizer=tf.keras.optimizers.Adam(2e-4),loss=
            tf.keras.losses.BinaryCrossentropy() if classes==2 else tf.keras.losses.CategoricalCrossentropy(from_logits=True))
        tf.keras.utils.set_random_seed(903)
        with tf.GradientTape() as tape:
            y=reference(features,training=True)
            truth=tf.constant([[target]],tf.float32) if classes==2 else tf.one_hot([target],3)
            ref_loss=reference.compiled_loss(truth,y,sample_weight=tf.constant([class_weights[target]]),regularization_losses=reference.losses)
        ref_grad=tape.gradient(ref_loss,reference.trainable_variables)
        tf.keras.utils.set_random_seed(903)
        with tf.GradientTape() as tape:loss,pred,reg=objective(port,features,target,class_weights,classes)
        grad=tape.gradient(loss,port.trainable_variables)
        np.testing.assert_array_equal(pred.numpy(),y.numpy());np.testing.assert_array_equal(loss.numpy(),ref_loss.numpy())
        assert float(reg)>0 and all(g is not None and np.isfinite(g.numpy()).all() for g in grad)
        for x,y in zip(grad,ref_grad):np.testing.assert_array_equal(x.numpy(),y.numpy())
        optimizer=tf.keras.optimizers.Adam(2e-4);optimizer.apply_gradients(zip(grad,port.trainable_variables))
        reference.optimizer.apply_gradients(zip(ref_grad,reference.trainable_variables))
        for x,y in zip(port.weights,reference.weights):np.testing.assert_array_equal(x.numpy(),y.numpy())
        for x,y in zip(optimizer.variables(),reference.optimizer.variables()):np.testing.assert_array_equal(x.numpy(),y.numpy())
        assert int(optimizer.iterations)==1
        passed(f'literal_full_{classes}class_graph_init_all_logits_classweighted_loss_L2_every_gradient_Adam_slot_bit_exact')
    x=np.tile(np.arange(7,dtype=np.float32)[:,None],(1,1024));prob=np.array([[.99],[.8],[.1],[.2],[.3],[.4],[.45]])
    selected,ids=binary_selection(x,prob,.3);assert ids.tolist()==[0,3,2]
    # Source -0 keeps the entire negative group; named repair returns none.
    prob=np.array([[.9],[.8],[.1]])
    _,ids=binary_selection(x[:3],prob,.3);assert ids.tolist()==[0,2]
    _,ids=binary_selection(x[:3],prob,.3,True);assert ids.tolist()==[0]
    passed('literal_positive_and_negative_tail_order_and_original_negative_zero_discrepancy')
    scores=np.array([[.9,.05,.05],[.2,.7,.1],[.1,.8,.1],[.3,.1,.6],[.05,.9,.05]])
    _,ids=majority_selection(x[:5],scores,4);assert ids.tolist()==[4,2,1,0]
    passed('three_class_majority_frequency_then_confidence_queue_retains_real_original_indices')
    pixels=np.array([0,127,128,255],dtype=np.uint8).reshape(4,1,1,1).repeat(3,3)
    legacy=preprocess(pixels,'source_signed_int8_caffe_then_rescale').numpy()
    paper=preprocess(pixels,'paper_uint8_caffe').numpy()
    np.testing.assert_array_equal(legacy,tf.keras.applications.resnet.preprocess_input(tf.constant(pixels.astype(np.int8).astype(np.float32))).numpy())
    assert np.array_equal(paper[:2],legacy[:2]) and np.all(paper[2:]-legacy[2:]==256)
    passed('signed_int8_wrap_Caffe_preprocess_and_explicit_uint8_reference_separated')
    from milplus.upstream import load_definitions
    reader=load_definitions('attention2majority','camelyon_util.py',['load_discimage'],{'tf':tf}).load_discimage
    original_rgb=np.tile(np.arange(224,dtype=np.uint8)[None,:,None],(224,1,3))
    stream=tf.data.Dataset.from_generator(lambda:iter([(original_rgb.transpose(2,0,1),1)]),
        output_signature=(tf.TensorSpec((3,224,224),tf.int8),tf.TensorSpec((),tf.int32)))
    options=tf.data.Options();options.threading.private_threadpool_size=1
    loaded=list(stream.map(reader).with_options(options).as_numpy_iterator())[0]
    np.testing.assert_array_equal(loaded[0],preprocess(original_rgb[None],'source_signed_int8_caffe_then_rescale').numpy()[0])
    assert int(loaded[1])==1
    passed('actual_author_int8_generator_and_CHW_reader_match_native_RGB_adapter')

    if a.scope=='aggregation':
        report.update(state='passed_aggregation_components_only',discriminator_validation='pending_full_pretrained_weights')
        write_json(out/'validation.json',report);return
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(904)
    model,encoder=build_discriminator(a.weights)
    assert encoder.output_shape==(None,1024)
    # Execute the actual source constructor assignments after fixing only its
    # unrelated syntax indentation; no full trainer/undefined data code runs.
    import ast
    from milplus.attention2majority_source import checked_source
    from milplus.upstream import pinned_text
    root,sha=checked_source();text=pinned_text(root,sha,'discriminator.py')
    old='    return [tf.keras.callbacks.ModelCheckpoint';new='        return [tf.keras.callbacks.ModelCheckpoint'
    assert text.count(old)==1;text=text.replace(old,new)
    run=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assignments=[n for n in run.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'resnet50','resnet_block3','disc_model'} for t in n.targets)]
    assert len(assignments)==3
    for node in ast.walk(assignments[0]):
        if isinstance(node,ast.keyword) and node.arg=='weights':node.value=ast.Constant(str(a.weights))
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(904)
    namespace={'tf':tf,'models':tf.keras.models,'layers':tf.keras.layers}
    exec(compile(ast.fix_missing_locations(ast.Module(body=assignments,type_ignores=[])),str(root/'discriminator.py')+':model_only','exec'),namespace)
    reference=namespace['disc_model'];reference(tf.zeros([1,224,224,3]),training=False)
    for x,y in zip(model.weights,reference.weights):np.testing.assert_array_equal(x.numpy(),y.numpy())
    image=np.random.default_rng(905).integers(0,256,(2,224,224,3),dtype=np.uint8)
    before=[v.numpy().copy() for v in model.trainable_variables]
    optimizer=tf.keras.optimizers.Adam()
    tf.keras.utils.set_random_seed(906)
    with tf.GradientTape() as tape:
        pred=model(preprocess(image,'source_signed_int8_caffe_then_rescale'),training=True)
        loss=tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.constant([[0.],[1.]]),pred))
    gradients=tape.gradient(loss,model.trainable_variables)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
    tf.keras.utils.set_random_seed(906)
    with tf.GradientTape() as tape:
        literal_pred=reference(preprocess(image,'source_signed_int8_caffe_then_rescale'),training=True)
        literal_loss=tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.constant([[0.],[1.]]),literal_pred))
    literal_grads=tape.gradient(literal_loss,reference.trainable_variables)
    np.testing.assert_array_equal(pred.numpy(),literal_pred.numpy());np.testing.assert_array_equal(loss.numpy(),literal_loss.numpy())
    for x,y in zip(gradients,literal_grads):np.testing.assert_array_equal(x.numpy(),y.numpy())
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    literal_optimizer=tf.keras.optimizers.Adam();literal_optimizer.apply_gradients(zip(literal_grads,reference.trainable_variables))
    for x,y in zip(model.weights,reference.weights):np.testing.assert_array_equal(x.numpy(),y.numpy())
    for x,y in zip(optimizer.variables(),literal_optimizer.variables()):np.testing.assert_array_equal(x.numpy(),y.numpy())
    assert len(gradients)>100 and int(optimizer.iterations)==1
    assert not np.array_equal(before[0],model.trainable_variables[0].numpy())
    assert not np.array_equal(before[-2],model.trainable_variables[-2].numpy())
    state=tf.train.Checkpoint(model=model,optimizer=optimizer);path=state.save(str(out/'checkpoint'))
    saved=[v.numpy().copy() for v in model.weights]+[v.numpy().copy() for v in optimizer.variables()]
    for v in model.weights:v.assign(tf.zeros_like(v))
    state.restore(path).assert_consumed()
    for old,v in zip(saved,list(model.weights)+list(optimizer.variables())):np.testing.assert_array_equal(old,v.numpy())
    report['discriminator_parameters']=model.count_params();report['discriminator_trainable_tensors']=len(gradients)
    passed('complete_actual_pretrained_Keras_truncated_ResNet50_backbone_and_head_update_full_TF_checkpoint_restore')
    report.update(state='passed',real_wsi_validation='not_run',paper_reproduction='not_run');write_json(out/'validation.json',report)


if __name__=='__main__':main()
