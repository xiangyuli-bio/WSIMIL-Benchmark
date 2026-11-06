"""Isolated native-TensorFlow complete Attention2majority trainer and evaluator.

This module is invoked with the method's TensorFlow environment, never imported
by the common Torch process. Missing author orchestration is reconstructed with
explicit TRAIN/VAL boundaries and replayable per-update seeds. Original graphs,
regularizers, native widths, actual learned image features and stopping behavior
remain observable. No generic frozen-feature trainer replaces these stages.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from .attention2majority_source import (build_discriminator,build_aggregator,
    binary_selection,majority_selection,objective,preprocess,protocol)
from .io import digest_file,digest_object,read_json,write_json


def runtime_identity():
    import importlib.metadata
    root=Path(__file__).resolve().parent
    return {'tensorflow':tf.__version__,'keras':importlib.metadata.version('keras'),
        'numpy':np.__version__,'h5py':h5py.__version__,'python':str(Path(sys.executable).resolve()),
        'tensorflow_build':dict(tf.sysconfig.get_build_info()),
        'NVIDIA_runtime_packages':{p.metadata['Name']:p.version for p in importlib.metadata.distributions()
            if p.metadata.get('Name','').lower().startswith('nvidia-')},
        'runtime_source_sha256':digest_object({str(p.relative_to(root)):digest_file(p) for p in sorted(root.rglob('*.py'))})}


def seeded(seed,stage,step):
    value=int(hashlib.sha256(f'{seed}:{stage}:{step}'.encode()).hexdigest()[:8],16)%(2**31-1)
    tf.keras.utils.set_random_seed(value)
    return value


def check_inputs(binding):
    for name,sha in binding['file_hashes'].items():
        if digest_file(name)!=sha:raise ValueError('Attention2majority task/patient/split/pool metadata changed')
    for row in binding['records'].values():
        p=Path(row['path']);stat=p.stat()
        if stat.st_size!=row['bytes'] or stat.st_mtime_ns!=row['mtime_ns']:
            raise ValueError('Attention2majority original raw RGB/patch identities changed')
    if digest_file(binding['weights'])!=binding['weights_sha256']:raise ValueError('Original Keras ImageNet weight bytes changed')


def pixels(binding,order):
    """Read every requested original patch in its actual sampled order."""
    groups={}
    for position,(slide,index) in enumerate(order):groups.setdefault(slide,[]).append((position,int(index)))
    output=[None]*len(order)
    for slide,items in groups.items():
        indices=sorted({i for _,i in items})
        with h5py.File(binding['records'][slide]['path'],'r') as f:data=f['pixels'][indices]
        lookup={i:image for i,image in zip(indices,data)}
        for position,index in items:output[position]=lookup[index]
    return np.stack(output)


def sampled_patches(binding,seed,subset,pool_only):
    rng=np.random.RandomState(seed);cap=2000 if binding['mode']=='camelyon_binary' else 200
    slides={r['slide_id']:r for r in binding['rows'] if r['subset']==subset}
    selected=binding['pool'] if pool_only else list(slides)
    ledger=[]
    for slide in selected:
        row=slides[slide];count=binding['records'][slide]['count']
        indices=np.arange(count) if count<cap else rng.choice(count,size=cap,replace=False)
        ledger += [{'slide_id':slide,'patch':int(i),'target':row['target']} for i in indices]
    if not ledger:raise ValueError('Empty original Attention2majority discriminator patch pool')
    return ledger


def patch_order(ledger,classes,seed):
    streams=[]
    for target in range(classes):
        ids=np.array([i for i,row in enumerate(ledger) if row['target']==target],dtype=np.int64)
        if not len(ids):raise ValueError('Actual discriminator TRAIN pool is missing a class')
        streams.append(tf.data.Dataset.from_tensor_slices(ids).shuffle(
            23838 if target!=1 else 23999,seed=seed+target,reshuffle_each_iteration=False))
    ds=tf.data.Dataset.sample_from_datasets(streams,weights=[1./classes]*classes,seed=seed,
        stop_on_empty_dataset=False,rerandomize_each_iteration=False)
    options=tf.data.Options();options.experimental_deterministic=True
    options.threading.private_threadpool_size=1;options.threading.max_intra_op_parallelism=2
    order=list(ds.with_options(options).as_numpy_iterator())
    if sorted(map(int,order))!=list(range(len(ledger))):raise ValueError('Original finite discriminator stream omitted/duplicated selected patches')
    return np.asarray(order,dtype=np.int64)


def tf_checkpoint(model,optimizer,folder,state,number):
    state={**state,'model_variable_devices':sorted({v.device for v in model.weights}),
        'optimizer_variable_devices':sorted({v.device for v in optimizer.variables()}),
        'model_parameters':model.count_params(),'trainable_tensors':len(model.trainable_variables)}
    checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer)
    manager=tf.train.CheckpointManager(checkpoint,str(folder/'checkpoints'),max_to_keep=2)
    prefix=manager.save(checkpoint_number=number)
    state={**state,'checkpoint_prefix':prefix,'checkpoint_files':{
        str(p):digest_file(p) for p in sorted(Path(prefix).parent.glob(Path(prefix).name+'.*'))}}
    write_json(folder/'state.json',state)
    return state


def restore_checkpoint(model,optimizer,state):
    if any(digest_file(p)!=sha for p,sha in state['checkpoint_files'].items()):
        raise ValueError('Native TensorFlow complete checkpoint bytes changed')
    optimizer.build(model.trainable_variables)
    tf.train.Checkpoint(model=model,optimizer=optimizer).restore(state['checkpoint_prefix']).assert_consumed()
    if int(optimizer.iterations)!=state['updates']:raise ValueError('Attention2majority full Adam iteration state was not restored')


def classification_loss(output,targets,classes):
    if classes==2:
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.cast(np.asarray(targets).reshape(-1,1),tf.float32),output))
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.one_hot(targets,3),output))


def correct(output,targets,classes):
    prediction=(output.numpy().reshape(-1)>.5).astype(np.int64) if classes==2 else output.numpy().argmax(1)
    return int((prediction==np.asarray(targets)).sum())


def discriminator(binding,request,out,resume,stop):
    classes=len(binding['classes']);cfg=request['config'];folder=out/'discriminator';folder.mkdir(exist_ok=resume)
    model,encoder=build_discriminator(binding['weights'],224 if classes==2 else 128,classes,cfg['preprocessing'])
    optimizer=tf.keras.optimizers.Adam(cfg['discriminator_lr'])
    signature=digest_object({'request':request,'binding':binding,'stage':'discriminator'})
    ledger=sampled_patches(binding,request['seed'],'train',True)
    validation=sampled_patches(binding,request['seed']+1,'val',False)
    history=[];updates=0;start=0;best=-math.inf;bad=0
    if (folder/'state.json').exists():
        state=read_json(folder/'state.json')
        if state['signature']!=signature:raise ValueError('Discriminator recovery changed complete input/recipe/software binding')
        restore_checkpoint(model,optimizer,state)
        history=state['history'];updates=state['updates'];start=state['epochs'];best=state['best_train_accuracy'];bad=state['bad']
        if state['state']=='completed':return model,encoder,state
    write_json(folder/'selected_TRAIN_patches.json',ledger)
    write_json(folder/'selected_VAL_patches.json',validation)
    for epoch in range(start,cfg['discriminator_epochs']):
        check_inputs(binding);seed=seeded(request['seed'],'discriminator_order',epoch)
        order=patch_order(ledger,classes,seed);np.save(folder/f'order_epoch{epoch}.npy',order,allow_pickle=False)
        loss_sum=0.;count=0;matches=0
        for offset in range(0,len(order),104):
            batch=[ledger[i] for i in order[offset:offset+104]]
            image=preprocess(pixels(binding,[(r['slide_id'],r['patch']) for r in batch]),cfg['preprocessing'])
            targets=[r['target'] for r in batch];seeded(request['seed'],'discriminator_update',updates)
            with tf.GradientTape() as tape:
                predictions=model(image,training=True);loss=classification_loss(predictions,targets,classes)
            gradients=tape.gradient(loss,model.trainable_variables)
            if any(g is None or not bool(tf.reduce_all(tf.math.is_finite(g)).numpy()) for g in gradients):
                raise FloatingPointError('Full original ResNet50/head gradient became missing or nonfinite')
            optimizer.apply_gradients(zip(gradients,model.trainable_variables));updates+=1
            loss_sum+=float(loss)*len(batch);count+=len(batch);matches+=correct(predictions,targets,classes)
        val_count=0;val_correct=0
        for offset in range(0,len(validation),104):
            batch=validation[offset:offset+104]
            image=preprocess(pixels(binding,[(r['slide_id'],r['patch']) for r in batch]),cfg['preprocessing'])
            val_correct+=correct(model(image,training=False),[r['target'] for r in batch],classes);val_count+=len(batch)
        accuracy=matches/count;val_accuracy=val_correct/val_count
        if accuracy>best:best=accuracy;bad=0
        else:bad+=1
        history.append({'epoch':epoch,'loss':loss_sum/count,'accuracy':accuracy,'val_accuracy':val_accuracy,
            'patches':count,'updates':updates,'order_path':str(folder/f'order_epoch{epoch}.npy'),
            'order_sha256':digest_file(folder/f'order_epoch{epoch}.npy')})
        done=epoch+1==cfg['discriminator_epochs'] or (cfg['discriminator_early_stopping'] and bad>=10)
        interrupted=not done and stop=={'stage':'discriminator','epochs':epoch+1}
        state=tf_checkpoint(model,optimizer,folder,{'signature':signature,'history':history,'updates':updates,
            'epochs':epoch+1,'best_train_accuracy':best,'bad':bad,'state':'completed' if done else 'interrupted' if interrupted else 'running',
            'native_patch_batch':104,'selected_model':'last_completed_epoch_no_restore_best_weights',
            'TRAIN_patients_only_for_updates':True,'seed_strategy':'explicit_per_update_global_seed_with_recorded_order'},epoch+1)
        print('discriminator',epoch+1,history[-1],flush=True)
        if done or interrupted:return model,encoder,state
    raise AssertionError('Discriminator budget produced no actual completed epoch')


def extract(binding,request,model,encoder,disc_state,out):
    folder=out/'features';folder.mkdir(exist_ok=True)
    contract=digest_object({'binding':binding,'request':request,'filter_checkpoint':disc_state['checkpoint_files']})
    if (folder/'manifest.json').exists():
        meta=read_json(folder/'manifest.json')
        if meta['contract']!=contract or any(digest_file(p)!=h for p,h in meta['files'].items()):
            raise ValueError('Task-trained discriminator features/probabilities/selected patch identities changed')
        return meta
    classes=len(binding['classes']);cfg=request['config'];records={};files={}
    for row in binding['rows']:
        slide=row['slide_id'];count=binding['records'][slide]['count'];all_features=[];all_scores=[]
        for offset in range(0,count,50):
            image=preprocess(pixels(binding,[(slide,i) for i in range(offset,min(count,offset+50))]),cfg['preprocessing'])
            all_features.append(encoder(image,training=False).numpy());all_scores.append(model(image,training=False).numpy())
        features=np.concatenate(all_features);scores=np.concatenate(all_scores)
        if features.shape!=(count,1024) or not np.isfinite(features).all():raise ValueError('Original complete task-trained1024D features were not extracted')
        if classes==2:
            selected,indices=binary_selection(features,scores,.3,cfg['zero_tail_repaired'])
        else:selected,indices=majority_selection(features,scores,2000)
        path=folder/(digest_object(slide)+'.npz')
        np.savez(path,all_features=features,all_probabilities=scores,selected_features=selected,selected_indices=indices)
        files[str(path)]=digest_file(path)
        ids=binding['records'][slide]['patch_ids'];records[slide]={'path':str(path),'original_count':count,
            'selected_indices':indices.tolist(),'selected_patch_ids':[ids[int(i)] for i in indices],
            'selected_count':len(indices),'original_patch_ids_sha256':binding['records'][slide]['patch_ids_sha256']}
    meta={'contract':contract,'records':records,'files':files,'discriminator_checkpoint':disc_state['checkpoint_files'],
        'trained_discriminator_shared_encoder':True,'all_original_patches_scored':True}
    write_json(folder/'manifest.json',meta);return meta


def bag_predictions(model,binding,cache,subset):
    classes=len(binding['classes']);result=[]
    for row in [r for r in binding['rows'] if r['subset']==subset]:
        with np.load(cache['records'][row['slide_id']]['path'],allow_pickle=False) as a:x=a['selected_features']
        raw=model(x[None].astype(np.float32),training=False)
        p=np.concatenate([1.-raw.numpy()[0],raw.numpy()[0]]) if classes==2 else tf.nn.softmax(raw,axis=-1).numpy()[0]
        result.append({**{k:row[k] for k in ['slide_id','patient_id','global_patient_id','target']},
            'subset':subset,'source_slides':row['slide_id'],'prediction':int(p.argmax()),
            **{f'p_{i}':float(value) for i,value in enumerate(p)}})
    return result


def aggregation(binding,request,cache,out,resume,stop):
    cfg=request['config'];classes=len(binding['classes']);folder=out/'aggregation';folder.mkdir(exist_ok=resume)
    seeded(request['seed'],'MIL_initialization',0);model=build_aggregator(classes)
    rows=[r for r in binding['rows'] if r['subset']=='train'];counts=np.bincount([r['target'] for r in rows],minlength=classes)
    class_weights={i:len(rows)/(classes*int(c)) for i,c in enumerate(counts)}
    schedule=tf.keras.optimizers.schedules.InverseTimeDecay(.0002,decay_steps=len(rows)*10,decay_rate=1,staircase=False)
    optimizer=tf.keras.optimizers.Adam(schedule);signature=digest_object({'request':request,'binding':binding,'cache':cache,'stage':'MIL'})
    history=[];start=0;updates=0;best=-math.inf;bad=0
    if (folder/'state.json').exists():
        state=read_json(folder/'state.json')
        if state['signature']!=signature:raise ValueError('MIL recovery changed task-trained instances, original graphs or complete recipe')
        restore_checkpoint(model,optimizer,state);history=state['history'];start=state['epochs'];updates=state['updates']
        best=state['best_train_accuracy'];bad=state['bad']
        if state['state']=='completed':return model,state
    for epoch in range(start,cfg['epochs']):
        check_inputs(binding)
        if any(digest_file(p)!=h for p,h in cache['files'].items()):raise ValueError('Actual discriminator-selected feature bytes changed')
        order=np.random.RandomState(seeded(request['seed'],'MIL_order',epoch)).permutation(len(rows)).tolist()
        losses=[];matches=0
        for index in order:
            row=rows[index]
            with np.load(cache['records'][row['slide_id']]['path'],allow_pickle=False) as a:x=a['selected_features'].astype(np.float32)
            seeded(request['seed'],'MIL_update',updates)
            with tf.GradientTape() as tape:loss,pred,reg=objective(model,x[None],row['target'],class_weights,classes)
            gradients=tape.gradient(loss,model.trainable_variables)
            if any(g is None or not bool(tf.reduce_all(tf.math.is_finite(g)).numpy()) for g in gradients):raise FloatingPointError('Complete Attention2majority gate/head gradient missing/nonfinite')
            optimizer.apply_gradients(zip(gradients,model.trainable_variables));updates+=1
            losses.append(float(loss));matches+=correct(tf.nn.softmax(pred,axis=-1) if classes==3 else pred,[row['target']],classes)
        accuracy=matches/len(rows)
        if accuracy>best:best=accuracy;bad=0
        else:bad+=1
        val=bag_predictions(model,binding,cache,'val');val_accuracy=sum(r['prediction']==r['target'] for r in val)/len(val)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'accuracy':accuracy,'val_accuracy':val_accuracy,
            'TRAIN_order':[rows[i]['slide_id'] for i in order],'updates':updates,'class_weights':class_weights})
        done=epoch+1==cfg['epochs'] or bad>=10;interrupted=not done and stop=={'stage':'MIL','epochs':epoch+1}
        state=tf_checkpoint(model,optimizer,folder,{'signature':signature,'epochs':epoch+1,'updates':updates,
            'history':history,'best_train_accuracy':best,'bad':bad,'state':'completed' if done else 'interrupted' if interrupted else 'running',
            'selected_model':'last_completed_epoch_source_EarlyStopping_restore_best_weights_False',
            'selection_not_test_or_validation_score':True},epoch+1)
        print('MIL',epoch+1,history[-1],flush=True)
        if done or interrupted:return model,state
    raise AssertionError('MIL training budget produced no actual epoch')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binding',required=True);parser.add_argument('--request',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--evaluate',choices=['val','test'])
    parser.add_argument('--stop-stage',choices=['discriminator','MIL']);parser.add_argument('--stop-after',type=int)
    a=parser.parse_args();out=Path(a.output);binding=read_json(a.binding);request=read_json(a.request)
    if tf.__version__!='2.15.1':raise ValueError('Pinned original graph integration requires TensorFlow2.15.1')
    if runtime_identity()!=request['tensorflow_runtime']:
        raise ValueError('Actual isolated TensorFlow/software environment differs from the locked complete method request')
    device=request['device']
    if device.startswith('cuda:'):
        devices=tf.config.list_physical_devices('GPU');index=int(device[5:])
        if len(devices)<=index:raise ValueError('Requested real TensorFlow CUDA device unavailable; no CPU fallback')
        tf.config.set_visible_devices(devices[index],'GPU');tf.config.experimental.set_memory_growth(devices[index],True)
    elif device!='cpu':raise ValueError('Explicit cpu or cuda:N is required')
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.threading.set_intra_op_parallelism_threads(request['config']['threads'])
    tf.keras.utils.set_random_seed(request['seed']);check_inputs(binding)
    stop={'stage':a.stop_stage,'epochs':a.stop_after} if a.stop_stage else None
    if a.evaluate:
        disc=read_json(out/'discriminator/state.json');cache=read_json(out/'features/manifest.json')
        if disc['state']!='completed' or any(digest_file(p)!=h for p,h in cache['files'].items()):raise ValueError('Actual complete trained feature cache is unavailable')
        model=build_aggregator(len(binding['classes']));n=sum(r['subset']=='train' for r in binding['rows'])
        opt=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(.0002,n*10,1,False))
        state=read_json(out/'aggregation/state.json')
        if state['state']!='completed':raise ValueError('Evaluation requires completed original MIL stopping budget')
        restore_checkpoint(model,opt,state);write_json(out/(a.evaluate+'_predictions_native.json'),bag_predictions(model,binding,cache,a.evaluate));return
    model,encoder,state=discriminator(binding,request,out,a.resume,stop)
    if state['state']!='completed':write_json(out/'native_status.json',{'state':'interrupted','stage':'discriminator'});return
    cache=extract(binding,request,model,encoder,state,out)
    del model,encoder;tf.keras.backend.clear_session()
    _,state=aggregation(binding,request,cache,out,a.resume,stop)
    write_json(out/'native_status.json',{'state':state['state'],'stage':'MIL','epochs':state['epochs'],
        'updates':state['updates'],'native_TensorFlow':tf.__version__,'protocol':protocol(),
        'formal_admission':False,'real_wsi_validation':'not_run','paper_reproduction':'not_run'})


if __name__=='__main__':main()
