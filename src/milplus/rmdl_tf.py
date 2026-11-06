"""Full original-graph RMDL localization, hard-negative mining and MIL training.

This is an explicitly named reconstruction of the unpublished orchestration.
The native graphs, physical scale, batches80/256, selected300 actual patches,
legacy Adam and7000 MIL updates are preserved. No frozen FM head replaces it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

from .io import digest_file, digest_object, read_json, write_json
from .rmdl_data import check_binding
from .rmdl_pixels import (Slide, annotation, training_candidates, balanced_batch,
    augmented_crop, grid_shape, select_instances, preprocess, BLOCK, BLOCK_STRIDE, PATCH)
from .rmdl_source import build_localizer, build_aggregator, normalize_features, objective, source_review, legacy_adam


def runtime_identity():
    import importlib.metadata
    root = Path(__file__).resolve().parent
    return {'tensorflow':tf.__version__,'keras':importlib.metadata.version('keras'),
            'numpy':np.__version__,'python':str(Path(sys.executable).resolve()),
            'build':dict(tf.sysconfig.get_build_info()),
            'pixel_dependencies':{name:importlib.metadata.version(name) for name in ['pillow','openslide-python','openslide-bin','h5py']},
            'NVIDIA_runtime_packages':{p.metadata['Name']:p.version for p in importlib.metadata.distributions()
                if p.metadata.get('Name','').lower().startswith('nvidia-')},
            'runtime_source_sha256':digest_object({str(p.relative_to(root)):digest_file(p) for p in sorted(root.rglob('*.py'))})}


def _seed(seed, stage, step):
    return int(hashlib.sha256(f'{seed}:{stage}:{step}'.encode()).hexdigest()[:8],16) % (2**31-1)


def _checkpoint(model, optimizer, folder, state):
    # Include all BN buffers, original normalization gamma/beta, explicit dropout
    # counters, Adam slots/iteration and the current learning-rate hypervariable.
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, str(folder/'checkpoints'), max_to_keep=2)
    prefix = manager.save(checkpoint_number=state['updates'])
    value = {**state,'checkpoint_prefix':prefix,'checkpoint_files':{
        str(p):digest_file(p) for p in sorted(Path(prefix).parent.glob(Path(prefix).name+'.*'))},
        'model_parameters':model.count_params(),'model_variable_devices':sorted({v.device for v in model.weights}),
        'optimizer_variable_devices':sorted({v.device for v in optimizer.variables()})}
    write_json(folder/'state.json',value)
    return value


def _restore(model, optimizer, state):
    if any(digest_file(p) != sha for p,sha in state['checkpoint_files'].items()):
        raise ValueError('Complete RMDL checkpoint bytes changed')
    # Match the actual uninterrupted update's lazy creation order and scope.
    # Its learning-rate assignment creates hypers before apply_gradients enters
    # the Adam scope. Slots/iteration created outside that scope had identical
    # numbers but changed the serialized TrackableObjectGraph.
    optimizer._create_hypers()
    with tf.name_scope(optimizer._name):
        optimizer._create_all_weights(model.trainable_variables)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    unused = checkpoint.save_counter
    checkpoint.restore(state['checkpoint_prefix']).assert_consumed()
    if int(optimizer.iterations) != state['updates']:
        raise ValueError('Complete RMDL Adam iteration/slot state was not recovered')


def _gradients(tape, loss, model, optimizer):
    gradients = tape.gradient(loss, model.trainable_variables)
    if not bool(tf.math.is_finite(loss)) or any(g is None or not bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients):
        raise FloatingPointError('A full RMDL graph gradient is missing or nonfinite')
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))


def _legacy_ce(probabilities, targets):
    p = probabilities / tf.reduce_sum(probabilities,axis=-1,keepdims=True)
    p = tf.clip_by_value(p,1e-7,1.-1e-7)
    return -tf.reduce_mean(tf.reduce_sum(tf.one_hot(targets,3,dtype=p.dtype)*tf.math.log(p),axis=-1))


def _read_localization_batch(binding, ledger, indices, seed):
    rng = np.random.RandomState(seed)
    images = []
    handles = {}
    try:
        for index in indices:
            row = ledger[int(index)];sid = row['slide_id']
            if binding['records'][sid]['subset'] != 'train':
                raise ValueError('Held-out slide entered localization updates')
            if sid not in handles:
                handles[sid] = Slide(binding['records'][sid])
            images.append(augmented_crop(handles[sid],row['center'],rng))
    finally:
        for slide in handles.values():
            slide.close()
    return preprocess(np.stack(images)), np.asarray([ledger[int(i)]['target'] for i in indices],dtype=np.int64)


def mine_hard_negatives(model, binding, initial):
    """Actual TRAIN-only false positives at224-stride annotated normal centers.

    Argmax-abnormal mining, one pass and adding all unique false positives are
    explicit reference choices; the original mining trainer was not published.
    """
    known = {(r['slide_id'],*r['center']) for r in initial}
    mined = []
    for sid,record in sorted(binding['records'].items()):
        if record['subset'] != 'train':
            continue
        width,height = record['working_size']
        xx,yy = np.meshgrid(np.arange(235,width-234,224),np.arange(235,height-234,224))
        labels = annotation(record,xx,yy)
        centers = [(int(x),int(y)) for x,y,t in zip(xx.flat,yy.flat,labels.flat)
                   if t == 0 and (sid,int(x),int(y)) not in known]
        with Slide(record) as slide:
            for offset in range(0,len(centers),80):
                batch = centers[offset:offset+80]
                pixels = preprocess(np.stack([slide.read(x-PATCH//2,y-PATCH//2,PATCH) for x,y in batch]))
                p = model(pixels,training=False).numpy()[:,0,0]
                for center,probability in zip(batch,p):
                    if int(np.argmax(probability)) != 0:
                        mined.append({'slide_id':sid,'center':list(center),'target':0,'origin':'TRAIN_annotated_false_positive',
                                      'mining_probability':probability.tolist()})
    return mined


def localizer(binding, request, out, stop):
    cfg = request['config'];folder = out/'localizer';folder.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(_seed(request['seed'],'localizer_initialization',0))
    model,encoder,initialization = build_localizer(binding['weights'])
    write_json(folder/'actual_original_ImageNet_initialization.json',initialization)
    optimizer = legacy_adam(learning_rate=1e-4,beta_1=.9,beta_2=.999,epsilon=1e-8,decay=.001)
    signature = digest_object({'request':request,'binding':binding,'stage':'full_original_localizer'})
    initial = training_candidates(binding['records'],_seed(request['seed'],'initial_pool',0),cfg['normals_per_slide'])
    ledger = initial;history = [];start = 0;mined = False
    if (folder/'state.json').exists():
        state = read_json(folder/'state.json')
        if state['signature'] != signature or state['initial_pool_sha256'] != digest_object(initial):
            raise ValueError('RMDL localization recovery changed actual supervision, graph or recipe')
        _restore(model,optimizer,state)
        history = state['history'];start = state['updates'];mined = state['mining_completed']
        if mined:
            extra = read_json(folder/'TRAIN_hard_negatives.json')
            if digest_object(extra) != state['hard_negatives_sha256']:
                raise ValueError('Actual mined TRAIN false-positive evidence changed')
            ledger = initial+extra
        if state['state'] == 'completed':
            return model,encoder,state
    else:
        write_json(folder/'initial_TRAIN_pixel_supervised_pool.json',initial)
    for step in range(start,cfg['localizer_iterations']):
        check_binding(binding)
        if step == cfg['hard_negative_after'] and not mined:
            check_binding(binding,deep=True)
            extra = mine_hard_negatives(model,binding,initial)
            write_json(folder/'TRAIN_hard_negatives.json',extra)
            ledger = initial+extra;mined = True
        indices = balanced_batch(ledger,request['seed'],step,batch=80)
        images,targets = _read_localization_batch(binding,ledger,indices,_seed(request['seed'],'localizer_augmentation',step))
        optimizer.learning_rate.assign(1e-4*(.9**(step//2000)))
        with tf.GradientTape() as tape:
            probabilities = model(images,training=True)[:,0,0]
            loss = _legacy_ce(probabilities,targets)
        _gradients(tape,loss,model,optimizer)
        history.append({'iteration':step+1,'loss':float(loss),'accuracy':float(np.mean(probabilities.numpy().argmax(-1)==targets)),
                        'learning_rate':float(optimizer.learning_rate.numpy()),'batch_size':80,
                        'class_counts':np.bincount(targets,minlength=3).tolist(),
                        'sample_indices_sha256':digest_object(indices.tolist()),'mining_completed':mined})
        interrupt = stop is not None and step+1 == stop
        if interrupt or step+1 == cfg['localizer_iterations'] or (step+1)%2000 == 0 or step+1 == cfg['hard_negative_after']:
            state = _checkpoint(model,optimizer,folder,{'state':'completed' if step+1 == cfg['localizer_iterations'] else 'interrupted' if interrupt else 'running',
                'signature':signature,'updates':step+1,'history':history,'initial_pool_sha256':digest_object(initial),
                'mining_completed':mined,'hard_negatives_sha256':digest_object(read_json(folder/'TRAIN_hard_negatives.json')) if mined else None,
                'TRAIN_only_pixel_labels':True,'reference_orchestration':True})
            if interrupt:
                return model,encoder,state
    if not state['mining_completed']:
        raise ValueError('Complete RMDL localization budget did not include actual TRAIN hard-negative mining')
    return model,encoder,state


def _slide_features(model, encoder, binding, sid, out, localizer_state):
    record = binding['records'][sid]
    folder = out/'selected'/digest_object(sid)
    signature = digest_object({'input_contract':binding['input_contract'],'slide_id':sid,
                               'localizer_checkpoints':localizer_state['checkpoint_files'],
                               'selection':'complete_FCN1899_stride1632_NMSarea0.68_100perclass_actual299_FConv_before_ReLU'})
    if (folder/'state.json').exists():
        state = read_json(folder/'state.json')
        if state['signature'] != signature or any(digest_file(folder/name) != sha for name,sha in state['files'].items()):
            raise ValueError('Actual RMDL selected pixels, probabilities or learned FConv cache changed')
        return np.load(folder/'features.npy',allow_pickle=False)
    folder.parent.mkdir(exist_ok=True)
    # Separate generation directories make a crash incomplete rather than a
    # valid cache. Preserve failed generations for diagnosis.
    generation = 0
    while (folder.parent/(folder.name+f'.generation{generation}.partial')).exists():
        generation += 1
    pending = folder.parent/(folder.name+f'.generation{generation}.partial');pending.mkdir()
    gy,gx = grid_shape(*record['working_size'])
    probabilities = np.empty((gy,gx,3),dtype=np.float32)
    with Slide(record) as slide:
        for by in range(0,gy,51):
            for bx in range(0,gx,51):
                image = slide.read((bx//51)*BLOCK_STRIDE,(by//51)*BLOCK_STRIDE,BLOCK)
                p = model(preprocess(image[None]),training=False).numpy()
                if p.shape != (1,51,51,3):
                    raise ValueError('Original full1899 localization graph must emit51x51 three-class probabilities')
                sy,sx = min(51,gy-by),min(51,gx-bx)
                probabilities[by:by+sy,bx:bx+sx] = p[0,:sy,:sx]
        selected = select_instances(probabilities)
        features = []
        for offset in range(0,len(selected),80):
            batch = selected[offset:offset+80]
            images = preprocess(np.stack([slide.read(*item['xy'],PATCH) for item in batch]))
            values = encoder(images,training=False).numpy()
            if values.shape != (len(batch),1,1,1024) or not np.isfinite(values).all():
                raise ValueError('RMDL needs actual selected299 RGB re-encoded into full pre-ReLU1024 FConv')
            features.extend(values[:,0,0])
    features = np.stack(features)
    if features.shape != (300,1024):
        raise ValueError('Complete RMDL instance selection did not yield300 actual learned features')
    np.save(pending/'features.npy',features,allow_pickle=False)
    np.save(pending/'localization_probabilities.npy',probabilities,allow_pickle=False)
    write_json(pending/'selected_actual_coordinates.json',selected)
    state = {'signature':signature,'slide_id':sid,'kind':'actual_learned_FConv1024_selected299_RGB',
             'shape':list(features.shape),'unique_physical_coordinates':len({tuple(r['xy']) for r in selected}),
             'cross_class_coordinate_reuse_is_allowed':True,'test_labels_used':False,
             'files':{p.name:digest_file(p) for p in sorted(pending.iterdir())}}
    write_json(pending/'state.json',state);os.rename(pending,folder)
    return features


def _predictions(model, binding, features, subset):
    rows = []
    for row in binding['identity']['rows']:
        if row['subset'] != subset:
            continue
        sid = row['slide_id'];x = normalize_features(features[sid])
        p = model(x[None],training=False).numpy()[0]
        if not np.isfinite(p).all():
            raise FloatingPointError('RMDL predicted nonfinite probabilities')
        rows.append({'slide_id':sid,'patient_id':row['patient_id'],'global_patient_id':row['global_patient_id'],
                     'target':int(row['target']),**{f'p_{i}':float(v) for i,v in enumerate(p)}})
    return rows


def aggregator(binding, request, out, features, stop):
    cfg = request['config'];folder = out/'MIL';folder.mkdir(exist_ok=True)
    seed = _seed(request['seed'],'MIL_initialization',0);tf.keras.utils.set_random_seed(seed)
    model,optimizer,dropouts = build_aggregator(classes=3,pooling=cfg['pooling'],seed=seed)
    signature = digest_object({'request':request,'binding':binding,'stage':'full_original_aggregation',
        'features':{sid:digest_file(out/'selected'/digest_object(sid)/'features.npy') for sid in sorted(features)}})
    start = 0;history = []
    if (folder/'state.json').exists():
        state = read_json(folder/'state.json')
        if state['signature'] != signature:
            raise ValueError('RMDL aggregation recovery changed full learned features, inputs or recipe')
        _restore(model,optimizer,state);start = state['updates'];history = state['history']
        if [int(layer.counter.numpy()) for layer in dropouts] != state['dropout_counters']:
            raise ValueError('Original-rate RMDL dropout replay counters were not recovered')
        if state['state'] == 'completed':
            return model,state
    train = [row for row in binding['identity']['rows'] if row['subset'] == 'train']
    normalized = {sid:normalize_features(x) for sid,x in features.items()}
    for step in range(start,cfg['MIL_iterations']):
        check_binding(binding)
        rng = np.random.RandomState(_seed(request['seed'],'MIL_batch_and_instance_permutation',step))
        indices = rng.choice(len(train),256,replace=True)
        x = np.stack([normalized[train[int(i)]['slide_id']][rng.permutation(300)] for i in indices])
        y = np.asarray([train[int(i)]['target'] for i in indices],dtype=np.int64)
        optimizer.learning_rate.assign(1e-3*(.8**(step//140)))
        with tf.GradientTape() as tape:
            loss,ce,l2,p = objective(model,x,y,True)
        _gradients(tape,loss,model,optimizer)
        history.append({'iteration':step+1,'loss':float(loss),'crossentropy':float(ce),'kernel_L2':float(l2),
            'accuracy':float(np.mean(p.numpy().argmax(-1)==y)),'learning_rate':float(optimizer.learning_rate.numpy()),
            'batch_size':256,'sample_indices_sha256':digest_object(indices.tolist())})
        interrupt = stop is not None and step+1 == stop
        if interrupt or step+1 == cfg['MIL_iterations'] or (step+1)%140 == 0:
            state = _checkpoint(model,optimizer,folder,{'state':'completed' if step+1 == cfg['MIL_iterations'] else 'interrupted' if interrupt else 'running',
                'signature':signature,'updates':step+1,'history':history,'dropout_counters':[int(d.counter.numpy()) for d in dropouts],
                'selection':'fixed_final_declared_iteration','native_MIL_batch':256,'native_instances':300,'native_FConv_width':1024})
            if interrupt:
                return model,state
    return model,state


def run(binding, request, out, *, stop_stage=None, stop_after=None, evaluate=None):
    check_binding(binding,deep=True)
    if request['tensorflow_runtime'] != runtime_identity():
        raise ValueError('Actual isolated RMDL runtime/source differs from its recorded request')
    if request['device'] == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
    else:
        devices = tf.config.list_physical_devices('GPU');index = int(request['device'][5:])
        if index >= len(devices):
            raise ValueError('Requested actual RMDL CUDA device is unavailable')
        tf.config.set_visible_devices(devices[index], 'GPU');tf.config.experimental.set_memory_growth(devices[index],True)
    tf.config.experimental.enable_op_determinism()
    if evaluate is not None:
        for stage in ['localizer','MIL']:
            path=out/stage/'state.json'
            if not path.exists() or read_json(path)['state']!='completed':
                raise ValueError('RMDL evaluation never trains an unfinished localization or aggregation stage')
    write_json(out/'original_graph_and_reference_choices.json',source_review())
    model,encoder,ls = localizer(binding,request,out,stop_after if stop_stage == 'localizer' else None)
    if ls['state'] != 'completed':
        write_json(out/'native_status.json',{'state':'interrupted','stage':'localizer','updates':ls['updates'],'test_evaluated':False})
        return
    features = {}
    subsets = {'train','val'} if evaluate is None else {'train','val',evaluate}
    for row in binding['identity']['rows']:
        if row['subset'] in subsets:
            features[row['slide_id']] = _slide_features(model,encoder,binding,row['slide_id'],out,ls)
    # Full localization is no longer updated. Release its large optimizer slots
    # by dropping the local scope before constructing the aggregation graph.
    del model,encoder
    import gc
    gc.collect()
    if evaluate is not None:
        # A test feature cache is outside the fitted aggregation signature.
        training_features = {sid:x for sid,x in features.items() if binding['records'][sid]['subset'] in {'train','val'}}
        mil,state = aggregator(binding,request,out,training_features,None)
        if state['state'] != 'completed':
            raise ValueError('RMDL evaluation cannot train an unfinished model')
        predictions = _predictions(mil,binding,features,evaluate)
        write_json(out/(evaluate+'_predictions_native.json'),predictions)
        return
    mil,state = aggregator(binding,request,out,features,stop_after if stop_stage == 'MIL' else None)
    if state['state'] == 'completed':
        write_json(out/'val_predictions_native.json',_predictions(mil,binding,features,'val'))
    write_json(out/'native_status.json',{'state':state['state'],'stage':'MIL','updates':state['updates'],
        'test_evaluated':False,'fixed_final_iteration_selection':True,'formal_admission':False})


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--binding',required=True);parser.add_argument('--request',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--stop-stage',choices=['localizer','MIL'])
    parser.add_argument('--stop-after',type=int);parser.add_argument('--evaluate',choices=['val','test'])
    args = parser.parse_args()
    run(read_json(args.binding),read_json(args.request),Path(args.output),stop_stage=args.stop_stage,stop_after=args.stop_after,evaluate=args.evaluate)


if __name__ == '__main__':
    main()
