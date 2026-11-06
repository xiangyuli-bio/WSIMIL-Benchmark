"""Isolated TensorFlow reference. Run in validation/camil-tf-requirements.txt env.

Only fixed author Git definitions are executed. No PULS model code is imported.
"""
import argparse
import ast
import subprocess
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import tensorflow as tf
    import tensorflow_addons as tfa
    from einops import rearrange, reduce
    from tensorflow.keras import backend as K
    from tensorflow.keras import initializers, regularizers
    from tensorflow.keras.layers import multiply
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    raw_softmax_error = False
    try:
        K.softmax(tf.constant([1., 2.]))
    except ValueError:
        raw_softmax_error = True
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=args.source, text=True)
    if sha != args.commit or dirty:
        raise ValueError('TensorFlow reference source is not the locked clean commit')
    scope = {'tf': tf, 'tfa': tfa, 'np': np, 'K': K, 'initializers': initializers,
             'regularizers': regularizers, 'multiply': multiply, 'rearrange': rearrange, 'reduce': reduce}

    def load(path, names, patches=()):
        code = subprocess.check_output(['git', 'show', f'{sha}:{path}'], cwd=args.source, text=True)
        for old, new in patches:
            if old not in code:
                raise ValueError('Reference compatibility patch changed')
            code = code.replace(old, new)
        nodes = [n for n in ast.parse(code).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise ValueError('Reference definition missing')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(args.source/path), 'exec'), scope)  # noqa: S102

    load('nystromformer/utils.py', ['MoorePenrosePseudoinverse'])
    load('nystromformer/nystrom_attention.py', ['NystromAttention'],
         [('filters=(dim_head/heads) * heads', 'filters=int((dim_head/heads) * heads)')])
    load('models/custom_layers.py', ['MILAttentionLayer', 'NeighborAggregator', 'Last_Sigmoid', 'CustomAttention', 'encoder'],
         [('alpha = K.softmax(A_raw)', 'alpha = tf.nn.softmax(A_raw, axis=0)')])
    data = np.load(args.input)
    classes = int(data['classes'])
    encoder = scope['encoder']()
    pool = scope['MILAttentionLayer'](128, use_gated=True, kernel_regularizer=tf.keras.regularizers.l2(1e-5))
    head = scope['Last_Sigmoid'](classes if classes > 2 else 1, subtyping=classes > 2,
                                kernel_regularizer=tf.keras.regularizers.l2(1e-5))
    x = tf.Variable(data['x'])
    ids, weights = data['indices'], data['edge_weights']
    ij = np.stack([np.repeat(np.arange(int(x.shape[0])), ids.shape[1]), ids.ravel()], axis=1)
    adj = tf.sparse.reorder(tf.sparse.SparseTensor(ij, weights.ravel(), [int(x.shape[0]), int(x.shape[0])]))
    combined, raw = encoder([x, adj]); attention = pool(combined); head(attention*combined)
    mapping = [
        ('nystrom.qkv.weight', encoder.nyst_att.to_qkv.kernel, 'linear'),
        ('nystrom.output.weight', encoder.nyst_att.to_out.layers[0].kernel, 'linear'),
        ('nystrom.output.bias', encoder.nyst_att.to_out.layers[0].bias, 'identity'),
        ('nystrom.residual.weight', encoder.nyst_att.res_conv.layers[0].kernel, 'conv'),
        ('query.weight', encoder.custom_att.wq_weight_params, 'linear'),
        ('key.weight', encoder.custom_att.wk_weight_params, 'linear'),
        ('value.weight', encoder.wv.kernel, 'linear'),
        ('value.bias', encoder.wv.bias, 'identity'),
        ('pool_v.weight', pool.v_weight_params, 'linear'),
        ('pool_u.weight', pool.u_weight_params, 'linear'),
        ('pool_w.weight', pool.w_weight_params, 'linear'),
        ('classifier.weight', head.kernel, 'linear'),
        ('classifier.bias', head.bias, 'identity')]
    for name, var, transform in mapping:
        value = data[name]
        var.assign(value.T if transform == 'linear' else value.transpose(2, 3, 1, 0) if transform == 'conv' else value)
    with tf.GradientTape() as tape:
        combined, raw = encoder([x, adj])
        attention = pool(combined)
        probability = head(attention*combined)
        target = tf.constant([int(data['target'])])
        loss = (tf.keras.losses.SparseCategoricalCrossentropy()(target, probability) if classes > 2
                else tf.keras.losses.BinaryCrossentropy()(tf.cast(target[:, None], tf.float32), probability))
    grads = tape.gradient(loss, [x]+[var for _, var, _ in mapping])
    if any(g is None for g in grads):
        raise ValueError('Reference active variable has no gradient')
    result = {'probability': probability.numpy(), 'raw': raw.numpy(), 'attention': attention.numpy(),
              'loss': loss.numpy(), 'input_gradient': grads[0].numpy(),
              'tensorflow_version': np.array(tf.__version__), 'tensorflow_addons_version': np.array(tfa.__version__),
              'commit': np.array(sha), 'declared_but_unused_l2': np.array(float(sum(pool.losses+head.losses)))}
    result['raw_1d_softmax_rejected'] = np.array(raw_softmax_error)
    singleton_error = False
    try:
        singleton_adj = tf.sparse.SparseTensor([[0, 0]], [1.], [1, 1])
        encoder([x[:1], singleton_adj])
    except (ValueError, tf.errors.InvalidArgumentError):
        singleton_error = True
    result['raw_singleton_squeeze_rejected'] = np.array(singleton_error)
    for (name, _, transform), grad in zip(mapping, grads[1:]):
        value = grad.numpy()
        result['grad.'+name] = value.T if transform == 'linear' else value.transpose(3, 2, 0, 1) if transform == 'conv' else value
    # Independently compare the optimizer epsilon position with real Keras.
    variable = tf.Variable(np.array([1., -2., 3.], dtype=np.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=.01, beta_1=.9, beta_2=.999, epsilon=1e-7)
    updates = []
    for values in [[1e-5, .1, -.2], [.02, -.1, 1e-6], [.4, .5, -.6]]:
        optimizer.apply_gradients([(tf.constant(values), variable)])
        updates.append(variable.numpy().copy())
    result['adam_updates'] = np.stack(updates)
    scope['Callback'] = tf.keras.callbacks.Callback
    load('models/custom_callbacks.py', ['CustomReduceLRoP'])
    lr = tf.Variable(.01)
    scheduler = scope['CustomReduceLRoP'](patience=10, factor=.2, optim_lr=lr, mode='min', reduce_lin=False)
    lrs = []
    for i in range(24):
        scheduler.on_epoch_end(i, 1.)
        lrs.append(float(lr.numpy()))
    result['plateau_lrs'] = np.array(lrs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **result)


if __name__ == '__main__':
    main()
