"""Verified2018 source feature graphs and complete original AGCN epoch/evaluation."""
import networkx as nx
import numpy as np
import scipy.sparse as sp

from .upstream import load_definitions, verified_source

COMMIT = '3149752deba68067f1a39b9f3d212cb16ffaf560'


def graph_functions():
    _, sha = verified_source('deepgraphsurv')
    if sha != COMMIT:
        raise ValueError('DeepGraphSurv original source commit differs')
    return load_definitions('deepgraphsurv', 'utils/graph_laplacian.py',
        ['compute_adj_matrix', 'get_adjacency', 'compute_laplacian'], {'np': np, 'nx': nx, 'sp': sp})


def metric():
    tree = load_definitions('lifelines_metric', 'lifelines/utils/btree.py', ['_BTree'], {'np': np})._BTree
    return load_definitions('lifelines_metric', 'lifelines/utils/concordance.py',
        ['concordance_index', '_preprocess_scoring_data', '_concordance_summary_statistics',
         '_handle_pairs', '_concordance_ratio'], {'np': np, '_BTree': tree}).concordance_index


def training_functions(provider, batch_size):
    """Only checkpoint/clinical I/O repair; windowing, Cox and patient rules intact."""
    _, sha = verified_source('deepgraphsurv')
    if sha != COMMIT:
        raise ValueError('DeepGraphSurv original source commit differs')
    original_metric = metric()

    def diagnostic_metric(*args, **kwargs):
        # Original batchwise CI is undefined for singleton tails and for some
        # otherwise valid Cox minibatches. Keep this diagnostic undefined;
        # do not invent a score or prevent the actual original optimizer step.
        try:
            return original_metric(*args, **kwargs)
        except ZeroDivisionError:
            return float('nan')

    return load_definitions('deepgraphsurv', 'models/agcn_survival.py',
        ['train_one_epoch', 'retrieve_feature', 'get_patient_ci', 'get_patient_feature'],
        {'np': np, 'prepare_data': provider, 'find_labels': provider.find_labels,
         'DATA_NAME': 'explicit_manifest', 'DATASET_NAME': 'explicit_manifest', 'LABEL_DIR': '',
         'BATCH_SIZE': batch_size, 'concordance_index': diagnostic_metric, 'log_string': lambda message: None},
        replacements=[
            ('if epoch > 19 and epoch % 10 == 0:',
             'if False:  # Replaced by unified complete model/BN/optimizer/RNG checkpoints.'),
            ('    assert all_features.shape[0] == data.shape[0] == all_preds.shape[0]',
             ('    p_feature = get_patient_feature(test_pid, all_features)\n'
              '    assert all_features.shape[0] == data.shape[0] == all_preds.shape[0]'))])
