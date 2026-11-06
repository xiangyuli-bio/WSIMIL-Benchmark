import numpy as np
import pytest

from milplus.mgtrans_source import source_definitions


@pytest.mark.parametrize('layout', ['unique', 'repeated', 'missing_tissue'])
def test_actual_hnsw_budget_self_first_and_geometric_distance_ties(layout):
    _, source = source_definitions()
    points = np.asarray([(x*1024, y*1024) for y in range(16) for x in range(32)], dtype=np.int64)
    if layout == 'repeated':
        points = np.tile(points[:16], (32, 1))
    elif layout == 'missing_tissue':
        points = points[np.arange(len(points)) % 3 != 0]
    index = source.Hnsw(space='l2').fit(points)
    assert index.index_params_ == {'M': 16, 'post': 0, 'efConstruction': 400, 'indexThreadQty': 1}
    assert index.query_params_ == {'ef': 90}
    for node in [0, 1, 15, len(points)//2, len(points)-1]:
        selected = index.query(points[node], topn=9, self_index=node)
        assert selected[0] == node and len(set(selected)) == 9
        distances = np.square(points.astype(np.float64)-points[node]).sum(1)
        # A small measured grid has unambiguous exact-distance reference ranks;
        # retain the native HNSW query, with explicit node-ID tie choices only.
        candidates = np.asarray([i for i in range(len(points)) if i != node])
        expected = candidates[np.lexsort((candidates, distances[candidates]))][:8]
        np.testing.assert_array_equal(selected[1:], expected)
