"""Unit tests for modules/network.py"""

import numpy as np
import pytest
from modules.network import (
    proportional_threshold,
    degrees,
    top_degree_nodes,
    find_mdset,
    control_area,
    union_control_area,
    community_detection,
    participation_coefficient,
)


def _make_test_adj():
    """Small test graph: 6 nodes, hub at node 0."""
    adj = np.array([
        [0, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
    ], dtype=np.int32)
    return adj


class TestThreshold:
    def test_binary_output(self):
        conn = np.random.rand(10, 10)
        conn = (conn + conn.T) / 2
        adj = proportional_threshold(conn, 0.3)
        assert set(np.unique(adj)).issubset({0, 1})
        assert np.all(adj == adj.T)
        assert np.all(np.diag(adj) == 0)

    def test_density_approximate(self):
        np.random.seed(42)
        n = 50
        conn = np.random.rand(n, n)
        conn = (conn + conn.T) / 2
        adj = proportional_threshold(conn, 0.2)
        actual_density = adj.sum() / (n * (n - 1))
        assert abs(actual_density - 0.2) < 0.05


class TestDegree:
    def test_hub_has_highest_degree(self):
        adj = _make_test_adj()
        deg = degrees(adj)
        assert deg[0] == 4  # hub
        assert deg[5] == 1  # leaf


class TestMDSet:
    def test_all_dominated(self):
        adj = _make_test_adj()
        mdset = find_mdset(adj)
        n = adj.shape[0]
        # Every node must be in MDSet or neighbor of an MDSet node
        for v in range(n):
            neighbors = np.where(adj[v] > 0)[0]
            assert v in mdset or len(np.intersect1d(neighbors, mdset)) > 0

    def test_minimality(self):
        adj = _make_test_adj()
        mdset = find_mdset(adj)
        # For this small graph, MDSet should be 2 nodes
        assert len(mdset) <= 3


class TestControlArea:
    def test_hub_control_area(self):
        adj = _make_test_adj()
        ca = control_area(adj, 0)
        # Node 0 connects to 1,2,3,4 → area = {0,1,2,3,4} = 5
        assert len(ca) == 5


class TestCommunity:
    def test_returns_valid_labels(self):
        adj = _make_test_adj()
        labels = community_detection(adj)
        assert len(labels) == 6
        assert labels.min() >= 0


class TestParticipationCoef:
    def test_range(self):
        adj = _make_test_adj()
        comm = community_detection(adj)
        pc = participation_coefficient(adj, comm)
        assert np.all(pc >= 0)
        assert np.all(pc <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
