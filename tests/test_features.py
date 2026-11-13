"""Unit tests for graph features using tiny synthetic graphs."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import scipy.sparse as sp

from tracechain.features import (
    compute_all_graph_features,
    compute_degrees,
    compute_mean_neighbor_degree,
    compute_pagerank,
)


def test_degrees_simple_cycle():
    """Test in/out/total degree computation on a 3-node directed cycle (0->1->2->0)."""
    # 0 -> 1, 1 -> 2, 2 -> 0
    row = np.array([0, 1, 2], dtype=np.int32)
    col = np.array([1, 2, 0], dtype=np.int32)
    data = np.ones(3, dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(3, 3))

    out_deg, in_deg, tot_deg = compute_degrees(adj)

    np.testing.assert_array_equal(out_deg, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(in_deg, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(tot_deg, [2.0, 2.0, 2.0])


def test_degrees_star_graph():
    """Test degrees on a directed star graph: hub (0) sends to 1, 2, 3."""
    row = np.array([0, 0, 0], dtype=np.int32)
    col = np.array([1, 2, 3], dtype=np.int32)
    data = np.ones(3, dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(4, 4))

    out_deg, in_deg, tot_deg = compute_degrees(adj)

    np.testing.assert_array_equal(out_deg, [3.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(in_deg, [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(tot_deg, [3.0, 1.0, 1.0, 1.0])


def test_pagerank_symmetric_cycle():
    """PageRank on a directed cycle should assign equal rank to all nodes."""
    row = np.array([0, 1, 2, 3], dtype=np.int32)
    col = np.array([1, 2, 3, 0], dtype=np.int32)
    data = np.ones(4, dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(4, 4))

    pr = compute_pagerank(adj, alpha=0.85, max_iter=100)

    assert len(pr) == 4
    assert np.isclose(np.sum(pr), 1.0, atol=1e-5)
    np.testing.assert_allclose(pr, [0.25, 0.25, 0.25, 0.25], atol=1e-5)


def test_pagerank_dangling_nodes():
    """PageRank must properly handle dangling nodes (nodes with 0 out-degree)."""
    # 0 -> 1, 0 -> 2; nodes 1 and 2 have no outgoing edges
    row = np.array([0, 0], dtype=np.int32)
    col = np.array([1, 2], dtype=np.int32)
    data = np.ones(2, dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(3, 3))

    pr = compute_pagerank(adj, alpha=0.85, max_iter=50)

    assert len(pr) == 3
    assert np.isclose(np.sum(pr), 1.0, atol=1e-5)
    # Node 1 and Node 2 should have equal PageRank by symmetry
    assert np.isclose(pr[1], pr[2], atol=1e-5)
    # Both receivers should have higher PageRank than the sender
    assert pr[1] > pr[0]


def test_mean_neighbor_degree_path():
    """Test mean neighbor degree on undirected path: 0 - 1 - 2.

    Degrees: deg(0)=1, deg(1)=2, deg(2)=1.
    Neighbors:
      node 0 neighbors: {1} (deg=2) -> mean = 2.0
      node 1 neighbors: {0, 2} (degs=1, 1) -> mean = (1+1)/2 = 1.0
      node 2 neighbors: {1} (deg=2) -> mean = 2.0
    """
    row = np.array([0, 1], dtype=np.int32)
    col = np.array([1, 2], dtype=np.int32)
    data = np.ones(2, dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(3, 3))

    mean_neigh = compute_mean_neighbor_degree(adj)

    np.testing.assert_allclose(mean_neigh, [2.0, 1.0, 2.0], atol=1e-5)


def test_mean_neighbor_degree_isolated_node():
    """An isolated node with degree 0 should have mean neighbor degree 0.0."""
    # Node 0 -> Node 1, Node 2 is isolated
    row = np.array([0], dtype=np.int32)
    col = np.array([1], dtype=np.int32)
    data = np.ones(1, dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(3, 3))

    mean_neigh = compute_mean_neighbor_degree(adj)

    assert mean_neigh[2] == 0.0


def test_compute_all_graph_features():
    """Test dict outputs of compute_all_graph_features."""
    row = np.array([0, 1, 2], dtype=np.int32)
    col = np.array([1, 2, 0], dtype=np.int32)
    data = np.ones(3, dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(3, 3))

    feat_dict = compute_all_graph_features(adj)

    assert set(feat_dict.keys()) == {
        "in_degree",
        "out_degree",
        "pagerank",
        "mean_neighbor_degree",
    }
    for k, v in feat_dict.items():
        assert isinstance(v, np.ndarray)
        assert v.dtype == np.float32
        assert len(v) == 3
