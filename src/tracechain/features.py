"""Graph feature computation utilities using scipy sparse matrices.

Computes in-degree, out-degree, PageRank (via sparse power iteration),
and mean neighbor degree without using label information.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def compute_degrees(
    adj: sp.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute out-degree, in-degree, and total degree from a directed sparse adjacency matrix.

    Args:
        adj: Directed adjacency matrix where adj[u, v] > 0 represents edge u -> v.

    Returns:
        tuple of (out_degree, in_degree, total_degree) as float32 1D numpy arrays.
    """
    out_degree = np.asarray(adj.sum(axis=1)).flatten().astype(np.float32)
    in_degree = np.asarray(adj.sum(axis=0)).flatten().astype(np.float32)
    total_degree = out_degree + in_degree
    return out_degree, in_degree, total_degree


def compute_pagerank(
    adj: sp.csr_matrix,
    alpha: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute PageRank scores using power iteration on a sparse adjacency matrix.

    Handles dangling nodes (out-degree 0) by uniform redistribution of probability mass.

    Args:
        adj: Directed adjacency matrix (CSR format).
        alpha: Damping factor (default: 0.85).
        max_iter: Maximum number of power iterations (default: 50).
        tol: L1 convergence tolerance (default: 1e-6).

    Returns:
        1D numpy array of PageRank scores (float32) summing to 1.0.
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.array([1.0], dtype=np.float32)

    out_degree = np.asarray(adj.sum(axis=1)).flatten().astype(np.float64)
    dangling = out_degree == 0

    inv_out = np.zeros(n, dtype=np.float64)
    nonzero = ~dangling
    inv_out[nonzero] = 1.0 / out_degree[nonzero]

    # Row-stochastic transition matrix P = D_inv * adj
    # P[i, j] is probability of transitioning from i to j
    d_inv = sp.diags(inv_out, format="csr")
    p_mat = d_inv.dot(adj)
    # p_mat_t is P^T, so (P^T * x)[j] = sum_i P[i, j] * x[i]
    p_mat_t = p_mat.transpose().tocsr()

    p = np.full(n, 1.0 / n, dtype=np.float64)
    base_val = (1.0 - alpha) / n

    for _ in range(max_iter):
        dangling_sum = float(p[dangling].sum())
        p_next = alpha * (p_mat_t.dot(p) + (dangling_sum / n)) + base_val
        err = float(np.sum(np.abs(p_next - p)))
        p = p_next
        if err < tol:
            break

    # Normalize to guarantee exact sum = 1.0
    p_sum = float(np.sum(p))
    if p_sum > 0:
        p /= p_sum

    return p.astype(np.float32)


def compute_mean_neighbor_degree(
    adj: sp.csr_matrix,
    total_degree: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the mean degree of neighbors for each node on the undirected graph.

    Args:
        adj: Directed adjacency matrix (CSR format).
        total_degree: Optional precomputed total degree array (in-degree + out-degree).

    Returns:
        1D numpy array of mean neighbor degrees (float32). Isolated nodes have 0.0.
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=np.float32)

    if total_degree is None:
        _, _, total_degree = compute_degrees(adj)

    # Build undirected binary adjacency without self-loops
    adj_undir = adj + adj.transpose()
    adj_undir = adj_undir.tocsr()
    adj_undir.setdiag(0)
    adj_undir.eliminate_zeros()

    adj_bin = adj_undir.astype(bool).astype(np.float32)
    neighbor_count = np.asarray(adj_bin.sum(axis=1)).flatten()

    neighbor_deg_sum = np.asarray(
        adj_bin.dot(total_degree.astype(np.float32))
    ).flatten()

    mean_neighbor_deg = np.zeros(n, dtype=np.float32)
    has_neighbors = neighbor_count > 0
    mean_neighbor_deg[has_neighbors] = (
        neighbor_deg_sum[has_neighbors] / neighbor_count[has_neighbors]
    )
    return mean_neighbor_deg


def compute_all_graph_features(
    adj: sp.csr_matrix,
    alpha: float = 0.85,
    pagerank_iter: int = 50,
) -> dict[str, np.ndarray]:
    """Compute all SPEC 4.2 graph features without using labels.

    Args:
        adj: Directed adjacency matrix (CSR format).
        alpha: PageRank damping factor.
        pagerank_iter: Number of PageRank iterations.

    Returns:
        dict mapping feature name to 1D float32 numpy array:
            - 'in_degree'
            - 'out_degree'
            - 'pagerank'
            - 'mean_neighbor_degree'
    """
    out_deg, in_deg, tot_deg = compute_degrees(adj)
    pr = compute_pagerank(adj, alpha=alpha, max_iter=pagerank_iter)
    mean_neigh = compute_mean_neighbor_degree(adj, total_degree=tot_deg)

    return {
        "in_degree": in_deg,
        "out_degree": out_deg,
        "pagerank": pr,
        "mean_neighbor_degree": mean_neigh,
    }
