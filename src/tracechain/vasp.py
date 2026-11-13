"""VASP proxy seed selection, shortest-path attribution, and path reconstruction.

Implements SPEC section 4.3:
- Ranks addresses by exchange-like structural activity (total degree, in-degree,
  out-degree, number of active time steps, and total BTC moved) without using test labels.
- Selects top K proxy seeds (skipping addresses labelled illicit in the training period only),
  named VASP-PROXY-01, VASP-PROXY-02, etc.
- Supports optional registry override at data/registry/vasp_registry.csv (columns: address, name)
  with kind="registry".
- Runs multi-source shortest path (unweighted BFS via scipy.sparse.csgraph) on the undirected
  address graph in chunks to limit memory.
- Computes hop distances, nearest seed IDs, confidence scores, confidence labels, and
  next-hop predecessors for fast on-demand path reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, dijkstra
from scipy.stats import rankdata


UNATTRIBUTED_HOPS = -1
UNATTRIBUTED_PREDECESSOR = -1


def build_undirected_graph(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Build an unweighted, symmetrised CSR adjacency matrix without self-loops.

    Matches the undirected graph convention in src/tracechain/features.py.
    Note: In a symmetric CSR matrix, each undirected edge {u, v} (u != v) is stored
    as two directed entries (u, v) and (v, u), so `adj_undir.nnz` equals twice the
    number of distinct undirected edges.

    Args:
        adj: Directed CSR adjacency matrix of shape (N, N).

    Returns:
        Symmetrised binary CSR matrix of shape (N, N) with zero diagonal.
    """
    adj_undir = (adj + adj.transpose()).tocsr()
    adj_undir.setdiag(0)
    adj_undir.eliminate_zeros()
    if adj_undir.nnz > 0:
        adj_undir.data = np.ones_like(adj_undir.data, dtype=np.float32)
    return adj_undir


def compute_graph_edge_counts(
    adj_directed: sp.csr_matrix,
    adj_undir: sp.csr_matrix | None = None,
) -> dict[str, int]:
    """Compute exact directed and undirected edge counts from the sparse adjacency matrix.

    Args:
        adj_directed: Directed CSR adjacency matrix (may contain self-loops).
        adj_undir: Optional precomputed symmetrised zero-diagonal CSR matrix.

    Returns:
        Dictionary with exact counts:
          - nodes: number of graph nodes (N)
          - directed_edges_total: total distinct directed (u, v) pairs in adj_directed (adj_directed.nnz)
          - directed_self_loops: number of nodes u with a self-loop (u -> u) in adj_directed
          - directed_non_self_loop_edges: distinct directed (u -> v) edges with u != v
          - distinct_undirected_edges: distinct unordered node pairs {u, v} with u != v joined in either direction
          - reciprocal_directed_edge_pairs: unordered pairs {u, v} (u != v) that have edges in both directions (u -> v and v -> u)
          - symmetrised_matrix_nonzeros: total nonzero entries in the symmetric zero-diagonal CSR matrix (2 * distinct_undirected_edges)
    """
    n = int(adj_directed.shape[0])
    directed_total = int(adj_directed.nnz)

    adj_no_diag = adj_directed.copy().tocsr()
    adj_no_diag.setdiag(0)
    adj_no_diag.eliminate_zeros()
    directed_non_self = int(adj_no_diag.nnz)
    directed_self_loops = directed_total - directed_non_self

    if adj_undir is None:
        adj_undir = build_undirected_graph(adj_directed)

    sym_nnz = int(adj_undir.nnz)
    distinct_undirected = sym_nnz // 2
    reciprocal_pairs = directed_non_self - distinct_undirected

    return {
        "nodes": n,
        "directed_edges_total": directed_total,
        "directed_self_loops": directed_self_loops,
        "directed_non_self_loop_edges": directed_non_self,
        "distinct_undirected_edges": distinct_undirected,
        "reciprocal_directed_edge_pairs": reciprocal_pairs,
        "symmetrised_matrix_nonzeros": sym_nnz,
    }


def compute_seed_scores(
    in_degree: np.ndarray,
    out_degree: np.ndarray,
    active_time_steps: np.ndarray,
    total_value_moved: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the VASP proxy seed score as the rank-average of structural features.

    SPEC 4.3: Rank-average of total degree, in-degree, out-degree, number of active
    time steps, and total value moved if such a feature exists.

    Args:
        in_degree: 1D array of node in-degrees.
        out_degree: 1D array of node out-degrees.
        active_time_steps: 1D array of number of active time steps per node.
        total_value_moved: Optional 1D array of total value moved (btc_transacted_total).

    Returns:
        1D float32 array of normalized rank-average scores in [0, 1].
    """
    n = len(in_degree)
    if n == 0:
        return np.array([], dtype=np.float32)

    total_degree = in_degree.astype(np.float64) + out_degree.astype(np.float64)
    components = [
        total_degree,
        in_degree.astype(np.float64),
        out_degree.astype(np.float64),
        active_time_steps.astype(np.float64),
    ]
    if total_value_moved is not None:
        components.append(total_value_moved.astype(np.float64))

    rank_sum = np.zeros(n, dtype=np.float64)
    for arr in components:
        ranks = rankdata(np.nan_to_num(arr, nan=0.0), method="average")
        rank_sum += ranks / float(n)

    score = rank_sum / float(len(components))
    return score.astype(np.float32)


def select_proxy_seeds(
    idx_to_address: np.ndarray,
    seed_scores: np.ndarray,
    train_illicit_addresses: set[str],
    k: int = 30,
) -> pd.DataFrame:
    """Select the top K addresses by seed score, skipping training-period illicit addresses.

    SPEC 4.3: Take the top K=30 addresses as proxy seeds, skipping addresses labelled
    illicit in the training period only. Name them VASP-PROXY-01 and so on.

    Args:
        idx_to_address: 1D array mapping integer node index to address string.
        seed_scores: 1D array of seed scores aligned with idx_to_address.
        train_illicit_addresses: Set of addresses labelled illicit in the training split.
        k: Number of proxy seeds to select (default 30).

    Returns:
        DataFrame with columns: ['seed_id', 'name', 'address', 'node_idx', 'kind', 'seed_score'].
    """
    n = len(idx_to_address)
    if n == 0 or k <= 0:
        return pd.DataFrame(
            columns=["seed_id", "name", "address", "node_idx", "kind", "seed_score"]
        )

    # Sort descending by score, breaking ties deterministically by node_idx ascending
    order = np.lexsort((np.arange(n), -seed_scores.astype(np.float64)))

    rows: list[dict[str, Any]] = []
    width = max(2, len(str(k)))
    for idx in order:
        addr = str(idx_to_address[idx])
        if addr in train_illicit_addresses:
            continue
        rank_num = len(rows) + 1
        seed_name = f"VASP-PROXY-{rank_num:0{width}d}"
        rows.append(
            {
                "seed_id": seed_name,
                "name": seed_name,
                "address": addr,
                "node_idx": int(idx),
                "kind": "proxy",
                "seed_score": float(seed_scores[idx]),
            }
        )
        if len(rows) >= k:
            break

    return pd.DataFrame(rows)


def load_or_select_seeds(
    idx_to_address: np.ndarray,
    address_to_idx: dict[str, int],
    seed_scores: np.ndarray,
    train_illicit_addresses: set[str],
    k: int = 30,
    registry_path: Path | str | None = Path("data/registry/vasp_registry.csv"),
) -> pd.DataFrame:
    """Load real VASP seeds from registry CSV if present, otherwise select proxy seeds.

    SPEC 4.3: If data/registry/vasp_registry.csv exists (columns: address, name),
    those addresses become seeds with kind "registry".

    Args:
        idx_to_address: 1D array mapping integer node index to address string.
        address_to_idx: Dict mapping address string to integer node index.
        seed_scores: 1D array of proxy seed scores.
        train_illicit_addresses: Set of addresses labelled illicit in the training split.
        k: Number of proxy seeds if registry is not used.
        registry_path: Path to optional vasp_registry.csv.

    Returns:
        DataFrame with columns: ['seed_id', 'name', 'address', 'node_idx', 'kind', 'seed_score'].
    """
    if registry_path is not None:
        reg_file = Path(registry_path)
        if reg_file.is_file():
            reg_df = pd.read_csv(reg_file, dtype={"address": str, "name": str})
            reg_df = reg_df.dropna(subset=["address", "name"]).drop_duplicates(
                subset=["address"], keep="first"
            )
            rows: list[dict[str, Any]] = []
            width = max(2, len(str(len(reg_df))))
            for _, row in reg_df.iterrows():
                addr = str(row["address"]).strip()
                name = str(row["name"]).strip()
                if addr not in address_to_idx:
                    continue
                node_idx = int(address_to_idx[addr])
                rank_num = len(rows) + 1
                seed_id = f"VASP-REG-{rank_num:0{width}d}"
                rows.append(
                    {
                        "seed_id": seed_id,
                        "name": name,
                        "address": addr,
                        "node_idx": node_idx,
                        "kind": "registry",
                        "seed_score": float(seed_scores[node_idx])
                        if len(seed_scores) > node_idx
                        else 0.0,
                    }
                )
            if rows:
                return pd.DataFrame(rows)

    return select_proxy_seeds(
        idx_to_address=idx_to_address,
        seed_scores=seed_scores,
        train_illicit_addresses=train_illicit_addresses,
        k=k,
    )


def compute_confidence(hops: int | float | None) -> tuple[float | None, str]:
    """Compute confidence_score and confidence label from hop distance.

    SPEC 4.3:
    confidence_score = 1 / (1 + hops).
    Label: "high" if hops <= 2, "medium" if hops <= 4, "low" otherwise.
    Unreachable means "unattributed" with no hop value.

    Args:
        hops: Integer hop count, or None / negative / inf if unreachable.

    Returns:
        tuple of (confidence_score, confidence_label).
    """
    if hops is None or not np.isfinite(hops) or int(hops) < 0:
        return None, "unattributed"
    h = int(hops)
    score = round(1.0 / (1.0 + h), 4)
    if h <= 2:
        label = "high"
    elif h <= 4:
        label = "medium"
    else:
        label = "low"
    return score, label


def run_chunked_multisource_bfs(
    adj_undir: sp.csr_matrix,
    seed_indices: list[int] | np.ndarray,
    chunk_size: int = 10,
    return_predecessors: bool = True,
) -> dict[str, np.ndarray]:
    """Run multi-source shortest path (BFS) in seed chunks using scipy.sparse.csgraph.

    Processes seeds in chunks of `chunk_size` to bound peak memory while finding,
    for every node in the graph:
      - `min_hops`: shortest hop distance to any seed (-1 if unreachable),
      - `nearest_seed_pos`: index into `seed_indices` (0..len(seeds)-1) of the nearest seed
        (-1 if unreachable), breaking ties by lower seed position,
      - `next_hop_idx` (if `return_predecessors=True`): the neighbor node index one step
        closer to the nearest seed (-1 for seeds themselves or unreachable nodes).

    Args:
        adj_undir: Undirected unweighted CSR matrix of shape (N, N).
        seed_indices: Sequence of integer node indices for the seeds.
        chunk_size: Number of seeds to pass per csgraph call.
        return_predecessors: Whether to track next-hop predecessor along shortest path.

    Returns:
        Dictionary with numpy arrays 'min_hops', 'nearest_seed_pos', and optionally 'next_hop_idx'.
    """
    n = adj_undir.shape[0]
    seeds = [int(s) for s in seed_indices]
    best_dist = np.full(n, np.inf, dtype=np.float64)
    nearest_seed_pos = np.full(n, -1, dtype=np.int32)
    next_hop_idx = (
        np.full(n, UNATTRIBUTED_PREDECESSOR, dtype=np.int32)
        if return_predecessors
        else np.array([], dtype=np.int32)
    )

    if n == 0 or not seeds:
        return {
            "min_hops": np.full(n, UNATTRIBUTED_HOPS, dtype=np.int32),
            "nearest_seed_pos": nearest_seed_pos,
            "next_hop_idx": next_hop_idx,
        }

    step = max(1, int(chunk_size))
    for start in range(0, len(seeds), step):
        chunk_seeds = seeds[start : start + step]
        if return_predecessors:
            dist_chunk, pred_chunk = dijkstra(
                adj_undir,
                directed=False,
                indices=chunk_seeds,
                unweighted=True,
                return_predecessors=True,
            )
            if dist_chunk.ndim == 1:
                dist_chunk = dist_chunk[np.newaxis, :]
                pred_chunk = pred_chunk[np.newaxis, :]

            for local_i in range(len(chunk_seeds)):
                global_seed_pos = start + local_i
                d_row = dist_chunk[local_i]
                p_row = pred_chunk[local_i]
                improved = d_row < best_dist
                best_dist[improved] = d_row[improved]
                nearest_seed_pos[improved] = global_seed_pos
                # In csgraph, p_row[u] is the predecessor of u on the shortest path from seed -> u,
                # which on an undirected graph is the exact next hop from u toward the seed!
                # For the seed itself, p_row[seed] is -9999; normalize negative values to -1.
                valid_pred = np.where(p_row >= 0, p_row, UNATTRIBUTED_PREDECESSOR)
                next_hop_idx[improved] = valid_pred[improved]
        else:
            dist_chunk = dijkstra(
                adj_undir,
                directed=False,
                indices=chunk_seeds,
                unweighted=True,
                return_predecessors=False,
            )
            if dist_chunk.ndim == 1:
                dist_chunk = dist_chunk[np.newaxis, :]

            for local_i in range(len(chunk_seeds)):
                global_seed_pos = start + local_i
                d_row = dist_chunk[local_i]
                improved = d_row < best_dist
                best_dist[improved] = d_row[improved]
                nearest_seed_pos[improved] = global_seed_pos

    reachable = np.isfinite(best_dist)
    min_hops = np.full(n, UNATTRIBUTED_HOPS, dtype=np.int32)
    min_hops[reachable] = best_dist[reachable].astype(np.int32)

    return {
        "min_hops": min_hops,
        "nearest_seed_pos": nearest_seed_pos,
        "next_hop_idx": next_hop_idx,
    }


def build_attribution_dataframe(
    idx_to_address: np.ndarray,
    seeds_df: pd.DataFrame,
    bfs_result: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Create the per-address VASP attribution table for saving in artifacts/.

    Args:
        idx_to_address: 1D array mapping node_idx to address string.
        seeds_df: DataFrame of selected seeds from load_or_select_seeds.
        bfs_result: Output dictionary from run_chunked_multisource_bfs.

    Returns:
        DataFrame with columns:
        ['address', 'nearest_seed_id', 'nearest_seed_name', 'nearest_seed_address',
         'nearest_seed_kind', 'hops', 'confidence', 'confidence_score', 'next_hop_idx']
    """
    n = len(idx_to_address)
    min_hops = bfs_result["min_hops"]
    nearest_pos = bfs_result["nearest_seed_pos"]
    next_hop = bfs_result.get(
        "next_hop_idx", np.full(n, UNATTRIBUTED_PREDECESSOR, dtype=np.int32)
    )

    seed_ids = seeds_df["seed_id"].to_numpy(dtype=object)
    seed_names = seeds_df["name"].to_numpy(dtype=object)
    seed_addrs = seeds_df["address"].to_numpy(dtype=object)
    seed_kinds = seeds_df["kind"].to_numpy(dtype=object)

    reachable = nearest_pos >= 0

    nearest_seed_id = np.full(n, None, dtype=object)
    nearest_seed_name = np.full(n, None, dtype=object)
    nearest_seed_addr = np.full(n, None, dtype=object)
    nearest_seed_kind = np.full(n, None, dtype=object)

    if len(seeds_df) > 0:
        valid_pos = nearest_pos[reachable]
        nearest_seed_id[reachable] = seed_ids[valid_pos]
        nearest_seed_name[reachable] = seed_names[valid_pos]
        nearest_seed_addr[reachable] = seed_addrs[valid_pos]
        nearest_seed_kind[reachable] = seed_kinds[valid_pos]

    # Vectorized confidence score & label computation
    conf_score = np.full(n, np.nan, dtype=np.float32)
    conf_score[reachable] = np.round(
        1.0 / (1.0 + min_hops[reachable].astype(np.float32)), 4
    )

    conf_label = np.full(n, "unattributed", dtype=object)
    conf_label[reachable & (min_hops <= 2)] = "high"
    conf_label[reachable & (min_hops > 2) & (min_hops <= 4)] = "medium"
    conf_label[reachable & (min_hops > 4)] = "low"

    return pd.DataFrame(
        {
            "address": idx_to_address,
            "nearest_seed_id": nearest_seed_id,
            "nearest_seed_name": nearest_seed_name,
            "nearest_seed_address": nearest_seed_addr,
            "nearest_seed_kind": nearest_seed_kind,
            "hops": min_hops,
            "confidence": conf_label,
            "confidence_score": conf_score,
            "next_hop_idx": next_hop,
        }
    )


def reconstruct_path_to_seed(
    wallet_idx: int,
    seed_idx: int | None,
    idx_to_address: np.ndarray,
    next_hop_idx: np.ndarray | None = None,
    adj_undir: sp.csr_matrix | None = None,
    max_hops: int = 50,
) -> list[str]:
    """Reconstruct shortest path of address strings from wallet_idx to its nearest VASP seed.

    Uses `next_hop_idx` (O(hops) time, sub-millisecond) when available; otherwise falls
    back to on-demand `scipy.sparse.csgraph.breadth_first_order` from `wallet_idx`.

    Args:
        wallet_idx: Integer node index of the queried wallet.
        seed_idx: Integer node index of the target VASP seed (or None if unattributed).
        idx_to_address: 1D array mapping integer node index to address string.
        next_hop_idx: Optional 1D int32 array mapping each node to its next hop toward its nearest seed.
        adj_undir: Optional undirected CSR graph (used if next_hop_idx is None).
        max_hops: Safety limit on path length.

    Returns:
        List of address strings `[wallet_address, ..., seed_address]`, or `[]` if unreachable.
    """
    if seed_idx is None or seed_idx < 0 or wallet_idx < 0:
        return []
    if wallet_idx == seed_idx:
        return [str(idx_to_address[wallet_idx])]

    if next_hop_idx is not None and len(next_hop_idx) > wallet_idx:
        path_indices = [int(wallet_idx)]
        curr = int(wallet_idx)
        for _ in range(max_hops):
            nxt = int(next_hop_idx[curr])
            if nxt < 0:
                return []
            path_indices.append(nxt)
            if nxt == int(seed_idx):
                return [str(idx_to_address[i]) for i in path_indices]
            curr = nxt
        return []

    if adj_undir is not None:
        _, predecessors = breadth_first_order(
            adj_undir, i_start=int(wallet_idx), directed=False, return_predecessors=True
        )
        if int(seed_idx) >= len(predecessors) or predecessors[int(seed_idx)] < 0:
            return []
        rev_indices = [int(seed_idx)]
        curr = int(seed_idx)
        for _ in range(max_hops):
            pred = int(predecessors[curr])
            if pred < 0:
                break
            rev_indices.append(pred)
            if pred == int(wallet_idx):
                rev_indices.reverse()
                return [str(idx_to_address[i]) for i in rev_indices]
            curr = pred

    return []


def build_trace_subgraph(
    wallet_address: str,
    address_to_idx: dict[str, int],
    idx_to_address: np.ndarray,
    adj_directed: sp.csr_matrix,
    attribution_df: pd.DataFrame,
    max_neighbors: int = 12,
    max_total_nodes: int = 60,
) -> dict[str, Any]:
    """Build the /api/trace/{address} response payload on demand.

    SPEC section 5:
    GET /api/trace/{address} returns:
      {"path": [str], "nodes": [{"id": str, "kind": "wallet" | "path" | "vasp" | "neighbor"}],
       "edges": [{"source": str, "target": str}]}
    At most 60 nodes: the path plus up to 12 neighbours of the wallet.

    Args:
        wallet_address: Queried Bitcoin address string.
        address_to_idx: Mapping from address string to integer node index.
        idx_to_address: Mapping from integer node index to address string.
        adj_directed: Directed CSR adjacency matrix (for true edge directions and neighbors).
        attribution_df: Attribution DataFrame indexed by node_idx (0..N-1).
        max_neighbors: Maximum number of direct neighbors of the wallet to include (default 12).
        max_total_nodes: Maximum total nodes in the returned subgraph (default 60).

    Returns:
        Dictionary matching the /api/trace/{address} JSON contract.
    """
    if wallet_address not in address_to_idx:
        raise KeyError(f"Unknown address: {wallet_address}")

    wallet_idx = int(address_to_idx[wallet_address])
    row = attribution_df.iloc[wallet_idx]
    hops = int(row["hops"])
    seed_addr = row["nearest_seed_address"]
    seed_idx = (
        int(address_to_idx[str(seed_addr)])
        if hops >= 0 and pd.notna(seed_addr) and str(seed_addr) in address_to_idx
        else None
    )

    next_hop_arr = (
        attribution_df["next_hop_idx"].to_numpy(dtype=np.int32)
        if "next_hop_idx" in attribution_df.columns
        else None
    )

    path_addrs = reconstruct_path_to_seed(
        wallet_idx=wallet_idx,
        seed_idx=seed_idx,
        idx_to_address=idx_to_address,
        next_hop_idx=next_hop_arr,
    )

    # Bound path length if needed so total nodes <= max_total_nodes
    if len(path_addrs) > max_total_nodes:
        path_addrs = path_addrs[:max_total_nodes]

    nodes_map: dict[str, str] = {}
    if path_addrs:
        for i, addr in enumerate(path_addrs):
            if i == 0:
                nodes_map[addr] = "wallet"
            elif i == len(path_addrs) - 1 and pd.notna(seed_addr) and addr == str(seed_addr):
                nodes_map[addr] = "vasp"
            else:
                nodes_map[addr] = "path"
        if pd.notna(seed_addr) and str(seed_addr) == wallet_address:
            # Wallet itself is a VASP seed; keep kind="wallet" for the queried node
            nodes_map[wallet_address] = "wallet"
    else:
        nodes_map[wallet_address] = "wallet"

    # Find up to `max_neighbors` direct neighbors (outgoing + incoming) of wallet_idx
    out_neighbors = adj_directed.indices[
        adj_directed.indptr[wallet_idx] : adj_directed.indptr[wallet_idx + 1]
    ]
    adj_csc = adj_directed.tocsc() if not sp.isspmatrix_csc(adj_directed) else adj_directed
    in_neighbors = adj_csc.indices[
        adj_csc.indptr[wallet_idx] : adj_csc.indptr[wallet_idx + 1]
    ]

    candidate_neighbors: list[int] = []
    seen_neighbor_idxs: set[int] = {wallet_idx}
    for nb_idx in np.concatenate([out_neighbors, in_neighbors]):
        nb_int = int(nb_idx)
        if nb_int not in seen_neighbor_idxs:
            seen_neighbor_idxs.add(nb_int)
            candidate_neighbors.append(nb_int)

    added_neighbors = 0
    for nb_int in candidate_neighbors:
        if added_neighbors >= max_neighbors or len(nodes_map) >= max_total_nodes:
            break
        nb_addr = str(idx_to_address[nb_int])
        if nb_addr not in nodes_map:
            nodes_map[nb_addr] = "neighbor"
            added_neighbors += 1

    # Collect edges among the selected subgraph nodes
    selected_addrs = list(nodes_map.keys())
    selected_idxs = [int(address_to_idx[a]) for a in selected_addrs]
    sub_adj = adj_directed[selected_idxs, :][:, selected_idxs].tocoo()

    edges_set: set[tuple[str, str]] = set()
    for r_local, c_local in zip(sub_adj.row, sub_adj.col):
        if r_local == c_local:
            continue
        u_addr = selected_addrs[int(r_local)]
        v_addr = selected_addrs[int(c_local)]
        edges_set.add((u_addr, v_addr))

    # Ensure consecutive path nodes have a visible connecting edge in the subgraph
    for i in range(len(path_addrs) - 1):
        u_addr = path_addrs[i]
        v_addr = path_addrs[i + 1]
        if (u_addr, v_addr) not in edges_set and (v_addr, u_addr) not in edges_set:
            edges_set.add((u_addr, v_addr))

    nodes_list = [{"id": addr, "kind": kind} for addr, kind in nodes_map.items()]
    edges_list = [
        {"source": u_addr, "target": v_addr}
        for u_addr, v_addr in sorted(edges_set)
    ]

    return {
        "path": path_addrs,
        "nodes": nodes_list,
        "edges": edges_list,
    }


def compute_attribution_stats_for_k(
    min_hops: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compute SPEC 4.3 attribution statistics for a given hop array and address label array.

    SPEC 4.3:
    - share of addresses reachable within 2, 4 and 6 hops;
    - median hops for illicit vs licit labelled addresses.

    Args:
        min_hops: 1D int32 array of hop counts (-1 for unreachable).
        labels: 1D int8 array aligned with min_hops (1=illicit, 0=licit, -1=unknown).

    Returns:
        Dictionary of attribution stats.
    """
    n = len(min_hops)
    if n == 0:
        return {
            "total_addresses": 0,
            "reachable_addresses": 0,
            "reachable_share": 0.0,
            "reachable_within_2_hops_share": 0.0,
            "reachable_within_4_hops_share": 0.0,
            "reachable_within_6_hops_share": 0.0,
            "median_hops_illicit": None,
            "median_hops_licit": None,
        }

    reachable = min_hops >= 0
    within_2 = reachable & (min_hops <= 2)
    within_4 = reachable & (min_hops <= 4)
    within_6 = reachable & (min_hops <= 6)

    illicit_reachable_hops = min_hops[reachable & (labels == 1)]
    licit_reachable_hops = min_hops[reachable & (labels == 0)]

    median_illicit = (
        float(np.median(illicit_reachable_hops))
        if len(illicit_reachable_hops) > 0
        else None
    )
    median_licit = (
        float(np.median(licit_reachable_hops))
        if len(licit_reachable_hops) > 0
        else None
    )

    return {
        "total_addresses": int(n),
        "reachable_addresses": int(reachable.sum()),
        "unattributed_addresses": int((~reachable).sum()),
        "reachable_share": round(float(reachable.sum() / n), 4),
        "reachable_within_2_hops_count": int(within_2.sum()),
        "reachable_within_2_hops_share": round(float(within_2.sum() / n), 4),
        "reachable_within_4_hops_count": int(within_4.sum()),
        "reachable_within_4_hops_share": round(float(within_4.sum() / n), 4),
        "reachable_within_6_hops_count": int(within_6.sum()),
        "reachable_within_6_hops_share": round(float(within_6.sum() / n), 4),
        "illicit_reachable_count": int(len(illicit_reachable_hops)),
        "licit_reachable_count": int(len(licit_reachable_hops)),
        "median_hops_illicit": median_illicit,
        "median_hops_licit": median_licit,
    }
