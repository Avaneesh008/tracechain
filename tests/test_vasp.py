"""Unit tests for VASP proxy seed selection, attribution, and path reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tracechain.vasp import (
    UNATTRIBUTED_HOPS,
    build_attribution_dataframe,
    build_trace_subgraph,
    build_undirected_graph,
    compute_attribution_stats_for_k,
    compute_confidence,
    compute_graph_edge_counts,
    compute_seed_scores,
    load_or_select_seeds,
    reconstruct_path_to_seed,
    run_chunked_multisource_bfs,
    select_proxy_seeds,
)


def _make_synthetic_graph() -> tuple[dict[str, int], np.ndarray, sp.csr_matrix]:
    """Create a 7-node synthetic graph with two connected components and a self-loop.

    Component 1:
      0 (addr0) -> 1 (addr1) -> 2 (addr2) -> 3 (addr3)
      0 (addr0) -> 4 (addr4)
      (plus self-loop 2 -> 2)
    Component 2 (disconnected from 0..4):
      5 (addr5) -> 6 (addr6)
    """
    addrs = [f"addr{i}" for i in range(7)]
    address_to_idx = {a: i for i, a in enumerate(addrs)}
    idx_to_address = np.array(addrs, dtype=object)

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 4),
        (2, 2),  # self-loop
        (5, 6),  # disconnected component
    ]
    rows = [u for u, _ in edges]
    cols = [v for _, v in edges]
    data = np.ones(len(edges), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(7, 7), dtype=np.float32).tocsr()
    return address_to_idx, idx_to_address, adj


def test_seed_selection_and_illicit_exclusion() -> None:
    """Verify proxy seeds rank high-activity nodes first and skip training illicit nodes."""
    _, idx_to_address, _ = _make_synthetic_graph()
    in_deg = np.array([10.0, 50.0, 5.0, 1.0, 2.0, 0.0, 0.0], dtype=np.float32)
    out_deg = np.array([20.0, 60.0, 5.0, 1.0, 2.0, 0.0, 0.0], dtype=np.float32)
    active_steps = np.array([15.0, 30.0, 8.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    btc_total = np.array([100.0, 500.0, 25.0, 1.0, 2.0, 0.1, 0.1], dtype=np.float32)

    scores = compute_seed_scores(in_deg, out_deg, active_steps, btc_total)
    assert len(scores) == 7
    # addr1 has highest values across all metrics, addr0 is second, addr2 is third
    assert int(np.argmax(scores)) == 1

    # Mark addr1 as illicit in training period -> should be skipped!
    seeds_df = select_proxy_seeds(
        idx_to_address=idx_to_address,
        seed_scores=scores,
        train_illicit_addresses={"addr1"},
        k=2,
    )
    assert len(seeds_df) == 2
    assert list(seeds_df["address"]) == ["addr0", "addr2"]
    assert list(seeds_df["seed_id"]) == ["VASP-PROXY-01", "VASP-PROXY-02"]
    assert list(seeds_df["kind"]) == ["proxy", "proxy"]


def test_multisource_bfs_hops_and_unreachable() -> None:
    """Verify nearest seed, hop counts, confidence levels, and unreachable nodes."""
    address_to_idx, idx_to_address, adj = _make_synthetic_graph()
    adj_undir = build_undirected_graph(adj)

    # Suppose seeds are addr0 (node 0, VASP-PROXY-01) and addr3 (node 3, VASP-PROXY-02)
    seeds_df = pd.DataFrame(
        [
            {
                "seed_id": "VASP-PROXY-01",
                "name": "VASP-PROXY-01",
                "address": "addr0",
                "node_idx": 0,
                "kind": "proxy",
                "seed_score": 0.9,
            },
            {
                "seed_id": "VASP-PROXY-02",
                "name": "VASP-PROXY-02",
                "address": "addr3",
                "node_idx": 3,
                "kind": "proxy",
                "seed_score": 0.8,
            },
        ]
    )

    bfs_res = run_chunked_multisource_bfs(
        adj_undir=adj_undir,
        seed_indices=[0, 3],
        chunk_size=1,  # test chunking across seeds
        return_predecessors=True,
    )
    attr_df = build_attribution_dataframe(idx_to_address, seeds_df, bfs_res)

    # Expected hops on undirected graph:
    # addr0: 0 hops to VASP-PROXY-01
    # addr1: 1 hop to VASP-PROXY-01 (dist 1 vs dist 2 to addr3)
    # addr2: 1 hop to VASP-PROXY-02 (dist 1 to addr3 vs dist 2 to addr0)
    # addr3: 0 hops to VASP-PROXY-02
    # addr4: 1 hop to VASP-PROXY-01
    # addr5, addr6: unreachable (-1)
    expected_hops = [0, 1, 1, 0, 1, UNATTRIBUTED_HOPS, UNATTRIBUTED_HOPS]
    assert list(attr_df["hops"]) == expected_hops

    assert attr_df.loc[0, "nearest_seed_id"] == "VASP-PROXY-01"
    assert attr_df.loc[1, "nearest_seed_id"] == "VASP-PROXY-01"
    assert attr_df.loc[2, "nearest_seed_id"] == "VASP-PROXY-02"
    assert attr_df.loc[3, "nearest_seed_id"] == "VASP-PROXY-02"
    assert attr_df.loc[4, "nearest_seed_id"] == "VASP-PROXY-01"
    assert pd.isna(attr_df.loc[5, "nearest_seed_id"])
    assert pd.isna(attr_df.loc[6, "nearest_seed_id"])

    assert attr_df.loc[0, "confidence"] == "high"
    assert attr_df.loc[0, "confidence_score"] == 1.0
    assert attr_df.loc[1, "confidence"] == "high"
    assert np.isclose(attr_df.loc[1, "confidence_score"], 0.5)
    assert attr_df.loc[5, "confidence"] == "unattributed"
    assert np.isnan(attr_df.loc[5, "confidence_score"])

    # Path reconstruction checks
    next_hop = attr_df["next_hop_idx"].to_numpy(dtype=np.int32)
    # Single seed at node 0 to test 3-hop path from node 3 -> 2 -> 1 -> 0
    bfs_single = run_chunked_multisource_bfs(adj_undir, [0], chunk_size=1)
    path_3_to_0 = reconstruct_path_to_seed(
        wallet_idx=3,
        seed_idx=0,
        idx_to_address=idx_to_address,
        next_hop_idx=bfs_single["next_hop_idx"],
    )
    assert path_3_to_0 == ["addr3", "addr2", "addr1", "addr0"]

    # Unreachable node 5 returns empty path
    path_unreachable = reconstruct_path_to_seed(
        wallet_idx=5,
        seed_idx=None,
        idx_to_address=idx_to_address,
        next_hop_idx=next_hop,
    )
    assert path_unreachable == []

    # Trace subgraph check
    single_seed_df = seeds_df.iloc[[0]].copy()
    single_attr_df = build_attribution_dataframe(idx_to_address, single_seed_df, bfs_single)
    trace = build_trace_subgraph(
        wallet_address="addr3",
        address_to_idx=address_to_idx,
        idx_to_address=idx_to_address,
        adj_directed=adj,
        attribution_df=single_attr_df,
    )
    assert trace["path"] == ["addr3", "addr2", "addr1", "addr0"]
    kinds = {n["id"]: n["kind"] for n in trace["nodes"]}
    assert kinds["addr3"] == "wallet"
    assert kinds["addr2"] == "path"
    assert kinds["addr1"] == "path"
    assert kinds["addr0"] == "vasp"


def test_registry_override(tmp_path: Path) -> None:
    """Verify that data/registry/vasp_registry.csv overrides proxy seeds with kind='registry'."""
    address_to_idx, idx_to_address, adj = _make_synthetic_graph()
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.2, 0.1], dtype=np.float32)

    reg_csv = tmp_path / "vasp_registry.csv"
    reg_csv.write_text(
        "address,name\naddr5,Exchange-Alpha\naddr3,Exchange-Beta\n",
        encoding="utf-8",
    )

    seeds_df = load_or_select_seeds(
        idx_to_address=idx_to_address,
        address_to_idx=address_to_idx,
        seed_scores=scores,
        train_illicit_addresses=set(),
        k=3,
        registry_path=reg_csv,
    )

    assert len(seeds_df) == 2
    assert list(seeds_df["kind"]) == ["registry", "registry"]
    assert list(seeds_df["name"]) == ["Exchange-Alpha", "Exchange-Beta"]
    assert list(seeds_df["address"]) == ["addr5", "addr3"]

    adj_undir = build_undirected_graph(adj)
    bfs_res = run_chunked_multisource_bfs(
        adj_undir, seeds_df["node_idx"].tolist(), chunk_size=2
    )
    attr_df = build_attribution_dataframe(idx_to_address, seeds_df, bfs_res)

    # Now addr6 is 1 hop away from Exchange-Alpha (addr5)!
    assert attr_df.loc[6, "hops"] == 1
    assert attr_df.loc[6, "nearest_seed_name"] == "Exchange-Alpha"
    assert attr_df.loc[6, "nearest_seed_kind"] == "registry"


def test_confidence_and_stats_helpers() -> None:
    """Verify confidence boundaries and attribution stats calculation."""
    assert compute_confidence(0) == (1.0, "high")
    assert compute_confidence(2) == (round(1.0 / 3.0, 4), "high")
    assert compute_confidence(3) == (0.25, "medium")
    assert compute_confidence(4) == (0.2, "medium")
    assert compute_confidence(5) == (round(1.0 / 6.0, 4), "low")
    assert compute_confidence(-1) == (None, "unattributed")
    assert compute_confidence(None) == (None, "unattributed")

    hops = np.array([0, 1, 2, 3, 5, -1], dtype=np.int32)
    labels = np.array([0, 0, 1, 1, 1, -1], dtype=np.int8)
    stats = compute_attribution_stats_for_k(hops, labels)
    assert stats["total_addresses"] == 6
    assert stats["reachable_addresses"] == 5
    assert stats["reachable_within_2_hops_count"] == 3
    assert stats["reachable_within_4_hops_count"] == 4
    assert stats["reachable_within_6_hops_count"] == 5
    assert stats["median_hops_illicit"] == 3.0
    assert stats["median_hops_licit"] == 0.5

    _, _, adj = _make_synthetic_graph()
    edge_counts = compute_graph_edge_counts(adj)
    assert edge_counts["nodes"] == 7
    assert edge_counts["directed_edges_total"] == 6
    assert edge_counts["directed_self_loops"] == 1
    assert edge_counts["directed_non_self_loop_edges"] == 5
    assert edge_counts["distinct_undirected_edges"] == 5
    assert edge_counts["reciprocal_directed_edge_pairs"] == 0
    assert edge_counts["symmetrised_matrix_nonzeros"] == 10

