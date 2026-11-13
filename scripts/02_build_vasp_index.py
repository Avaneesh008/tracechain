"""Build VASP proxy/registry seed attribution index and reports/attribution_stats.json.

Implements SPEC section 4.3:
- Loads address graph and address mapping from artifacts/.
- Loads in-degree and out-degree from artifacts/graph_features.parquet, plus active
  time steps and total BTC moved from data/raw/wallets_features.csv.
- Identifies training-period illicit addresses from artifacts/train.parquet (so test
  period labels are never leaked into seed selection).
- Supports optional registry override at data/registry/vasp_registry.csv.
- Runs chunked multi-source BFS on the undirected address graph for K=10, K=30, and K=100.
- Saves the K=30 (or registry) VASP attribution index under artifacts/vasp_index.parquet
  and seed metadata under artifacts/vasp_seeds.parquet.
- Benchmarks single-address path reconstruction and trace subgraph generation.
- Saves reports/attribution_stats.json with reachability and median hop metrics for K=10, 30, 100.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from tracechain.data import (
    ADDRESS_COL,
    TIME_STEP_COL,
    load_and_clean_classes,
    load_artifacts,
)
from tracechain.vasp import (
    build_attribution_dataframe,
    build_trace_subgraph,
    build_undirected_graph,
    compute_attribution_stats_for_k,
    compute_graph_edge_counts,
    compute_seed_scores,
    load_or_select_seeds,
    reconstruct_path_to_seed,
    run_chunked_multisource_bfs,
    select_proxy_seeds,
)


RAW_DIR = Path("data/raw")
ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
REGISTRY_PATH = Path("data/registry/vasp_registry.csv")
DEFAULT_K = 30
SENSITIVITY_KS = (10, 30, 100)
BFS_CHUNK_SIZE = 10


def _load_per_address_activity_and_volume(
    features_csv: Path,
    address_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load active time steps and total BTC moved per address aligned to 0..N-1.

    Uses `num_timesteps_appeared_in` and `btc_transacted_total` from wallets_features.csv,
    which are static wallet-level summary columns in Elliptic++ Actors.
    """
    n = len(address_to_idx)
    df = pd.read_csv(
        features_csv,
        usecols=[ADDRESS_COL, "num_timesteps_appeared_in", "btc_transacted_total"],
        dtype={
            ADDRESS_COL: str,
            "num_timesteps_appeared_in": np.float32,
            "btc_transacted_total": np.float32,
        },
    )
    df = df.drop_duplicates(subset=[ADDRESS_COL], keep="first")

    active_steps = np.zeros(n, dtype=np.float32)
    btc_total = np.zeros(n, dtype=np.float32)

    idxs = df[ADDRESS_COL].map(address_to_idx).to_numpy(dtype=np.int32)
    active_steps[idxs] = df["num_timesteps_appeared_in"].to_numpy(dtype=np.float32)
    btc_total[idxs] = df["btc_transacted_total"].to_numpy(dtype=np.float32)
    return active_steps, btc_total


def main() -> int:
    """Run VASP seed selection, chunked BFS attribution, and stats generation."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    t_start = time.perf_counter()

    print("=" * 70)
    print("TraceChain v2: VASP Seeds & Attribution Index (SPEC 4.3)")
    print("=" * 70)

    # 1. Load graph artifacts and degree features
    print("\n[1/6] Loading sparse graph and address mapping from artifacts/...")
    address_to_idx, idx_to_address, adj_directed = load_artifacts(ARTIFACTS_DIR)
    n_nodes = len(idx_to_address)
    print(
        f"  Loaded {n_nodes:,} nodes and {adj_directed.nnz:,} directed edges."
    )

    graph_feats_df = pd.read_parquet(ARTIFACTS_DIR / "graph_features.parquet")
    in_degree = graph_feats_df["in_degree"].to_numpy(dtype=np.float32)
    out_degree = graph_feats_df["out_degree"].to_numpy(dtype=np.float32)

    # 2. Load per-address active time steps and total BTC moved
    print("\n[2/6] Loading per-address active time steps and total BTC moved...")
    active_steps, btc_total = _load_per_address_activity_and_volume(
        RAW_DIR / "wallets_features.csv", address_to_idx
    )

    # 3. Load labels and training-period illicit addresses
    print("\n[3/6] Loading class labels and training-period illicit exclusions...")
    classes_df = load_and_clean_classes(RAW_DIR / "wallets_classes.csv")
    labels_by_idx = np.full(n_nodes, -1, dtype=np.int8)
    c_idxs = classes_df[ADDRESS_COL].map(address_to_idx).to_numpy(dtype=np.int32)
    labels_by_idx[c_idxs] = classes_df["label"].to_numpy(dtype=np.int8)

    train_df = pd.read_parquet(
        ARTIFACTS_DIR / "train.parquet", columns=[ADDRESS_COL, "label"]
    )
    train_illicit_addrs = set(
        train_df.loc[train_df["label"] == 1, ADDRESS_COL].unique()
    )
    print(
        f"  Training-period illicit addresses excluded from proxy seed eligibility: "
        f"{len(train_illicit_addrs):,}"
    )

    # 4. Compute seed scores and select top 100 proxy seeds (covers K=10, 30, 100)
    print("\n[4/6] Computing VASP proxy seed scores (rank-average of 5 structural metrics)...")
    seed_scores = compute_seed_scores(
        in_degree=in_degree,
        out_degree=out_degree,
        active_time_steps=active_steps,
        total_value_moved=btc_total,
    )

    max_k = max(SENSITIVITY_KS)
    top_100_proxy_seeds = select_proxy_seeds(
        idx_to_address=idx_to_address,
        seed_scores=seed_scores,
        train_illicit_addresses=train_illicit_addrs,
        k=max_k,
    )
    # Standardize IDs for K<=30 so VASP-PROXY-01..VASP-PROXY-30 have 2-digit zero-padded names
    top_100_proxy_seeds["seed_id"] = [
        f"VASP-PROXY-{i + 1:02d}" for i in range(len(top_100_proxy_seeds))
    ]
    top_100_proxy_seeds["name"] = top_100_proxy_seeds["seed_id"]

    # Check if optional registry exists
    registry_used = REGISTRY_PATH.is_file()
    if registry_used:
        active_seeds_df = load_or_select_seeds(
            idx_to_address=idx_to_address,
            address_to_idx=address_to_idx,
            seed_scores=seed_scores,
            train_illicit_addresses=train_illicit_addrs,
            k=DEFAULT_K,
            registry_path=REGISTRY_PATH,
        )
        print(
            f"  Registry file found at {REGISTRY_PATH}: loaded {len(active_seeds_df)} registry seeds."
        )
    else:
        active_seeds_df = top_100_proxy_seeds.iloc[:DEFAULT_K].copy()
        print(
            f"  No registry file at {REGISTRY_PATH}; selected top K={DEFAULT_K} proxy seeds "
            f"(VASP-PROXY-01 .. VASP-PROXY-{DEFAULT_K:02d})."
        )

    # 5. Build undirected graph and run chunked BFS across top 100 proxy seeds
    print(
        f"\n[5/6] Building undirected graph and running chunked BFS (chunk_size={BFS_CHUNK_SIZE})..."
    )
    adj_undir = build_undirected_graph(adj_directed)
    graph_counts = compute_graph_edge_counts(adj_directed, adj_undir)
    print(
        f"  Directed edges total (incl. self-loops): {graph_counts['directed_edges_total']:,}"
    )
    print(
        f"  Directed non-self-loop edges:            {graph_counts['directed_non_self_loop_edges']:,} "
        f"(self-loops: {graph_counts['directed_self_loops']:,})"
    )
    print(
        f"  Distinct undirected edges (u != v):      {graph_counts['distinct_undirected_edges']:,} "
        f"(reciprocal pairs: {graph_counts['reciprocal_directed_edge_pairs']:,})"
    )
    print(
        f"  Symmetrised sparse matrix nonzeros:      {graph_counts['symmetrised_matrix_nonzeros']:,} "
        f"(2 x distinct_undirected_edges)"
    )

    # Run incremental chunked BFS over the 100 proxy seeds in chunks of 10 so we capture K=10, K=30, K=100
    # in a single pass of 10 chunks (100 seeds total)!
    seed_indices_100 = top_100_proxy_seeds["node_idx"].astype(int).tolist()
    best_dist = np.full(n_nodes, np.inf, dtype=np.float64)
    nearest_seed_pos = np.full(n_nodes, -1, dtype=np.int32)
    next_hop_idx = np.full(n_nodes, -1, dtype=np.int32)

    stats_by_k: dict[str, dict[str, Any]] = {}
    bfs_k30_snapshot: dict[str, np.ndarray] = {}

    for chunk_start in range(0, max_k, BFS_CHUNK_SIZE):
        chunk_end = min(chunk_start + BFS_CHUNK_SIZE, max_k)
        chunk_seeds = seed_indices_100[chunk_start:chunk_end]
        t_chunk = time.perf_counter()
        chunk_res = run_chunked_multisource_bfs(
            adj_undir=adj_undir,
            seed_indices=chunk_seeds,
            chunk_size=BFS_CHUNK_SIZE,
            return_predecessors=True,
        )
        c_hops = chunk_res["min_hops"]
        c_pos = chunk_res["nearest_seed_pos"]
        c_next = chunk_res["next_hop_idx"]

        c_dist = np.where(c_hops >= 0, c_hops.astype(np.float64), np.inf)
        improved = c_dist < best_dist
        best_dist[improved] = c_dist[improved]
        nearest_seed_pos[improved] = chunk_start + c_pos[improved]
        next_hop_idx[improved] = c_next[improved]

        print(
            f"  Processed seeds {chunk_start + 1:>3}..{chunk_end:>3} in {time.perf_counter() - t_chunk:.2f}s"
        )

        if chunk_end in SENSITIVITY_KS:
            reachable_mask = np.isfinite(best_dist)
            current_min_hops = np.full(n_nodes, -1, dtype=np.int32)
            current_min_hops[reachable_mask] = best_dist[reachable_mask].astype(
                np.int32
            )
            stats_by_k[f"K={chunk_end}"] = compute_attribution_stats_for_k(
                current_min_hops, labels_by_idx
            )
            if chunk_end == DEFAULT_K:
                bfs_k30_snapshot = {
                    "min_hops": current_min_hops.copy(),
                    "nearest_seed_pos": nearest_seed_pos.copy(),
                    "next_hop_idx": next_hop_idx.copy(),
                }

    # If a custom registry file was present, run chunked BFS specifically for the registry seeds
    if registry_used:
        reg_bfs = run_chunked_multisource_bfs(
            adj_undir=adj_undir,
            seed_indices=active_seeds_df["node_idx"].astype(int).tolist(),
            chunk_size=BFS_CHUNK_SIZE,
            return_predecessors=True,
        )
        attribution_df = build_attribution_dataframe(
            idx_to_address=idx_to_address,
            seeds_df=active_seeds_df,
            bfs_result=reg_bfs,
        )
        active_stats = compute_attribution_stats_for_k(
            reg_bfs["min_hops"], labels_by_idx
        )
    else:
        attribution_df = build_attribution_dataframe(
            idx_to_address=idx_to_address,
            seeds_df=active_seeds_df,
            bfs_result=bfs_k30_snapshot,
        )
        active_stats = stats_by_k[f"K={DEFAULT_K}"]

    # Save VASP attribution artifacts
    print("\n[6/6] Saving VASP attribution index and benchmarking path reconstruction...")
    vasp_index_path = ARTIFACTS_DIR / "vasp_index.parquet"
    vasp_seeds_path = ARTIFACTS_DIR / "vasp_seeds.parquet"
    attribution_df.to_parquet(vasp_index_path, index=False)
    active_seeds_df.to_parquet(vasp_seeds_path, index=False)
    print(f"  Saved {vasp_index_path} ({len(attribution_df):,} rows)")
    print(f"  Saved {vasp_seeds_path} ({len(active_seeds_df):,} seeds)")

    # Benchmark path reconstruction and full /api/trace subgraph construction on a real 3+ hop wallet
    next_hop_arr = attribution_df["next_hop_idx"].to_numpy(dtype=np.int32)
    hops_arr = attribution_df["hops"].to_numpy(dtype=np.int32)
    sample_candidates = np.where((hops_arr >= 3) & (hops_arr <= 6))[0]
    sample_wallet_idx = (
        int(sample_candidates[0])
        if len(sample_candidates) > 0
        else int(np.where(hops_arr > 0)[0][0])
    )
    sample_wallet_addr = str(idx_to_address[sample_wallet_idx])
    sample_seed_addr = str(attribution_df.loc[sample_wallet_idx, "nearest_seed_address"])
    sample_seed_idx = int(address_to_idx[sample_seed_addr])

    # Time pure path reconstruction
    t_path_0 = time.perf_counter()
    sample_path = reconstruct_path_to_seed(
        wallet_idx=sample_wallet_idx,
        seed_idx=sample_seed_idx,
        idx_to_address=idx_to_address,
        next_hop_idx=next_hop_arr,
    )
    path_recon_ms = (time.perf_counter() - t_path_0) * 1000.0

    # Time on-demand BFS path reconstruction (without precomputed next_hop_idx)
    t_bfs_path_0 = time.perf_counter()
    sample_path_bfs = reconstruct_path_to_seed(
        wallet_idx=sample_wallet_idx,
        seed_idx=sample_seed_idx,
        idx_to_address=idx_to_address,
        next_hop_idx=None,
        adj_undir=adj_undir,
    )
    bfs_path_recon_ms = (time.perf_counter() - t_bfs_path_0) * 1000.0

    # Time full /api/trace subgraph construction (path + up to 12 neighbors + edges)
    t_trace_0 = time.perf_counter()
    sample_trace = build_trace_subgraph(
        wallet_address=sample_wallet_addr,
        address_to_idx=address_to_idx,
        idx_to_address=idx_to_address,
        adj_directed=adj_directed,
        attribution_df=attribution_df,
    )
    trace_subgraph_ms = (time.perf_counter() - t_trace_0) * 1000.0

    elapsed_s = time.perf_counter() - t_start
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)

    # Build reports/attribution_stats.json
    report_payload: dict[str, Any] = {
        "seed_kind": "registry" if registry_used else "proxy",
        "default_k": DEFAULT_K,
        "registry_override_active": registry_used,
        "run_time_seconds": round(elapsed_s, 2),
        "peak_memory_mb": round(peak_mem_mb, 2),
        "graph": graph_counts,
        "path_reconstruction_benchmark": {
            "sample_wallet_address": sample_wallet_addr,
            "sample_nearest_seed_id": str(
                attribution_df.loc[sample_wallet_idx, "nearest_seed_id"]
            ),
            "sample_hops": int(hops_arr[sample_wallet_idx]),
            "path_length_nodes": len(sample_path),
            "index_path_reconstruction_ms": round(path_recon_ms, 4),
            "on_demand_bfs_path_reconstruction_ms": round(bfs_path_recon_ms, 2),
            "full_trace_subgraph_ms": round(trace_subgraph_ms, 3),
            "trace_subgraph_nodes": len(sample_trace["nodes"]),
            "trace_subgraph_edges": len(sample_trace["edges"]),
        },
        "active_index_stats": active_stats,
        "sensitivity_by_k": {
            "10": stats_by_k["K=10"],
            "30": stats_by_k["K=30"],
            "100": stats_by_k["K=100"],
        },
        "top_10_proxy_seeds": [
            {
                "seed_id": str(row["seed_id"]),
                "address": str(row["address"]),
                "kind": str(row["kind"]),
                "seed_score": round(float(row["seed_score"]), 6),
                "in_degree": int(in_degree[int(row["node_idx"])]),
                "out_degree": int(out_degree[int(row["node_idx"])]),
                "active_time_steps": int(active_steps[int(row["node_idx"])]),
                "btc_transacted_total": round(
                    float(btc_total[int(row["node_idx"])]), 4
                ),
            }
            for _, row in top_100_proxy_seeds.head(10).iterrows()
        ],
    }

    report_path = REPORTS_DIR / "attribution_stats.json"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("VASP ATTRIBUTION SUMMARY")
    print("=" * 70)
    print(f"Total run time:                     {elapsed_s:.2f} s")
    print(f"Peak memory (tracemalloc):          {peak_mem_mb:.2f} MB")
    print(
        f"One path reconstruction (indexed):  {path_recon_ms:.4f} ms "
        f"(hops={hops_arr[sample_wallet_idx]}, nodes={len(sample_path)})"
    )
    print(
        f"One path reconstruction (raw BFS):  {bfs_path_recon_ms:.2f} ms "
        f"(matches indexed path: {sample_path == sample_path_bfs})"
    )
    print(
        f"Full /api/trace subgraph build:     {trace_subgraph_ms:.3f} ms "
        f"({len(sample_trace['nodes'])} nodes, {len(sample_trace['edges'])} edges)"
    )
    print("\n--- Sensitivity Across K = 10, 30, 100 ---")
    for k_val in SENSITIVITY_KS:
        st = stats_by_k[f"K={k_val}"]
        print(
            f"K={k_val:<3} | Reachable: {st['reachable_share']*100:6.2f}% "
            f"| <=2 hops: {st['reachable_within_2_hops_share']*100:6.2f}% "
            f"| <=4 hops: {st['reachable_within_4_hops_share']*100:6.2f}% "
            f"| <=6 hops: {st['reachable_within_6_hops_share']*100:6.2f}% "
            f"| median hops (illicit={st['median_hops_illicit']}, licit={st['median_hops_licit']})"
        )
    print(f"\nSaved attribution stats to {report_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
