"""Data preparation and graph feature extraction pipeline.

Implements SPEC sections 4.1 and 4.2:
- Loads and cleans raw Elliptic++ Actors dataset files.
- Maps string address IDs to integer indices (0..822,941).
- Builds scipy sparse directed address graph.
- Computes unsupervised graph features: in-degree, out-degree, PageRank, mean neighbor degree.
- Performs temporal split (train: steps 1-34, test: steps 35-49).
- Conducts address leakage analysis between train and test periods.
- Saves all artifacts under artifacts/.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tracechain.data import (
    ADDRESS_COL,
    FEATURE_COLUMNS,
    TIME_STEP_COL,
    build_address_mapping,
    build_sparse_graph,
    check_leakage,
    load_and_clean_classes,
    load_and_clean_features,
    save_artifacts,
    split_temporal,
)
from tracechain.features import compute_all_graph_features


RAW_DIR = Path("data/raw")
ARTIFACTS_DIR = Path("artifacts")
REQUIRED_FILES = (
    "wallets_features.csv",
    "wallets_classes.csv",
    "AddrAddr_edgelist.csv",
)


def verify_raw_files(raw_dir: Path) -> list[str]:
    """Check that all required raw data files exist."""
    return [name for name in REQUIRED_FILES if not (raw_dir / name).is_file()]


def main() -> int:
    """Run data preparation, graph feature computation, and temporal split."""
    missing = verify_raw_files(RAW_DIR)
    if missing:
        print("ERROR: Missing required dataset files in data/raw/:")
        for m in missing:
            print(f"  - data/raw/{m}")
        print("\nPlease download the Elliptic++ Actors dataset from:")
        print("https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l?usp=sharing")
        print("and place them into data/raw/ before running this script.")
        return 1

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    start_time = time.perf_counter()

    print("=" * 70)
    print("TraceChain v2: Data Preparation & Graph Features (SPEC 4.1 & 4.2)")
    print("=" * 70)

    # 1. Load classes and build address mapping
    print("\n[1/6] Loading wallets_classes.csv and building address mapping...")
    classes_df = load_and_clean_classes(RAW_DIR / "wallets_classes.csv")
    print(f"  Loaded {len(classes_df):,} addresses.")

    address_to_idx, idx_to_address = build_address_mapping(classes_df[ADDRESS_COL])
    print(f"  Mapped {len(address_to_idx):,} unique addresses to indices 0..{len(address_to_idx)-1:,}.")

    # 2. Build sparse address graph
    print("\n[2/6] Building directed sparse graph from AddrAddr_edgelist.csv...")
    adj_matrix = build_sparse_graph(RAW_DIR / "AddrAddr_edgelist.csv", address_to_idx)
    print(f"  Graph shape: {adj_matrix.shape[0]:,} x {adj_matrix.shape[1]:,}, directed edges (nnz): {adj_matrix.nnz:,}")

    # Save graph artifacts
    save_artifacts(ARTIFACTS_DIR, address_to_idx, idx_to_address, adj_matrix)
    print(f"  Saved graph artifacts to {ARTIFACTS_DIR}/")

    # 3. Compute graph features
    print("\n[3/6] Computing unsupervised graph features (in/out-degree, PageRank, mean neighbor degree)...")
    graph_feats = compute_all_graph_features(adj_matrix, alpha=0.85, pagerank_iter=50)

    graph_feats_df = pd.DataFrame(
        {
            ADDRESS_COL: idx_to_address,
            "in_degree": graph_feats["in_degree"],
            "out_degree": graph_feats["out_degree"],
            "pagerank": graph_feats["pagerank"],
            "mean_neighbor_degree": graph_feats["mean_neighbor_degree"],
        }
    )
    graph_feats_path = ARTIFACTS_DIR / "graph_features.parquet"
    graph_feats_df.to_parquet(graph_feats_path, index=False)
    print(f"  Saved graph features to {graph_feats_path}")

    # 4. Load features and clean duplicates
    print("\n[4/6] Loading and deduplicating wallets_features.csv...")
    features_df = load_and_clean_features(RAW_DIR / "wallets_features.csv")
    print(f"  Features rows after deduplication: {len(features_df):,} (from distinct address-timestep pairs)")

    # 5. Merge labels and graph features
    print("\n[5/6] Merging labels and graph features...")
    # Merge class labels
    merged_df = features_df.merge(
        classes_df[[ADDRESS_COL, "label", "raw_class"]],
        on=ADDRESS_COL,
        how="inner",
    )
    # Merge graph features
    merged_df = merged_df.merge(graph_feats_df, on=ADDRESS_COL, how="inner")
    print(f"  Merged dataset rows: {len(merged_df):,}, columns: {len(merged_df.columns)}")

    # 6. Temporal split and leakage check
    print("\n[6/6] Performing temporal split and leakage check...")
    train_df, test_df = split_temporal(merged_df, train_range=(1, 34), test_range=(35, 49))
    leakage_metrics, cold_test_df = check_leakage(train_df, test_df)

    # Save dataset splits
    train_df.to_parquet(ARTIFACTS_DIR / "train.parquet", index=False)
    test_df.to_parquet(ARTIFACTS_DIR / "test.parquet", index=False)
    cold_test_df.to_parquet(ARTIFACTS_DIR / "cold_test.parquet", index=False)

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_memory / (1024 * 1024)

    # Class distribution
    train_illicit = int((train_df["label"] == 1).sum())
    train_licit = int((train_df["label"] == 0).sum())
    test_illicit = int((test_df["label"] == 1).sum())
    test_licit = int((test_df["label"] == 0).sum())
    cold_illicit = int((cold_test_df["label"] == 1).sum())
    cold_licit = int((cold_test_df["label"] == 0).sum())

    summary: dict[str, object] = {
        "run_time_seconds": round(elapsed_time, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "total_addresses": len(address_to_idx),
        "total_edges": adj_matrix.nnz,
        "splits": {
            "train": {
                "timesteps": "1 to 34",
                "rows": len(train_df),
                "unique_addresses": leakage_metrics["train_unique_addresses"],
                "illicit_count": train_illicit,
                "licit_count": train_licit,
                "illicit_share": round(train_illicit / len(train_df), 4) if len(train_df) else 0.0,
            },
            "test_all": {
                "timesteps": "35 to 49",
                "rows": len(test_df),
                "unique_addresses": leakage_metrics["test_unique_addresses"],
                "illicit_count": test_illicit,
                "licit_count": test_licit,
                "illicit_share": round(test_illicit / len(test_df), 4) if len(test_df) else 0.0,
            },
            "test_cold": {
                "timesteps": "35 to 49 (never in train)",
                "rows": len(cold_test_df),
                "unique_addresses": leakage_metrics["cold_test_addresses"],
                "illicit_count": cold_illicit,
                "licit_count": cold_licit,
                "illicit_share": round(cold_illicit / len(cold_test_df), 4) if len(cold_test_df) else 0.0,
            },
        },
        "leakage": {
            "test_addresses_total": leakage_metrics["test_unique_addresses"],
            "overlap_with_train_addresses": leakage_metrics["overlap_addresses"],
            "overlap_address_share": leakage_metrics["overlap_address_share"],
            "cold_test_addresses": leakage_metrics["cold_test_addresses"],
            "cold_address_share": leakage_metrics["cold_address_share"],
        },
    }

    metadata_path = ARTIFACTS_DIR / "prepared_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("PREPARATION SUMMARY")
    print("=" * 70)
    print(f"Run time:    {elapsed_time:.2f} seconds")
    print(f"Peak memory: {peak_memory_mb:.2f} MB")
    print("\n--- Row Counts and Class Counts per Split ---")
    print(f"Train (steps 1-34):      {len(train_df):>7,} rows | illicit: {train_illicit:>5,} | licit: {train_licit:>7,}")
    print(f"Test (all, steps 35-49): {len(test_df):>7,} rows | illicit: {test_illicit:>5,} | licit: {test_licit:>7,}")
    print(f"Test (cold subset):      {len(cold_test_df):>7,} rows | illicit: {cold_illicit:>5,} | licit: {cold_licit:>7,}")
    print("\n--- Leakage Check Numbers ---")
    print(f"Test unique addresses:             {leakage_metrics['test_unique_addresses']:,}")
    print(f"Addresses seen in training period: {leakage_metrics['overlap_addresses']:,} ({leakage_metrics['overlap_address_share']*100:.2f}%)")
    print(f"Cold test addresses (never seen):  {leakage_metrics['cold_test_addresses']:,} ({leakage_metrics['cold_address_share']*100:.2f}%)")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
