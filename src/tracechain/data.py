"""Data loading, cleaning, mapping, and temporal split utilities.

Follows SPEC section 4.1 and 4.2 using real schema recorded in DATA_NOTES.md.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp


# Exact column names from DATA_NOTES.md
ADDRESS_COL = "address"
TIME_STEP_COL = "Time step"
CLASS_COL = "class"
INPUT_ADDR_COL = "input_address"
OUTPUT_ADDR_COL = "output_address"

# 55 numeric feature columns from DATA_NOTES.md
FEATURE_COLUMNS: list[str] = [
    "num_txs_as_sender",
    "num_txs_as receiver",
    "first_block_appeared_in",
    "last_block_appeared_in",
    "lifetime_in_blocks",
    "total_txs",
    "first_sent_block",
    "first_received_block",
    "num_timesteps_appeared_in",
    "btc_transacted_total",
    "btc_transacted_min",
    "btc_transacted_max",
    "btc_transacted_mean",
    "btc_transacted_median",
    "btc_sent_total",
    "btc_sent_min",
    "btc_sent_max",
    "btc_sent_mean",
    "btc_sent_median",
    "btc_received_total",
    "btc_received_min",
    "btc_received_max",
    "btc_received_mean",
    "btc_received_median",
    "fees_total",
    "fees_min",
    "fees_max",
    "fees_mean",
    "fees_median",
    "fees_as_share_total",
    "fees_as_share_min",
    "fees_as_share_max",
    "fees_as_share_mean",
    "fees_as_share_median",
    "blocks_btwn_txs_total",
    "blocks_btwn_txs_min",
    "blocks_btwn_txs_max",
    "blocks_btwn_txs_mean",
    "blocks_btwn_txs_median",
    "blocks_btwn_input_txs_total",
    "blocks_btwn_input_txs_min",
    "blocks_btwn_input_txs_max",
    "blocks_btwn_input_txs_mean",
    "blocks_btwn_input_txs_median",
    "blocks_btwn_output_txs_total",
    "blocks_btwn_output_txs_min",
    "blocks_btwn_output_txs_max",
    "blocks_btwn_output_txs_mean",
    "blocks_btwn_output_txs_median",
    "num_addr_transacted_multiple",
    "transacted_w_address_total",
    "transacted_w_address_min",
    "transacted_w_address_max",
    "transacted_w_address_mean",
    "transacted_w_address_median",
]

# Raw class code mapping to model labels:
# 1 -> 1 (illicit)
# 2 -> 0 (licit)
# 3 -> -1 (unknown, excluded from train/test evaluation)
CLASS_MAP: dict[int, int] = {
    1: 1,
    2: 0,
    3: -1,
}


def load_and_clean_classes(classes_path: Path | str) -> pd.DataFrame:
    """Load wallets_classes.csv and map raw class codes to labels.

    Args:
        classes_path: Path to wallets_classes.csv.

    Returns:
        DataFrame with columns: ['address', 'raw_class', 'label'].
    """
    df = pd.read_csv(
        classes_path,
        dtype={ADDRESS_COL: str, CLASS_COL: np.int32},
    )
    df["raw_class"] = df[CLASS_COL]
    df["label"] = df["raw_class"].map(CLASS_MAP).astype(np.int8)
    return df[[ADDRESS_COL, "raw_class", "label"]]


def build_address_mapping(
    addresses: Iterable[str],
) -> tuple[dict[str, int], np.ndarray]:
    """Map string addresses to contiguous integer indices 0..N-1 deterministically.

    Args:
        addresses: Iterable of address strings.

    Returns:
        tuple of (address_to_idx dict, idx_to_address 1D numpy array).
    """
    unique_sorted = sorted(set(addresses))
    address_to_idx = {addr: i for i, addr in enumerate(unique_sorted)}
    idx_to_address = np.array(unique_sorted, dtype=object)
    return address_to_idx, idx_to_address


def build_sparse_graph(
    edgelist_path: Path | str,
    address_to_idx: dict[str, int],
) -> sp.csr_matrix:
    """Construct a directed sparse adjacency matrix from AddrAddr_edgelist.csv.

    Args:
        edgelist_path: Path to AddrAddr_edgelist.csv.
        address_to_idx: Mapping from address string to integer node index.

    Returns:
        sp.csr_matrix of shape (N, N) where N is len(address_to_idx).
    """
    n = len(address_to_idx)
    edges_df = pd.read_csv(
        edgelist_path,
        dtype={INPUT_ADDR_COL: str, OUTPUT_ADDR_COL: str},
    )

    src = edges_df[INPUT_ADDR_COL].map(address_to_idx).to_numpy(dtype=np.int32)
    dst = edges_df[OUTPUT_ADDR_COL].map(address_to_idx).to_numpy(dtype=np.int32)

    # Edge weights are 1.0 (binary directed edges)
    data = np.ones(len(src), dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(n, n), dtype=np.float32)
    return adj.tocsr()


def load_and_clean_features(
    features_path: Path | str,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load wallets_features.csv with explicit dtypes and deduplicate exact duplicate rows.

    Args:
        features_path: Path to wallets_features.csv.
        feature_cols: Optional list of numeric feature column names. Defaults to FEATURE_COLUMNS.

    Returns:
        Cleaned DataFrame deduplicated on [address, 'Time step'].
    """
    cols_to_use = feature_cols if feature_cols is not None else FEATURE_COLUMNS

    dtype_dict: dict[str, Any] = {
        ADDRESS_COL: str,
        TIME_STEP_COL: np.int32,
    }
    for col in cols_to_use:
        dtype_dict[col] = np.float32

    df = pd.read_csv(
        features_path,
        dtype=dtype_dict,
    )

    # Deduplicate exact duplicate (address, Time step) rows (DATA_NOTES.md Point 1)
    df = df.drop_duplicates(subset=[ADDRESS_COL, TIME_STEP_COL], keep="first")
    return df


def split_temporal(
    df: pd.DataFrame,
    train_range: tuple[int, int] = (1, 34),
    test_range: tuple[int, int] = (35, 49),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform temporal split on labelled rows (illicit=1, licit=0).

    Args:
        df: DataFrame containing ADDRESS_COL, TIME_STEP_COL, and 'label'.
        train_range: (min_step, max_step) inclusive for training.
        test_range: (min_step, max_step) inclusive for testing.

    Returns:
        tuple of (train_df, test_df) containing only labelled rows.
    """
    labelled = df[df["label"].isin([0, 1])].copy()

    train_mask = (labelled[TIME_STEP_COL] >= train_range[0]) & (
        labelled[TIME_STEP_COL] <= train_range[1]
    )
    test_mask = (labelled[TIME_STEP_COL] >= test_range[0]) & (
        labelled[TIME_STEP_COL] <= test_range[1]
    )

    train_df = labelled[train_mask].copy()
    test_df = labelled[test_mask].copy()
    return train_df, test_df


def check_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Check address leakage between training and testing splits.

    Identifies which test addresses appeared in the training period and creates
    the 'cold' test subset (addresses never seen in training).

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.

    Returns:
        tuple of (leakage_metrics dict, cold_test_df).
    """
    train_addrs = set(train_df[ADDRESS_COL].unique())
    test_addrs = set(test_df[ADDRESS_COL].unique())

    overlap_addrs = train_addrs.intersection(test_addrs)
    overlap_count = len(overlap_addrs)
    overlap_share = (
        float(overlap_count / len(test_addrs)) if len(test_addrs) > 0 else 0.0
    )

    # Mark rows that belong to cold addresses
    is_cold_row = ~test_df[ADDRESS_COL].isin(train_addrs)
    cold_test_df = test_df[is_cold_row].copy()
    cold_addrs = set(cold_test_df[ADDRESS_COL].unique())

    metrics: dict[str, Any] = {
        "train_rows": len(train_df),
        "train_unique_addresses": len(train_addrs),
        "test_rows": len(test_df),
        "test_unique_addresses": len(test_addrs),
        "overlap_addresses": overlap_count,
        "overlap_address_share": round(overlap_share, 4),
        "cold_test_rows": len(cold_test_df),
        "cold_test_addresses": len(cold_addrs),
        "cold_address_share": round(
            float(len(cold_addrs) / len(test_addrs)) if test_addrs else 0.0, 4
        ),
    }

    return metrics, cold_test_df


def get_latest_features_per_address(df: pd.DataFrame) -> pd.DataFrame:
    """Select the latest row (highest Time step) for each address for serving.

    Args:
        df: DataFrame containing ADDRESS_COL and TIME_STEP_COL.

    Returns:
        DataFrame with one row per unique address.
    """
    sorted_df = df.sort_values(TIME_STEP_COL)
    latest_df = sorted_df.groupby(ADDRESS_COL, as_index=False).last()
    return latest_df


def save_artifacts(
    artifacts_dir: Path | str,
    address_to_idx: dict[str, int],
    idx_to_address: np.ndarray,
    adj_matrix: sp.csr_matrix,
) -> None:
    """Save graph artifacts to artifacts directory.

    Args:
        artifacts_dir: Destination directory.
        address_to_idx: Mapping dict from address string to int index.
        idx_to_address: Array mapping int index to address string.
        adj_matrix: Sparse adjacency matrix.
    """
    path = Path(artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(address_to_idx, path / "address_to_idx.joblib")
    joblib.dump(idx_to_address, path / "idx_to_address.joblib")
    sp.save_npz(path / "address_graph.npz", adj_matrix)


def load_artifacts(
    artifacts_dir: Path | str,
) -> tuple[dict[str, int], np.ndarray, sp.csr_matrix]:
    """Load graph artifacts from artifacts directory.

    Args:
        artifacts_dir: Source directory.

    Returns:
        tuple of (address_to_idx, idx_to_address, adj_matrix).
    """
    path = Path(artifacts_dir)
    address_to_idx: dict[str, int] = joblib.load(path / "address_to_idx.joblib")
    idx_to_address: np.ndarray = joblib.load(path / "idx_to_address.joblib")
    adj_matrix: sp.csr_matrix = sp.load_npz(path / "address_graph.npz")
    return address_to_idx, idx_to_address, adj_matrix
