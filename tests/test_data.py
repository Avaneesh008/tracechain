"""Unit tests for data loading, cleaning, mapping, and temporal split."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from tracechain.data import (
    ADDRESS_COL,
    TIME_STEP_COL,
    build_address_mapping,
    build_sparse_graph,
    check_leakage,
    get_latest_features_per_address,
    load_and_clean_classes,
    load_and_clean_features,
    load_artifacts,
    save_artifacts,
    split_temporal,
)


def test_load_and_clean_classes(tmp_path: Path):
    """Test raw class mapping: 1->1 (illicit), 2->0 (licit), 3->-1 (unknown)."""
    csv_file = tmp_path / "wallets_classes.csv"
    csv_file.write_text(
        "address,class\n"
        "addr_illicit,1\n"
        "addr_licit,2\n"
        "addr_unknown,3\n",
        encoding="utf-8",
    )

    df = load_and_clean_classes(csv_file)
    assert len(df) == 3
    assert set(df.columns) == {"address", "raw_class", "label"}

    row_illicit = df[df["address"] == "addr_illicit"].iloc[0]
    assert row_illicit["label"] == 1

    row_licit = df[df["address"] == "addr_licit"].iloc[0]
    assert row_licit["label"] == 0

    row_unknown = df[df["address"] == "addr_unknown"].iloc[0]
    assert row_unknown["label"] == -1


def test_build_address_mapping():
    """Test deterministic address to index mapping."""
    addresses = ["addrC", "addrA", "addrB", "addrA"]
    addr_to_idx, idx_to_addr = build_address_mapping(addresses)

    assert len(addr_to_idx) == 3
    assert addr_to_idx["addrA"] == 0
    assert addr_to_idx["addrB"] == 1
    assert addr_to_idx["addrC"] == 2

    assert list(idx_to_addr) == ["addrA", "addrB", "addrC"]


def test_build_sparse_graph(tmp_path: Path):
    """Test graph construction from edgelist."""
    edgelist_csv = tmp_path / "AddrAddr_edgelist.csv"
    edgelist_csv.write_text(
        "input_address,output_address\n"
        "addrA,addrB\n"
        "addrB,addrC\n"
        "addrC,addrA\n",
        encoding="utf-8",
    )
    addr_to_idx = {"addrA": 0, "addrB": 1, "addrC": 2}

    adj = build_sparse_graph(edgelist_csv, addr_to_idx)
    assert adj.shape == (3, 3)
    assert adj.nnz == 3
    assert adj[0, 1] == 1.0
    assert adj[1, 2] == 1.0
    assert adj[2, 0] == 1.0
    assert adj[0, 2] == 0.0


def test_load_and_clean_features_deduplication(tmp_path: Path):
    """Test that exact duplicates on [address, Time step] are dropped."""
    features_csv = tmp_path / "wallets_features.csv"
    features_csv.write_text(
        "address,Time step,num_txs_as_sender\n"
        "addrA,1,10.0\n"
        "addrA,1,10.0\n"  # Duplicate
        "addrA,2,20.0\n"
        "addrB,1,5.0\n",
        encoding="utf-8",
    )

    df = load_and_clean_features(
        features_csv, feature_cols=["num_txs_as_sender"]
    )
    assert len(df) == 3
    assert df["num_txs_as_sender"].dtype == np.float32
    assert df[TIME_STEP_COL].dtype == np.int32


def test_split_temporal():
    """Test temporal split excludes unknown labels and respects time step boundaries."""
    data = {
        "address": ["a1", "a2", "a3", "a4", "a5"],
        "Time step": [10, 30, 35, 45, 20],
        "label": [1, 0, 1, 0, -1],  # a5 is unknown (-1)
        "val": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    df = pd.DataFrame(data)

    train_df, test_df = split_temporal(
        df, train_range=(1, 34), test_range=(35, 49)
    )

    assert len(train_df) == 2
    assert set(train_df["address"]) == {"a1", "a2"}
    assert set(train_df["label"]) == {0, 1}

    assert len(test_df) == 2
    assert set(test_df["address"]) == {"a3", "a4"}
    assert "a5" not in train_df["address"].values
    assert "a5" not in test_df["address"].values


def test_check_leakage():
    """Test leakage metrics and cold test subset identification."""
    train_data = {
        "address": ["seen1", "seen2"],
        "Time step": [10, 20],
        "label": [1, 0],
    }
    test_data = {
        "address": ["seen1", "cold1", "cold2"],
        "Time step": [36, 40, 42],
        "label": [1, 0, 1],
    }
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    metrics, cold_test_df = check_leakage(train_df, test_df)

    assert metrics["train_unique_addresses"] == 2
    assert metrics["test_unique_addresses"] == 3
    assert metrics["overlap_addresses"] == 1
    assert np.isclose(metrics["overlap_address_share"], 1 / 3, atol=1e-3)
    assert metrics["cold_test_rows"] == 2
    assert set(cold_test_df["address"]) == {"cold1", "cold2"}


def test_get_latest_features_per_address():
    """Test serving rule: pick latest Time step per address."""
    df = pd.DataFrame(
        {
            "address": ["a1", "a1", "a2", "a2"],
            "Time step": [5, 10, 15, 2],
            "feature": [100, 200, 300, 400],
        }
    )
    latest = get_latest_features_per_address(df)
    assert len(latest) == 2
    a1_row = latest[latest["address"] == "a1"].iloc[0]
    assert a1_row["Time step"] == 10
    assert a1_row["feature"] == 200

    a2_row = latest[latest["address"] == "a2"].iloc[0]
    assert a2_row["Time step"] == 15
    assert a2_row["feature"] == 300


def test_save_and_load_artifacts(tmp_path: Path):
    """Test saving and loading graph artifacts."""
    addr_to_idx = {"addr1": 0, "addr2": 1}
    idx_to_addr = np.array(["addr1", "addr2"], dtype=object)
    adj = sp.csr_matrix([[0, 1], [1, 0]], dtype=np.float32)

    save_artifacts(tmp_path, addr_to_idx, idx_to_addr, adj)

    loaded_a2i, loaded_i2a, loaded_adj = load_artifacts(tmp_path)
    assert loaded_a2i == addr_to_idx
    np.testing.assert_array_equal(loaded_i2a, idx_to_addr)
    assert (loaded_adj != adj).nnz == 0

