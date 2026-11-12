"""Tests for scripts/00_inspect_data.py and repository scaffold."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_inspect_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "00_inspect_data.py"
    spec = importlib.util.spec_from_file_location("inspect_data", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspect_synthetic_files(tmp_path: Path) -> None:
    """Verify inspection helpers on a tiny synthetic dataset."""
    mod = _load_inspect_module()

    classes_csv = tmp_path / "wallets_classes.csv"
    classes_csv.write_text(
        "address,class\naddr1,1\naddr2,2\naddr3,3\n",
        encoding="utf-8",
    )

    features_csv = tmp_path / "wallets_features.csv"
    features_csv.write_text(
        "address,Time step,num_txs_as_sender,num_txs_as receiver\n"
        "addr1,1,1.0,0.0\n"
        "addr1,2,2.0,1.0\n"
        "addr2,1,0.0,1.0\n"
        "addr3,5,1.0,1.0\n",
        encoding="utf-8",
    )

    edges_csv = tmp_path / "AddrAddr_edgelist.csv"
    edges_csv.write_text(
        "input_address,output_address\naddr1,addr2\naddr2,addr3\naddr3,addr3\n",
        encoding="utf-8",
    )

    assert mod.check_files_exist(tmp_path) == []

    c_info = mod.inspect_classes(classes_csv)
    assert c_info["row_count"] == 3
    assert c_info["class_counts"] == {"1": 1, "2": 1, "3": 1}

    f_info = mod.inspect_features(features_csv)
    assert f_info["row_count"] == 4
    assert len(f_info["unique_addresses"]) == 3
    assert f_info["multi_row_addrs"] == 1

    e_info = mod.inspect_edges(edges_csv)
    assert e_info["row_count"] == 3
    assert e_info["self_loops"] == 1
    assert len(e_info["unique_edge_addrs"]) == 3
