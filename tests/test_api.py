"""Contract and integration tests for FastAPI routes (api/main.py) and TraceChainService."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root and src/ are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier

from api.main import create_app
from tracechain.data import ADDRESS_COL, TIME_STEP_COL, save_artifacts
from tracechain.service import REQUIRED_ARTIFACT_FILES


def _create_synthetic_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a complete tiny synthetic artifacts + reports environment for fast contract tests."""
    artifacts_dir = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports"
    raw_dir = tmp_path / "raw"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    addrs = ["addr_vasp", "addr_mid", "addr_illicit", "addr_licit", "addr_isolated"]
    address_to_idx = {a: i for i, a in enumerate(addrs)}
    idx_to_address = np.array(addrs, dtype=object)

    # Directed edges: addr_illicit (2) -> addr_mid (1) -> addr_vasp (0), addr_licit (3) -> addr_vasp (0)
    rows = [2, 1, 3]
    cols = [1, 0, 0]
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(5, 5), dtype=np.float32).tocsr()
    save_artifacts(artifacts_dir, address_to_idx, idx_to_address, adj)

    feature_names = [
        "num_txs_as_sender",
        "num_txs_as receiver",
        "in_degree",
        "out_degree",
        "pagerank",
        "mean_neighbor_degree",
        "hops",
    ]
    (artifacts_dir / "features.json").write_text(
        json.dumps(feature_names), encoding="utf-8"
    )
    feature_stats = {
        name: {"mean": 1.0, "std": 1.0} for name in feature_names
    }
    (artifacts_dir / "feature_stats.json").write_text(
        json.dumps(feature_stats), encoding="utf-8"
    )

    # Train a tiny Random Forest on 7 features
    x_train = pd.DataFrame(
        [
            [1.0, 10.0, 2.0, 0.0, 0.3, 1.0, 0.0],
            [2.0, 2.0, 1.0, 1.0, 0.2, 1.5, 1.0],
            [25.0, 1.0, 0.0, 1.0, 0.1, 2.0, 2.0],
            [1.0, 1.0, 0.0, 1.0, 0.2, 2.0, 1.0],
        ],
        columns=feature_names,
    )
    y_train = np.array([0, 0, 1, 0], dtype=np.int8)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(x_train, y_train)
    joblib.dump(clf, artifacts_dir / "model.joblib")

    # Graph features
    graph_df = pd.DataFrame(
        {
            ADDRESS_COL: addrs,
            "in_degree": np.array([2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "out_degree": np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "pagerank": np.array([0.35, 0.25, 0.15, 0.15, 0.10], dtype=np.float32),
            "mean_neighbor_degree": np.array([1.5, 2.0, 2.0, 2.0, 0.0], dtype=np.float32),
        }
    )
    graph_df.to_parquet(artifacts_dir / "graph_features.parquet", index=False)

    # VASP seeds & attribution index (using stored next_hop_idx pointers)
    vasp_seeds_df = pd.DataFrame(
        [
            {
                "seed_id": "VASP-PROXY-01",
                "name": "VASP-PROXY-01",
                "address": "addr_vasp",
                "node_idx": 0,
                "kind": "proxy",
                "seed_score": 0.99,
            }
        ]
    )
    vasp_seeds_df.to_parquet(artifacts_dir / "vasp_seeds.parquet", index=False)

    vasp_index_df = pd.DataFrame(
        {
            ADDRESS_COL: addrs,
            "nearest_seed_id": [
                "VASP-PROXY-01",
                "VASP-PROXY-01",
                "VASP-PROXY-01",
                "VASP-PROXY-01",
                None,
            ],
            "nearest_seed_name": [
                "VASP-PROXY-01",
                "VASP-PROXY-01",
                "VASP-PROXY-01",
                "VASP-PROXY-01",
                None,
            ],
            "nearest_seed_address": [
                "addr_vasp",
                "addr_vasp",
                "addr_vasp",
                "addr_vasp",
                None,
            ],
            "nearest_seed_kind": ["proxy", "proxy", "proxy", "proxy", None],
            "hops": np.array([0, 1, 2, 1, -1], dtype=np.int32),
            "confidence": ["high", "high", "high", "high", "unattributed"],
            "confidence_score": np.array([1.0, 0.5, 0.3333, 0.5, np.nan], dtype=np.float32),
            "next_hop_idx": np.array([-1, 0, 1, 0, -1], dtype=np.int32),
        }
    )
    vasp_index_df.to_parquet(artifacts_dir / "vasp_index.parquet", index=False)

    # Test split parquet for /api/wallets/sample and feature lookup
    test_df = pd.DataFrame(
        {
            ADDRESS_COL: ["addr_illicit", "addr_licit", "addr_vasp", "addr_mid", "addr_isolated"],
            TIME_STEP_COL: np.array([40, 42, 38, 39, 41], dtype=np.int32),
            "label": np.array([1, 0, 0, 0, 0], dtype=np.int8),
            "num_txs_as_sender": np.array([25.0, 1.0, 1.0, 2.0, 0.0], dtype=np.float32),
            "num_txs_as receiver": np.array([1.0, 1.0, 10.0, 2.0, 0.0], dtype=np.float32),
        }
    )
    test_df.to_parquet(artifacts_dir / "test.parquet", index=False)

    metrics_json = {
        "baseline": {"test_all": {"f1": 0.4416, "pr_auc": 0.3974}},
        "graph_enhanced": {"test_all": {"f1": 0.5762, "pr_auc": 0.6057}},
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics_json), encoding="utf-8"
    )
    attr_json = {
        "seed_kind": "proxy",
        "default_k": 30,
        "active_index_stats": {"total_addresses": 5, "reachable_addresses": 4},
    }
    (reports_dir / "attribution_stats.json").write_text(
        json.dumps(attr_json), encoding="utf-8"
    )

    return artifacts_dir, reports_dir, raw_dir


def test_503_when_artifacts_missing(tmp_path: Path) -> None:
    """Verify routes return 503 when pointed at an empty temporary artifacts folder."""
    empty_artifacts = tmp_path / "empty_artifacts"
    empty_reports = tmp_path / "empty_reports"
    empty_artifacts.mkdir()
    empty_reports.mkdir()

    app = create_app(
        artifacts_dir=empty_artifacts,
        reports_dir=empty_reports,
        raw_dir=tmp_path,
    )
    with TestClient(app) as client:
        # /api/health returns 200 with model_loaded=False and graph_nodes=0
        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        assert r_health.json() == {
            "status": "ok",
            "model_loaded": False,
            "graph_nodes": 0,
        }

        # All data/scoring routes return 503 with {"detail": "..."}
        for path in (
            "/api/wallets/sample?label=illicit&n=5",
            "/api/score/any_address",
            "/api/trace/any_address",
            "/api/metrics",
        ):
            resp = client.get(path)
            assert resp.status_code == 503
            assert "detail" in resp.json()


def test_contract_200_shapes_and_404_on_synthetic_artifacts(tmp_path: Path) -> None:
    """Verify exact SPEC section 5 JSON contract for all 5 routes plus 404 for unknown address."""
    artifacts_dir, reports_dir, raw_dir = _create_synthetic_artifacts(tmp_path)
    app = create_app(
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        raw_dir=raw_dir,
    )

    with TestClient(app) as client:
        # 1. GET /api/health (also verify CORS header for http://localhost:5173)
        r_health = client.get(
            "/api/health", headers={"Origin": "http://localhost:5173"}
        )
        assert r_health.status_code == 200
        assert r_health.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert r_health.json() == {
            "status": "ok",
            "model_loaded": True,
            "graph_nodes": 5,
        }

        # 2. GET /api/wallets/sample
        r_sample_illicit = client.get("/api/wallets/sample?label=illicit&n=5")
        assert r_sample_illicit.status_code == 200
        s_illicit = r_sample_illicit.json()
        assert "wallets" in s_illicit
        assert len(s_illicit["wallets"]) == 1
        assert s_illicit["wallets"][0] == {
            "address": "addr_illicit",
            "label": "illicit",
            "time_step": 40,
        }

        r_sample_licit = client.get("/api/wallets/sample?label=licit&n=2")
        assert r_sample_licit.status_code == 200
        s_licit = r_sample_licit.json()
        assert len(s_licit["wallets"]) == 2
        for w in s_licit["wallets"]:
            assert set(w.keys()) == {"address", "label", "time_step"}
            assert w["label"] == "licit"

        # 3. GET /api/score/{address}
        r_score = client.get("/api/score/addr_illicit")
        assert r_score.status_code == 200
        score_data = r_score.json()
        assert set(score_data.keys()) == {
            "address",
            "time_step",
            "risk_score",
            "risk_label",
            "nearest_vasp",
            "top_features",
            "explanation_method",
        }
        assert score_data["address"] == "addr_illicit"
        assert score_data["time_step"] == 40
        assert 0.0 <= score_data["risk_score"] <= 1.0
        assert score_data["risk_label"] in ("illicit", "licit")
        assert score_data["explanation_method"] in ("shap", "importance_x_deviation")

        vasp_info = score_data["nearest_vasp"]
        assert set(vasp_info.keys()) == {
            "id",
            "name",
            "kind",
            "hops",
            "confidence",
            "confidence_score",
        }
        assert vasp_info["id"] == "VASP-PROXY-01"
        assert vasp_info["kind"] == "proxy"
        assert vasp_info["hops"] == 2
        assert vasp_info["confidence"] == "high"
        assert isinstance(vasp_info["confidence_score"], float)

        assert len(score_data["top_features"]) == 5
        for tf in score_data["top_features"]:
            assert set(tf.keys()) == {"name", "value", "contribution"}

        # Unattributed node check
        r_unattr = client.get("/api/score/addr_isolated")
        assert r_unattr.status_code == 200
        unattr_vasp = r_unattr.json()["nearest_vasp"]
        assert unattr_vasp["hops"] is None
        assert unattr_vasp["confidence_score"] is None
        assert unattr_vasp["confidence"] == "low"

        # 4. GET /api/trace/{address}
        r_trace = client.get("/api/trace/addr_illicit")
        assert r_trace.status_code == 200
        trace_data = r_trace.json()
        assert set(trace_data.keys()) == {"path", "nodes", "edges"}
        assert trace_data["path"] == ["addr_illicit", "addr_mid", "addr_vasp"]
        assert len(trace_data["nodes"]) <= 60
        node_kinds = {n["id"]: n["kind"] for n in trace_data["nodes"]}
        assert node_kinds["addr_illicit"] == "wallet"
        assert node_kinds["addr_mid"] == "path"
        assert node_kinds["addr_vasp"] == "vasp"
        for edge in trace_data["edges"]:
            assert set(edge.keys()) == {"source", "target"}

        # 5. GET /api/metrics
        r_metrics = client.get("/api/metrics")
        assert r_metrics.status_code == 200
        m_data = r_metrics.json()
        assert "baseline" in m_data
        assert "graph_enhanced" in m_data
        assert "attribution_stats" in m_data

        # 6. 404 for unknown address
        r_404_score = client.get("/api/score/not_a_real_bitcoin_address")
        assert r_404_score.status_code == 404
        assert "detail" in r_404_score.json()

        r_404_trace = client.get("/api/trace/not_a_real_bitcoin_address")
        assert r_404_trace.status_code == 404
        assert "detail" in r_404_trace.json()


def _real_artifacts_available() -> bool:
    """Return True if all real artifacts and reports/metrics.json exist in the workspace."""
    artifacts_dir = _REPO_ROOT / "artifacts"
    reports_dir = _REPO_ROOT / "reports"
    for name in REQUIRED_ARTIFACT_FILES:
        if not (artifacts_dir / name).is_file():
            return False
    return (reports_dir / "metrics.json").is_file()


@pytest.mark.skipif(
    not _real_artifacts_available(),
    reason="Real artifacts not present in artifacts/ or reports/metrics.json",
)
def test_against_real_artifacts() -> None:
    """Integration test against the real 822,942-node artifacts and trained Random Forest."""
    app = create_app()
    with TestClient(app) as client:
        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        assert r_health.json() == {
            "status": "ok",
            "model_loaded": True,
            "graph_nodes": 822942,
        }

        illicit_sample = client.get("/api/wallets/sample?label=illicit&n=2").json()["wallets"][0]
        licit_sample = client.get("/api/wallets/sample?label=licit&n=2").json()["wallets"][0]

        for sample in (illicit_sample, licit_sample):
            addr = sample["address"]
            s_resp = client.get(f"/api/score/{addr}")
            assert s_resp.status_code == 200
            s_json = s_resp.json()
            assert s_json["address"] == addr
            assert s_json["nearest_vasp"]["kind"] == "proxy"
            assert len(s_json["top_features"]) == 5

            t_resp = client.get(f"/api/trace/{addr}")
            assert t_resp.status_code == 200
            t_json = t_resp.json()
            assert len(t_json["path"]) >= 1
            assert len(t_json["nodes"]) <= 60

