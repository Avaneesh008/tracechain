"""Service layer for wallet risk scoring, explanations, VASP attribution, and graph tracing.

Implements the backend logic for SPEC section 5:
- Loads all required artifacts (model, features, feature stats, address graph,
  graph features, VASP attribution index, dataset splits, and report metrics) once at startup.
- Serves /api/health, /api/wallets/sample, /api/score/{address}, /api/trace/{address}, and /api/metrics.
- Reuses src/tracechain/data.py, explain.py, and vasp.py without duplicating logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from tracechain.data import (
    ADDRESS_COL,
    FEATURE_COLUMNS,
    TIME_STEP_COL,
    get_latest_features_per_address,
    load_and_clean_features,
    load_artifacts,
)
from tracechain.explain import explain_wallet
from tracechain.vasp import build_trace_subgraph


REQUIRED_ARTIFACT_FILES = (
    "model.joblib",
    "features.json",
    "feature_stats.json",
    "address_graph.npz",
    "address_to_idx.joblib",
    "idx_to_address.joblib",
    "graph_features.parquet",
    "vasp_index.parquet",
    "vasp_seeds.parquet",
)


class ArtifactsNotLoadedError(RuntimeError):
    """Raised when required model or graph artifacts are missing or not loaded (HTTP 503)."""


class AddressNotFoundError(KeyError):
    """Raised when a queried Bitcoin address does not exist in the dataset (HTTP 404)."""


class TraceChainService:
    """In-memory service holding loaded artifacts and serving API requests."""

    def __init__(
        self,
        artifacts_dir: Path | str = Path("artifacts"),
        reports_dir: Path | str = Path("reports"),
        raw_dir: Path | str = Path("data/raw"),
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.reports_dir = Path(reports_dir)
        self.raw_dir = Path(raw_dir)

        self.model_loaded: bool = False
        self.model: Any = None
        self.feature_names: list[str] = []
        self.feature_stats: dict[str, dict[str, float]] = {}

        self.address_to_idx: dict[str, int] = {}
        self.idx_to_address: np.ndarray = np.array([], dtype=object)
        self.adj_directed: sp.csr_matrix | None = None
        self.adj_csc: sp.csc_matrix | None = None

        self.graph_features_df: pd.DataFrame | None = None
        self.vasp_index_df: pd.DataFrame | None = None
        self.vasp_seeds_df: pd.DataFrame | None = None

        # Pre-indexed latest wallet feature rows (N x len(FEATURE_COLUMNS)) aligned with 0..N-1
        self.base_features_matrix: np.ndarray | None = None
        self.latest_time_steps: np.ndarray | None = None
        self.base_col_to_pos: dict[str, int] = {}

        # Pre-indexed test samples for /api/wallets/sample
        self.test_samples_by_label: dict[str, list[dict[str, Any]]] = {
            "illicit": [],
            "licit": [],
        }

        self.metrics_payload: dict[str, Any] = {}
        self.attribution_stats_payload: dict[str, Any] = {}

    def check_required_files(self) -> list[str]:
        """Return list of missing required artifact or report files."""
        missing = [
            name
            for name in REQUIRED_ARTIFACT_FILES
            if not (self.artifacts_dir / name).is_file()
        ]
        if not (self.reports_dir / "metrics.json").is_file():
            missing.append("reports/metrics.json")
        return missing

    def load_all(self) -> bool:
        """Load all artifacts and report JSON files once at startup.

        Returns:
            True if all required artifacts loaded successfully, False otherwise.
        """
        missing = self.check_required_files()
        if missing:
            self.model_loaded = False
            return False

        try:
            # 1. Load model, feature names, and feature stats
            self.model = joblib.load(self.artifacts_dir / "model.joblib")
            # Execute single-row prediction inside the caller thread to avoid threadpool overhead
            if hasattr(self.model, "n_jobs"):
                self.model.n_jobs = 1

            self.feature_names = json.loads(
                (self.artifacts_dir / "features.json").read_text(encoding="utf-8")
            )
            self.feature_stats = json.loads(
                (self.artifacts_dir / "feature_stats.json").read_text(encoding="utf-8")
            )

            # 2. Load sparse graph and address mappings via data.load_artifacts
            self.address_to_idx, self.idx_to_address, self.adj_directed = load_artifacts(
                self.artifacts_dir
            )
            self.adj_csc = self.adj_directed.tocsc()

            # 3. Load graph features and VASP index
            self.graph_features_df = pd.read_parquet(
                self.artifacts_dir / "graph_features.parquet"
            )
            self.vasp_index_df = pd.read_parquet(
                self.artifacts_dir / "vasp_index.parquet"
            )
            self.vasp_seeds_df = pd.read_parquet(
                self.artifacts_dir / "vasp_seeds.parquet"
            )

            # 4. Build or load latest per-address base feature matrix aligned with 0..N-1
            self._load_latest_wallet_features()

            # 5. Load test-period samples for /api/wallets/sample
            self._load_test_samples()

            # 6. Load reports/metrics.json and reports/attribution_stats.json
            self.metrics_payload = json.loads(
                (self.reports_dir / "metrics.json").read_text(encoding="utf-8")
            )
            attr_stats_path = self.reports_dir / "attribution_stats.json"
            if attr_stats_path.is_file():
                self.attribution_stats_payload = json.loads(
                    attr_stats_path.read_text(encoding="utf-8")
                )
            else:
                self.attribution_stats_payload = {}

            self.model_loaded = True
            return True
        except Exception:
            self.model_loaded = False
            return False

    def _load_latest_wallet_features(self) -> None:
        """Load the latest feature row (highest Time step) per address into an O(1) lookup array."""
        n = len(self.idx_to_address)
        base_cols = [c for c in FEATURE_COLUMNS if c in self.feature_names]
        if not base_cols:
            base_cols = [
                c
                for c in self.feature_names
                if c
                not in {
                    "in_degree",
                    "out_degree",
                    "pagerank",
                    "mean_neighbor_degree",
                    "hops",
                }
            ]
        self.base_col_to_pos = {c: i for i, c in enumerate(base_cols)}

        cache_path = self.artifacts_dir / "latest_wallet_features.parquet"
        raw_features_path = self.raw_dir / "wallets_features.csv"

        latest_df: pd.DataFrame | None = None
        if cache_path.is_file():
            cached = pd.read_parquet(cache_path)
            if len(cached) == n:
                latest_df = cached

        if latest_df is None and raw_features_path.is_file():
            raw_df = load_and_clean_features(raw_features_path, feature_cols=base_cols)
            latest_df = get_latest_features_per_address(raw_df)
            # Align to 0..N-1 order by node_idx
            latest_df["_idx"] = latest_df[ADDRESS_COL].map(self.address_to_idx)
            latest_df = (
                latest_df.dropna(subset=["_idx"])
                .sort_values("_idx")
                .drop(columns=["_idx"])
                .reset_index(drop=True)
            )
            if len(latest_df) == n:
                try:
                    latest_df.to_parquet(cache_path, index=False)
                except Exception:
                    pass

        if latest_df is not None and len(latest_df) == n:
            self.latest_time_steps = latest_df[TIME_STEP_COL].to_numpy(dtype=np.int32)
            self.base_features_matrix = latest_df[base_cols].to_numpy(dtype=np.float32)
            return

        # Fallback when raw CSV is absent (e.g., synthetic tests or train/test parquets only)
        self.latest_time_steps = np.ones(n, dtype=np.int32)
        self.base_features_matrix = np.zeros((n, len(base_cols)), dtype=np.float32)

        frames: list[pd.DataFrame] = []
        for split_name in ("train.parquet", "test.parquet"):
            sp_path = self.artifacts_dir / split_name
            if sp_path.is_file():
                df_sp = pd.read_parquet(sp_path)
                cols_to_keep = [ADDRESS_COL, TIME_STEP_COL] + [
                    c for c in base_cols if c in df_sp.columns
                ]
                frames.append(df_sp[cols_to_keep])
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            latest_split = get_latest_features_per_address(combined)
            valid_rows = latest_split[
                latest_split[ADDRESS_COL].isin(self.address_to_idx)
            ].copy()
            idxs = valid_rows[ADDRESS_COL].map(self.address_to_idx).to_numpy(
                dtype=np.int32
            )
            self.latest_time_steps[idxs] = valid_rows[TIME_STEP_COL].to_numpy(
                dtype=np.int32
            )
            present_cols = [c for c in base_cols if c in valid_rows.columns]
            for c in present_cols:
                col_pos = self.base_col_to_pos[c]
                self.base_features_matrix[idxs, col_pos] = valid_rows[c].to_numpy(
                    dtype=np.float32
                )

    def _load_test_samples(self) -> None:
        """Pre-index test-period addresses with known labels for /api/wallets/sample."""
        self.test_samples_by_label = {"illicit": [], "licit": []}
        test_path = self.artifacts_dir / "test.parquet"
        if not test_path.is_file():
            return

        test_df = pd.read_parquet(test_path)
        latest_test = get_latest_features_per_address(test_df)

        # If model is loaded, prioritize samples where the model predicts clearly (high confidence)
        # so the demo sample buttons showcase clear illicit (>=0.5) and licit (<0.5) cases.
        base_cols = [c for c in FEATURE_COLUMNS if c in self.feature_names]
        if (
            self.model is not None
            and self.vasp_index_df is not None
            and all(c in latest_test.columns for c in base_cols)
        ):
            v_hops = self.vasp_index_df[[ADDRESS_COL, "hops"]]
            scored_test = latest_test.merge(v_hops, on=ADDRESS_COL, how="left")
            scored_test["hops"] = scored_test["hops"].fillna(-1).astype(np.float32)
            if all(c in scored_test.columns for c in self.feature_names):
                probs = self.model.predict_proba(scored_test[self.feature_names])[:, 1]
                scored_test = scored_test.copy()
                scored_test["_prob"] = probs
                # Prefer samples with 1..4 hops to a VASP seed so the demo trace graph shows a multi-hop path
                scored_test["_good_hops"] = (
                    (scored_test["hops"] >= 1) & (scored_test["hops"] <= 4)
                ).astype(np.int8)
                illicit_df = scored_test[scored_test["label"] == 1].sort_values(
                    ["_good_hops", "_prob", TIME_STEP_COL, ADDRESS_COL],
                    ascending=[False, False, False, True],
                )
                licit_df = scored_test[scored_test["label"] == 0].sort_values(
                    ["_good_hops", "_prob", TIME_STEP_COL, ADDRESS_COL],
                    ascending=[False, True, False, True],
                )
            else:
                illicit_df = latest_test[latest_test["label"] == 1].sort_values(
                    [TIME_STEP_COL, ADDRESS_COL], ascending=[False, True]
                )
                licit_df = latest_test[latest_test["label"] == 0].sort_values(
                    [TIME_STEP_COL, ADDRESS_COL], ascending=[False, True]
                )
        else:
            illicit_df = latest_test[latest_test["label"] == 1].sort_values(
                [TIME_STEP_COL, ADDRESS_COL], ascending=[False, True]
            )
            licit_df = latest_test[latest_test["label"] == 0].sort_values(
                [TIME_STEP_COL, ADDRESS_COL], ascending=[False, True]
            )

        self.test_samples_by_label["illicit"] = [
            {
                "address": str(row[ADDRESS_COL]),
                "label": "illicit",
                "time_step": int(row[TIME_STEP_COL]),
            }
            for _, row in illicit_df.head(500).iterrows()
        ]
        self.test_samples_by_label["licit"] = [
            {
                "address": str(row[ADDRESS_COL]),
                "label": "licit",
                "time_step": int(row[TIME_STEP_COL]),
            }
            for _, row in licit_df.head(500).iterrows()
        ]

    def ensure_loaded(self) -> None:
        """Raise ArtifactsNotLoadedError if artifacts are not loaded."""
        if not self.model_loaded:
            raise ArtifactsNotLoadedError(
                "Service artifacts are not loaded. Run pipeline scripts first."
            )

    def get_health(self) -> dict[str, Any]:
        """Return /api/health payload per SPEC section 5."""
        return {
            "status": "ok",
            "model_loaded": bool(self.model_loaded),
            "graph_nodes": int(len(self.idx_to_address)) if self.model_loaded else 0,
        }

    def get_sample_wallets(
        self, label: str = "illicit", n: int = 5
    ) -> dict[str, list[dict[str, Any]]]:
        """Return /api/wallets/sample?label=illicit|licit&n=5 per SPEC section 5."""
        self.ensure_loaded()
        norm_label = label.strip().lower()
        if norm_label not in ("illicit", "licit"):
            raise ValueError("Query parameter 'label' must be 'illicit' or 'licit'.")
        count = max(1, min(int(n), 100))
        candidates = self.test_samples_by_label.get(norm_label, [])
        return {"wallets": candidates[:count]}

    def _build_wallet_feature_vector(self, node_idx: int) -> tuple[np.ndarray, int]:
        """Assemble the 1D feature vector in self.feature_names order and latest time_step."""
        assert self.base_features_matrix is not None
        assert self.latest_time_steps is not None
        assert self.graph_features_df is not None
        assert self.vasp_index_df is not None

        time_step = int(self.latest_time_steps[node_idx])
        base_row = self.base_features_matrix[node_idx]
        g_row = self.graph_features_df.iloc[node_idx]
        v_row = self.vasp_index_df.iloc[node_idx]

        vec = np.zeros(len(self.feature_names), dtype=np.float32)
        for i, col_name in enumerate(self.feature_names):
            if col_name in self.base_col_to_pos:
                vec[i] = float(base_row[self.base_col_to_pos[col_name]])
            elif col_name in ("in_degree", "out_degree", "pagerank", "mean_neighbor_degree"):
                vec[i] = float(g_row[col_name])
            elif col_name == "hops":
                vec[i] = float(v_row["hops"])
        return vec, time_step

    def score_address(self, address: str) -> dict[str, Any]:
        """Return /api/score/{address} response per SPEC section 5."""
        self.ensure_loaded()
        if address not in self.address_to_idx:
            raise AddressNotFoundError(f"Address '{address}' not found in dataset.")

        node_idx = int(self.address_to_idx[address])
        feat_vec, time_step = self._build_wallet_feature_vector(node_idx)

        # Predict illicit probability using loaded Random Forest
        x_df = pd.DataFrame([feat_vec], columns=self.feature_names)
        if len(self.model.classes_) == 1:
            risk_score = 1.0 if int(self.model.classes_[0]) == 1 else 0.0
        else:
            probs = self.model.predict_proba(x_df)
            risk_score = float(probs[0, 1])

        risk_label = "illicit" if risk_score >= 0.5 else "licit"

        # Explain single wallet row using src/tracechain/explain.py
        expl = explain_wallet(
            clf=self.model,
            wallet_features=feat_vec,
            feature_names=self.feature_names,
            feature_stats=self.feature_stats,
            top_k=5,
        )

        # Look up nearest VASP seed from vasp_index.parquet
        assert self.vasp_index_df is not None
        v_row = self.vasp_index_df.iloc[node_idx]
        raw_hops = int(v_row["hops"])
        if raw_hops >= 0 and pd.notna(v_row["nearest_seed_id"]):
            seed_id = str(v_row["nearest_seed_id"])
            seed_name = str(v_row["nearest_seed_name"])
            seed_kind = str(v_row["nearest_seed_kind"])
            hops_val: int | None = raw_hops
            conf_label = str(v_row["confidence"])
            conf_score: float | None = round(float(v_row["confidence_score"]), 4)
        else:
            seed_id = "UNATTRIBUTED"
            seed_name = "Unattributed"
            seed_kind = "proxy"
            hops_val = None
            conf_label = "low"
            conf_score = None

        return {
            "address": address,
            "time_step": int(time_step),
            "risk_score": round(risk_score, 4),
            "risk_label": risk_label,
            "nearest_vasp": {
                "id": seed_id,
                "name": seed_name,
                "kind": seed_kind,
                "hops": hops_val,
                "confidence": conf_label,
                "confidence_score": conf_score,
            },
            "top_features": expl["top_features"],
            "explanation_method": expl["explanation_method"],
        }

    def trace_address(self, address: str) -> dict[str, Any]:
        """Return /api/trace/{address} response per SPEC section 5 using stored next-hop pointers."""
        self.ensure_loaded()
        if address not in self.address_to_idx:
            raise AddressNotFoundError(f"Address '{address}' not found in dataset.")

        assert self.adj_directed is not None
        assert self.vasp_index_df is not None

        return build_trace_subgraph(
            wallet_address=address,
            address_to_idx=self.address_to_idx,
            idx_to_address=self.idx_to_address,
            adj_directed=self.adj_directed,
            attribution_df=self.vasp_index_df,
            max_neighbors=12,
            max_total_nodes=60,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Return /api/metrics combining reports/metrics.json and attribution_stats.json."""
        self.ensure_loaded()
        return {
            **self.metrics_payload,
            "attribution_stats": self.attribution_stats_payload,
        }
