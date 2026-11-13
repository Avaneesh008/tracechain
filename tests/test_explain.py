"""Unit tests for explanation utilities and fallback logic."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from tracechain.explain import (
    explain_fallback_importance_x_deviation,
    explain_wallet,
)


def test_fallback_importance_x_deviation():
    """Verify fallback explanation logic computes expected contributions."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, 4))
    y = (x[:, 0] > 0).astype(int)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(x, y)

    feature_names = ["feat_0", "feat_1", "feat_2", "feat_3"]
    feature_stats = {
        name: {"mean": 0.0, "std": 1.0} for name in feature_names
    }

    # Wallet with extreme value on feat_0
    wallet_vec = np.array([5.0, 0.0, -1.0, 0.5], dtype=np.float32)

    top_features = explain_fallback_importance_x_deviation(
        clf=clf,
        feature_vector=wallet_vec,
        feature_names=feature_names,
        feature_stats=feature_stats,
        top_k=3,
    )

    assert len(top_features) == 3
    # Check dictionary keys
    for item in top_features:
        assert "name" in item
        assert "value" in item
        assert "contribution" in item
        assert isinstance(item["contribution"], float)

    # Values should be sorted descending by absolute contribution
    abs_contribs = [abs(item["contribution"]) for item in top_features]
    assert abs_contribs == sorted(abs_contribs, reverse=True)


def test_explain_wallet_fallback_and_labeling():
    """Verify explain_wallet produces valid structure and does not mislabel fallback."""
    rng = np.random.default_rng(99)
    x = rng.normal(size=(30, 6))
    y = rng.choice([0, 1], size=30)

    clf = RandomForestClassifier(n_estimators=5, random_state=99)
    clf.fit(x, y)

    feature_names = [f"f_{i}" for i in range(6)]
    wallet_dict = {f"f_{i}": float(i * 2.0) for i in range(6)}

    result = explain_wallet(
        clf=clf,
        wallet_features=wallet_dict,
        feature_names=feature_names,
        feature_stats=None,
        top_k=5,
    )

    assert "top_features" in result
    assert "explanation_method" in result
    assert result["explanation_method"] in ["shap", "importance_x_deviation"]
    assert len(result["top_features"]) == 5

    # If shap is not installed, it must be importance_x_deviation
    import sys
    if "shap" not in sys.modules:
        assert result["explanation_method"] == "importance_x_deviation"

