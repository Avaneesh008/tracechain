"""Unit tests for model training and evaluation functions."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from tracechain.model import (
    compute_feature_stats,
    evaluate_model,
    train_and_evaluate_split,
    train_rf,
)


def test_train_rf_and_evaluate_model():
    """Test training and evaluation on a small synthetic dataset."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 4

    # Synthetic separable dataset
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    # Class 1 when x[:, 0] > 0
    y = (x[:, 0] > 0).astype(np.int8)

    # Train model with small estimators for speed
    clf = train_rf(x[:70], y[:70], params={"n_estimators": 10, "random_state": 42})
    metrics = evaluate_model(clf, x[70:], y[70:])

    assert metrics["rows"] == 30
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics

    cm = metrics["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 30
    assert metrics["precision"] > 0.6
    assert metrics["recall"] > 0.6


def test_compute_feature_stats():
    """Test computation of feature mean and standard deviation."""
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "f2": [10.0, 10.0, 10.0],
        }
    )
    stats = compute_feature_stats(df, ["f1", "f2"])

    assert "f1" in stats and "f2" in stats
    assert np.isclose(stats["f1"]["mean"], 2.0)
    assert np.isclose(stats["f1"]["std"], np.std([1.0, 2.0, 3.0], ddof=0))
    assert np.isclose(stats["f2"]["mean"], 10.0)
    assert np.isclose(stats["f2"]["std"], 0.0)


def test_train_and_evaluate_split():
    """Test train_and_evaluate_split on mock dataframes."""
    rng = np.random.default_rng(123)
    features = ["feat_a", "feat_b"]

    train_df = pd.DataFrame(
        {
            "feat_a": rng.normal(size=50),
            "feat_b": rng.normal(size=50),
            "label": rng.choice([0, 1], size=50),
        }
    )
    test_df = pd.DataFrame(
        {
            "feat_a": rng.normal(size=20),
            "feat_b": rng.normal(size=20),
            "label": rng.choice([0, 1], size=20),
        }
    )
    cold_df = test_df.iloc[:10].copy()

    clf, metrics = train_and_evaluate_split(
        train_df,
        test_df,
        cold_df,
        features,
        params={"n_estimators": 5, "random_state": 42},
    )

    assert hasattr(clf, "predict")
    assert metrics["model_type"] == "RandomForestClassifier"
    assert "test_all" in metrics
    assert "test_cold" in metrics
    assert metrics["test_all"]["rows"] == 20
    assert metrics["test_cold"]["rows"] == 10

