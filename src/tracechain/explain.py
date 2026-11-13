"""Per-wallet feature explanation utilities.

Implements SPEC section 4.5:
- SHAP TreeExplainer for the illicit class on single-wallet feature vectors.
- Fallback to feature importance times deviation (importance_x_deviation) when SHAP is
  unavailable, fails, or exceeds the time budget.
- Never labels fallback explanations as SHAP.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd


_EXPLAINER_CACHE: dict[int, Any] = {}


def _get_shap_module() -> Any | None:
    """Safely attempt to import the shap library."""
    try:
        import shap  # type: ignore[import-untyped]
        return shap
    except (ImportError, Exception):
        return None


def explain_fallback_importance_x_deviation(
    clf: Any,
    feature_vector: np.ndarray,
    feature_names: list[str],
    feature_stats: dict[str, dict[str, float]] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Compute top feature contributions using feature importance multiplied by z-score deviation.

    Args:
        clf: Trained tree-based model with feature_importances_ attribute.
        feature_vector: 1D array of feature values for the wallet.
        feature_names: List of feature names matching feature_vector order.
        feature_stats: Precomputed training mean and std per feature.
        top_k: Number of top features to return.

    Returns:
        List of dicts: [{'name': str, 'value': float, 'contribution': float}, ...]
    """
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        importances = np.ones(len(feature_names), dtype=np.float32) / max(len(feature_names), 1)

    scored_features: list[tuple[float, str, float, float]] = []

    for i, name in enumerate(feature_names):
        val = float(feature_vector[i])
        imp = float(importances[i])

        if feature_stats and name in feature_stats:
            mean_val = feature_stats[name].get("mean", 0.0)
            std_val = feature_stats[name].get("std", 1.0)
            if std_val <= 1e-8:
                std_val = 1.0
            dev = (val - mean_val) / std_val
        else:
            dev = val

        contribution = imp * dev
        scored_features.append((abs(contribution), name, val, contribution))

    # Sort descending by absolute contribution magnitude
    scored_features.sort(key=lambda item: item[0], reverse=True)

    top_features = [
        {
            "name": name,
            "value": round(val, 4),
            "contribution": round(contrib, 4),
        }
        for _, name, val, contrib in scored_features[:top_k]
    ]

    return top_features


def explain_wallet(
    clf: Any,
    wallet_features: np.ndarray | pd.Series | dict[str, float],
    feature_names: list[str],
    feature_stats: dict[str, dict[str, float]] | None = None,
    top_k: int = 5,
    max_time_seconds: float = 2.0,
) -> dict[str, Any]:
    """Generate per-wallet feature explanation for the illicit class.

    Attempts SHAP TreeExplainer first. If SHAP is unavailable, fails, or exceeds
    max_time_seconds, falls back to importance_x_deviation.

    Args:
        clf: Trained classifier.
        wallet_features: Single-wallet features (array, Series, or dict).
        feature_names: Feature names in the order expected by the model.
        feature_stats: Precomputed training statistics (mean, std) for fallback.
        top_k: Number of top features to return (default: 5).
        max_time_seconds: Maximum time allowed for SHAP before fallback.

    Returns:
        dict with 'top_features' and 'explanation_method'.
    """
    # Convert wallet_features to 1D float array
    if isinstance(wallet_features, dict):
        feat_vec = np.array([wallet_features.get(f, 0.0) for f in feature_names], dtype=np.float32)
    elif isinstance(wallet_features, pd.Series):
        feat_vec = np.array([wallet_features.get(f, 0.0) for f in feature_names], dtype=np.float32)
    else:
        feat_vec = np.asarray(wallet_features, dtype=np.float32).flatten()

    shap_lib = _get_shap_module()

    if shap_lib is not None:
        t0 = time.perf_counter()
        try:
            clf_id = id(clf)
            if clf_id not in _EXPLAINER_CACHE:
                _EXPLAINER_CACHE[clf_id] = shap_lib.TreeExplainer(clf)
            explainer = _EXPLAINER_CACHE[clf_id]

            # Explain single 2D row
            row_2d = feat_vec.reshape(1, -1)
            shap_values = explainer.shap_values(row_2d)

            # Handle SHAP multi-class output shapes
            if isinstance(shap_values, list):
                # Binary classification: index 1 is illicit
                val_arr = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # Shape: (1, n_features, n_classes)
                val_arr = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
            elif hasattr(shap_values, "values"):
                # shap.Explanation object
                val_arr = shap_values.values[0, :, 1] if shap_values.values.ndim == 3 else shap_values.values[0]
            else:
                val_arr = np.asarray(shap_values).flatten()

            elapsed = time.perf_counter() - t0
            if elapsed <= max_time_seconds and len(val_arr) == len(feature_names):
                scored = [
                    (abs(float(val_arr[i])), feature_names[i], float(feat_vec[i]), float(val_arr[i]))
                    for i in range(len(feature_names))
                ]
                scored.sort(key=lambda item: item[0], reverse=True)
                top_features = [
                    {
                        "name": name,
                        "value": round(val, 4),
                        "contribution": round(contrib, 4),
                    }
                    for _, name, val, contrib in scored[:top_k]
                ]
                return {
                    "top_features": top_features,
                    "explanation_method": "shap",
                }
        except Exception:
            # SHAP computation failed or exceeded; fall back gracefully
            pass

    # Fallback explanation
    fallback_features = explain_fallback_importance_x_deviation(
        clf=clf,
        feature_vector=feat_vec,
        feature_names=feature_names,
        feature_stats=feature_stats,
        top_k=top_k,
    )
    return {
        "top_features": fallback_features,
        "explanation_method": "importance_x_deviation",
    }
