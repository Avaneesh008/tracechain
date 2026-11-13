"""Risk classification model training and evaluation.

Implements SPEC section 4.4:
- Random Forest model training with exact spec hyperparameters.
- Evaluation metrics for the illicit class (precision, recall, f1, PR-AUC, confusion matrix).
- Modular design to support baseline and graph-enhanced models.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "min_samples_leaf": 5,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": 42,
}


def train_rf(
    x_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    params: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """Train a Random Forest classifier using the spec parameters.

    Args:
        x_train: Training feature matrix.
        y_train: Training labels (0 for licit, 1 for illicit).
        params: Optional hyperparameter overrides.

    Returns:
        Fitted RandomForestClassifier instance.
    """
    config = dict(DEFAULT_RF_PARAMS)
    if params:
        config.update(params)

    clf = RandomForestClassifier(**config)
    clf.fit(x_train, y_train)
    return clf


def evaluate_model(
    clf: RandomForestClassifier,
    x_eval: np.ndarray | pd.DataFrame,
    y_eval: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate classifier performance for the illicit class (label 1).

    Computes precision, recall, f1, PR-AUC, and the confusion matrix.

    Args:
        clf: Trained classifier.
        x_eval: Evaluation feature matrix.
        y_eval: True evaluation labels (0 or 1).
        threshold: Decision threshold for classification (default: 0.5).

    Returns:
        dict containing counts, precision, recall, f1, pr_auc, and confusion matrix.
    """
    y_true = np.asarray(y_eval, dtype=np.int8)

    # Probabilities for illicit class (index 1)
    # Handle edge case where only one class was present in training
    if len(clf.classes_) == 1:
        single_cls = clf.classes_[0]
        probs = np.ones(len(y_true), dtype=np.float32) if single_cls == 1 else np.zeros(len(y_true), dtype=np.float32)
    else:
        probs = clf.predict_proba(x_eval)[:, 1]

    preds = (probs >= threshold).astype(np.int8)

    prec = float(precision_score(y_true, preds, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, probs, pos_label=1)
    pr_auc_val = float(auc(rec_curve, prec_curve))

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "rows": int(len(y_true)),
        "illicit_count": int(np.sum(y_true == 1)),
        "licit_count": int(np.sum(y_true == 0)),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "pr_auc": round(pr_auc_val, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def compute_feature_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, dict[str, float]]:
    """Compute mean and standard deviation of each feature for explanations.

    Args:
        df: DataFrame containing the training features.
        feature_cols: List of feature names.

    Returns:
        dict mapping feature name to {'mean': float, 'std': float}.
    """
    stats: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        col_mean = float(df[col].mean())
        col_std = float(df[col].std(ddof=0))
        stats[col] = {
            "mean": round(col_mean, 6),
            "std": round(col_std, 6),
        }
    return stats


def train_and_evaluate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cold_test_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any] | None = None,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Train a Random Forest model on train_df and evaluate on test_df and cold_test_df.

    Args:
        train_df: Training DataFrame containing feature_cols and 'label'.
        test_df: Full test DataFrame.
        cold_test_df: Cold test subset DataFrame.
        feature_cols: List of feature column names to train on.
        params: Optional hyperparameter overrides.

    Returns:
        tuple of (trained classifier, metrics dictionary).
    """
    x_train = train_df[feature_cols]
    y_train = train_df["label"]

    clf = train_rf(x_train, y_train, params=params)

    test_all_metrics = evaluate_model(clf, test_df[feature_cols], test_df["label"])
    test_cold_metrics = evaluate_model(clf, cold_test_df[feature_cols], cold_test_df["label"])

    model_config = dict(DEFAULT_RF_PARAMS)
    if params:
        model_config.update(params)

    metrics_entry: dict[str, Any] = {
        "model_type": "RandomForestClassifier",
        "params": {
            "n_estimators": model_config["n_estimators"],
            "min_samples_leaf": model_config["min_samples_leaf"],
            "class_weight": model_config["class_weight"],
            "random_state": model_config["random_state"],
        },
        "features_count": len(feature_cols),
        "feature_names": feature_cols,
        "threshold": 0.5,
        "test_all": test_all_metrics,
        "test_cold": test_cold_metrics,
    }

    return clf, metrics_entry
