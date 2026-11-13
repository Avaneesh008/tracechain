"""Train baseline and graph-enhanced Random Forest models, evaluate, plot figures, and explain samples.

Implements SPEC section 4.4 and 4.5:
- Trains baseline Random Forest on original 55 features.
- Trains graph-enhanced Random Forest on original 55 features + 4 graph features + 1 hop distance (60 features).
- Evaluates illicit class metrics (precision, recall, F1, PR-AUC, confusion matrix) on all test rows and cold subset.
- Saves models, feature lists, and feature statistics under artifacts/.
- Generates comparative precision-recall curve, confusion matrices, and top 15 feature importance plots in reports/figures/.
- Saves reports/metrics.json with both 'baseline' and 'graph_enhanced' metrics.
- Tests explanations on 3 real test addresses and reports runtimes.
- Triggers scripts/04_make_results.py to produce reports/RESULTS.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from tracechain.data import FEATURE_COLUMNS
from tracechain.explain import explain_wallet
from tracechain.model import (
    compute_feature_stats,
    train_and_evaluate_split,
)


ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
GRAPH_FEATURE_COLS = ["in_degree", "out_degree", "pagerank", "mean_neighbor_degree"]
VASP_HOP_COL = "hops"


def plot_comparative_pr_curves(
    clf_base,
    clf_graph,
    test_df: pd.DataFrame,
    cold_test_df: pd.DataFrame,
    base_cols: list[str],
    graph_cols: list[str],
    output_path: Path,
) -> None:
    """Plot comparative Precision-Recall curves for baseline vs graph-enhanced models."""
    plt.figure(figsize=(9, 6.5))

    # Baseline curves
    p_b_all = clf_base.predict_proba(test_df[base_cols])[:, 1]
    prec_b_all, rec_b_all, _ = precision_recall_curve(test_df["label"], p_b_all, pos_label=1)
    auc_b_all = auc(rec_b_all, prec_b_all)
    plt.plot(rec_b_all, prec_b_all, label=f"Baseline (All Test, AUC = {auc_b_all:.4f})", color="#7f7f7f", linestyle=":", linewidth=2)

    p_b_cold = clf_base.predict_proba(cold_test_df[base_cols])[:, 1]
    prec_b_cold, rec_b_cold, _ = precision_recall_curve(cold_test_df["label"], p_b_cold, pos_label=1)
    auc_b_cold = auc(rec_b_cold, prec_b_cold)
    plt.plot(rec_b_cold, prec_b_cold, label=f"Baseline (Cold Test, AUC = {auc_b_cold:.4f})", color="#c7c7c7", linestyle=":", linewidth=2)

    # Graph-enhanced curves
    p_g_all = clf_graph.predict_proba(test_df[graph_cols])[:, 1]
    prec_g_all, rec_g_all, _ = precision_recall_curve(test_df["label"], p_g_all, pos_label=1)
    auc_g_all = auc(rec_g_all, prec_g_all)
    plt.plot(rec_g_all, prec_g_all, label=f"Graph-Enhanced (All Test, AUC = {auc_g_all:.4f})", color="#1f77b4", linewidth=2.5)

    p_g_cold = clf_graph.predict_proba(cold_test_df[graph_cols])[:, 1]
    prec_g_cold, rec_g_cold, _ = precision_recall_curve(cold_test_df["label"], p_g_cold, pos_label=1)
    auc_g_cold = auc(rec_g_cold, prec_g_cold)
    plt.plot(rec_g_cold, prec_g_cold, label=f"Graph-Enhanced (Cold Test, AUC = {auc_g_cold:.4f})", color="#ff7f0e", linestyle="--", linewidth=2.5)

    plt.xlabel("Recall (Illicit)", fontsize=12)
    plt.ylabel("Precision (Illicit)", fontsize=12)
    plt.title("Precision-Recall Curve: Baseline vs Graph-Enhanced Random Forest", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrices(
    metrics_dict: dict,
    output_path: Path,
    title_prefix: str = "Graph-Enhanced",
) -> None:
    """Plot confusion matrices for all test and cold test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, split_key, title in [
        (axes[0], "test_all", f"{title_prefix} - All Test"),
        (axes[1], "test_cold", f"{title_prefix} - Cold Test"),
    ]:
        cm_dict = metrics_dict[split_key]["confusion_matrix"]
        cm_array = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])

        ax.matshow(cm_array, cmap=plt.cm.Blues, alpha=0.8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_array[i, j]:,}", ha="center", va="center", fontsize=12, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Licit", "Pred Illicit"], fontsize=10)
        ax.set_yticklabels(["True Licit", "True Illicit"], fontsize=10)
        ax.set_title(title, fontsize=12, pad=15, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_importances(
    clf,
    feature_cols: list[str],
    output_path: Path,
    top_n: int = 15,
    title: str = "Top 15 Feature Importances",
) -> None:
    """Plot top N feature importances as a horizontal bar chart."""
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:top_n]

    top_names = [feature_cols[i] for i in sorted_idx][::-1]
    top_scores = importances[sorted_idx][::-1]

    plt.figure(figsize=(10, 6.5))
    colors = ["#d62728" if name in GRAPH_FEATURE_COLS or name == VASP_HOP_COL else "#1f77b4" for name in top_names]
    plt.barh(range(len(top_names)), top_scores, color=colors, align="center")
    plt.yticks(range(len(top_names)), top_names, fontsize=10)
    plt.xlabel("Gini Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> int:
    """Train baseline and graph-enhanced models, evaluate, generate figures and results."""
    train_path = ARTIFACTS_DIR / "train.parquet"
    test_path = ARTIFACTS_DIR / "test.parquet"
    cold_path = ARTIFACTS_DIR / "cold_test.parquet"
    vasp_index_path = ARTIFACTS_DIR / "vasp_index.parquet"

    for p in [train_path, test_path, cold_path]:
        if not p.is_file():
            print(f"ERROR: Missing artifact {p}. Run scripts/01_prepare.py first.")
            return 1

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TraceChain v2: Baseline & Graph-Enhanced Model Training (SPEC 4.4 & 4.5)")
    print("=" * 70)

    # 1. Load prepared splits
    print("\n[1/7] Loading prepared dataset splits from artifacts/...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    cold_test_df = pd.read_parquet(cold_path)

    # Merge hop distance from vasp_index.parquet if available
    has_vasp = vasp_index_path.is_file()
    if has_vasp:
        print("  Found artifacts/vasp_index.parquet. Merging hop distances...")
        vasp_hops = pd.read_parquet(vasp_index_path)[["address", "hops"]]
        train_df = train_df.merge(vasp_hops, on="address", how="left")
        test_df = test_df.merge(vasp_hops, on="address", how="left")
        cold_test_df = cold_test_df.merge(vasp_hops, on="address", how="left")
        train_df["hops"] = train_df["hops"].fillna(-1).astype(np.float32)
        test_df["hops"] = test_df["hops"].fillna(-1).astype(np.float32)
        cold_test_df["hops"] = cold_test_df["hops"].fillna(-1).astype(np.float32)

    base_feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    graph_feature_cols = list(base_feature_cols)
    for g_col in GRAPH_FEATURE_COLS:
        if g_col in train_df.columns and g_col not in graph_feature_cols:
            graph_feature_cols.append(g_col)
    if has_vasp and "hops" in train_df.columns and "hops" not in graph_feature_cols:
        graph_feature_cols.append("hops")

    print(f"  Train rows: {len(train_df):,} | Test rows: {len(test_df):,} | Cold rows: {len(cold_test_df):,}")
    print(f"  Baseline features: {len(base_feature_cols)} (original features)")
    print(f"  Graph-enhanced features: {len(graph_feature_cols)} (original + graph metrics + hops)")

    # 2. Compute training feature stats for fallback explanation
    print("\n[2/7] Computing training feature statistics (mean, std) for explanations...")
    feature_stats = compute_feature_stats(train_df, graph_feature_cols)
    stats_path = ARTIFACTS_DIR / "feature_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(feature_stats, f, indent=2)
    print(f"  Saved feature statistics for all {len(graph_feature_cols)} features to {stats_path}")

    # 3. Train Baseline Random Forest
    print("\n[3/7] Training baseline Random Forest (n_estimators=200, min_samples_leaf=5)...")
    t0 = time.perf_counter()
    clf_base, base_metrics = train_and_evaluate_split(
        train_df=train_df,
        test_df=test_df,
        cold_test_df=cold_test_df,
        feature_cols=base_feature_cols,
    )
    t_base = time.perf_counter() - t0
    print(f"  Baseline training completed in {t_base:.2f} seconds.")

    # 4. Train Graph-Enhanced Random Forest
    print("\n[4/7] Training graph-enhanced Random Forest (n_estimators=200, min_samples_leaf=5)...")
    t1 = time.perf_counter()
    clf_graph, graph_metrics = train_and_evaluate_split(
        train_df=train_df,
        test_df=test_df,
        cold_test_df=cold_test_df,
        feature_cols=graph_feature_cols,
    )
    t_graph = time.perf_counter() - t1
    print(f"  Graph-enhanced training completed in {t_graph:.2f} seconds.")

    # 5. Save models and feature lists
    print("\n[5/7] Saving models and artifacts under artifacts/...")
    base_model_path = ARTIFACTS_DIR / "baseline_model.joblib"
    joblib.dump(clf_base, base_model_path, compress=3)
    base_size_mb = base_model_path.stat().st_size / (1024 * 1024)

    graph_model_path = ARTIFACTS_DIR / "graph_enhanced_model.joblib"
    joblib.dump(clf_graph, graph_model_path, compress=3)
    graph_size_mb = graph_model_path.stat().st_size / (1024 * 1024)

    # Also save model.joblib pointing to graph_enhanced_model for serving
    serving_model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(clf_graph, serving_model_path, compress=3)

    with (ARTIFACTS_DIR / "baseline_features.json").open("w", encoding="utf-8") as f:
        json.dump(base_feature_cols, f, indent=2)

    with (ARTIFACTS_DIR / "graph_enhanced_features.json").open("w", encoding="utf-8") as f:
        json.dump(graph_feature_cols, f, indent=2)

    with (ARTIFACTS_DIR / "features.json").open("w", encoding="utf-8") as f:
        json.dump(graph_feature_cols, f, indent=2)

    print(f"  Saved baseline model ({base_size_mb:.2f} MB) and graph-enhanced model ({graph_size_mb:.2f} MB)")

    metrics_path = REPORTS_DIR / "metrics.json"
    full_metrics = {
        "baseline": base_metrics,
        "graph_enhanced": graph_metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"  Saved full metrics to {metrics_path}")

    # 6. Generate comparative figures
    print("\n[6/7] Generating evaluation figures in reports/figures/...")
    pr_comparison_path = FIGURES_DIR / "pr_curve_comparison.png"
    cm_base_path = FIGURES_DIR / "confusion_matrix_baseline.png"
    cm_graph_path = FIGURES_DIR / "confusion_matrix_graph_enhanced.png"
    feat_imp_graph_path = FIGURES_DIR / "feature_importances_graph_enhanced.png"
    feat_imp_base_path = FIGURES_DIR / "feature_importances_baseline.png"

    plot_comparative_pr_curves(
        clf_base,
        clf_graph,
        test_df,
        cold_test_df,
        base_feature_cols,
        graph_feature_cols,
        pr_comparison_path,
    )
    print(f"  Saved comparative PR curves to {pr_comparison_path}")

    plot_confusion_matrices(base_metrics, cm_base_path, title_prefix="Baseline")
    plot_confusion_matrices(graph_metrics, cm_graph_path, title_prefix="Graph-Enhanced")
    print(f"  Saved confusion matrices to {cm_base_path} and {cm_graph_path}")

    plot_feature_importances(clf_base, base_feature_cols, feat_imp_base_path, top_n=15, title="Top 15 Feature Importances (Baseline)")
    plot_feature_importances(clf_graph, graph_feature_cols, feat_imp_graph_path, top_n=15, title="Top 15 Feature Importances (Graph-Enhanced, Red=Graph/VASP)")
    print(f"  Saved feature importances to {feat_imp_graph_path}")

    # 7. Benchmark explanations on 3 real test addresses with graph-enhanced model
    print("\n[7/7] Testing explanation function on 3 real test addresses (Graph-Enhanced model)...")
    illicit_samples = test_df[test_df["label"] == 1]
    licit_samples = test_df[test_df["label"] == 0]

    sample_addrs: list[tuple[str, int, str]] = []
    if len(illicit_samples) > 0:
        row1 = illicit_samples.iloc[0]
        sample_addrs.append((str(row1["address"]), int(row1["label"]), "illicit"))
    if len(illicit_samples) > 1:
        row2 = illicit_samples.iloc[1]
        sample_addrs.append((str(row2["address"]), int(row2["label"]), "illicit"))
    if len(licit_samples) > 0:
        row3 = licit_samples.iloc[0]
        sample_addrs.append((str(row3["address"]), int(row3["label"]), "licit"))

    explanation_benchmark: list[dict] = []
    for addr, label, label_name in sample_addrs:
        addr_row = test_df[test_df["address"] == addr].iloc[0]
        t_exp = time.perf_counter()
        expl = explain_wallet(
            clf=clf_graph,
            wallet_features=addr_row,
            feature_names=graph_feature_cols,
            feature_stats=feature_stats,
            top_k=5,
        )
        elapsed_ms = (time.perf_counter() - t_exp) * 1000.0
        explanation_benchmark.append(
            {
                "address": addr,
                "label": label_name,
                "elapsed_ms": round(elapsed_ms, 2),
                "method": expl["explanation_method"],
                "top_features": expl["top_features"],
            }
        )

    # 8. Generate reports/RESULTS.md
    subprocess.run([sys.executable, "scripts/04_make_results.py"], check=True)

    # 9. Print comprehensive summary
    print("\n" + "=" * 70)
    print("MODEL EVALUATION SUMMARY: BASELINE VS GRAPH-ENHANCED")
    print("=" * 70)
    print(f"Baseline model file size:       {base_size_mb:.2f} MB")
    print(f"Graph-enhanced model file size: {graph_size_mb:.2f} MB")

    b_all = base_metrics["test_all"]
    b_cold = base_metrics["test_cold"]
    g_all = graph_metrics["test_all"]
    g_cold = graph_metrics["test_cold"]

    print("\n--- Performance Comparison (Illicit Class, Threshold = 0.5) ---")
    header = f"{'Metric':<18} {'Base (All)':<14} {'Base (Cold)':<14} {'Graph (All)':<14} {'Graph (Cold)':<14} {'Delta (All)':<12}"
    print(header)
    print("-" * len(header))
    print(f"{'Precision':<18} {b_all['precision']:<14.4f} {b_cold['precision']:<14.4f} {g_all['precision']:<14.4f} {g_cold['precision']:<14.4f} {g_all['precision'] - b_all['precision']:+12.4f}")
    print(f"{'Recall':<18} {b_all['recall']:<14.4f} {b_cold['recall']:<14.4f} {g_all['recall']:<14.4f} {g_cold['recall']:<14.4f} {g_all['recall'] - b_all['recall']:+12.4f}")
    print(f"{'F1 Score':<18} {b_all['f1']:<14.4f} {b_cold['f1']:<14.4f} {g_all['f1']:<14.4f} {g_cold['f1']:<14.4f} {g_all['f1'] - b_all['f1']:+12.4f}")
    print(f"{'PR-AUC':<18} {b_all['pr_auc']:<14.4f} {b_cold['pr_auc']:<14.4f} {g_all['pr_auc']:<14.4f} {g_cold['pr_auc']:<14.4f} {g_all['pr_auc'] - b_all['pr_auc']:+12.4f}")

    print("\n--- Confusion Matrix (All Test: 97,756 rows) ---")
    print(f"  Baseline:       TP={b_all['confusion_matrix']['tp']:,}, FP={b_all['confusion_matrix']['fp']:,}, FN={b_all['confusion_matrix']['fn']:,}, TN={b_all['confusion_matrix']['tn']:,}")
    print(f"  Graph-Enhanced: TP={g_all['confusion_matrix']['tp']:,}, FP={g_all['confusion_matrix']['fp']:,}, FN={g_all['confusion_matrix']['fn']:,}, TN={g_all['confusion_matrix']['tn']:,}")

    print("\n--- Per-Wallet Explanation Benchmark (Graph-Enhanced Model) ---")
    for item in explanation_benchmark:
        print(f"\nAddress: {item['address']} ({item['label']})")
        print(f"Method:  {item['method']} | Time: {item['elapsed_ms']:.2f} ms")
        print("Top 5 contributing features:")
        for feat in item["top_features"]:
            print(f"  - {feat['name']:<30}: value = {feat['value']:<10} contribution = {feat['contribution']:+.4f}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
