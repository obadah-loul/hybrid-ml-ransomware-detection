"""
scripts/compare_models.py
==========================
Loads saved metrics CSVs from both pipelines and produces:

  results/evidence_package/
    ├── comparison_table_all_models.csv   ← Table 3 / Table 4 in paper
    ├── comparison_chart_f1_macro.png     ← Figure for Section 5
    ├── comparison_chart_accuracy.png
    ├── best_confusion_matrix.png         ← copied from winning model
    ├── best_per_class_report.csv         ← per-class F1 table
    └── summary_stats.txt                 ← human-readable summary

Run AFTER both pipeline scripts have completed:
  python scripts/cicids_pipeline.py
  python scripts/unsw_pipeline.py
  python scripts/compare_models.py
"""

import os
import sys
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.evaluation import save_comparison_chart

RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
CICIDS_DIR   = os.path.join(RESULTS_DIR, "cicids")
UNSW_DIR     = os.path.join(RESULTS_DIR, "unsw")
EVIDENCE_DIR = os.path.join(RESULTS_DIR, "evidence_package")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics(path: str, dataset_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  WARNING: metrics file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["dataset"] = dataset_name   # ensure consistent dataset label
    return df


def find_best_model(df_all: pd.DataFrame,
                    metric: str = "f1_macro") -> dict:
    """
    Return the row with the highest metric value.
    Returns a dict with keys: model, dataset, and all metric values.
    """
    if df_all.empty:
        return {}
    idx = df_all[metric].idxmax()
    return df_all.loc[idx].to_dict()


def copy_best_artifacts(best: dict, evidence_dir: str) -> None:
    """
    Copy the confusion matrix and per-class report of the best model
    into the evidence package directory.
    """
    model_name = best.get("model", "")
    dataset    = best.get("dataset", "")
    safe_name  = model_name.lower().replace(" ", "_")

    # Determine source directory
    if "cicids" in dataset.lower():
        src_dir = os.path.join(RESULTS_DIR, "cicids")
    else:
        src_dir = os.path.join(RESULTS_DIR, "unsw")

    # Confusion matrix
    cm_src = os.path.join(src_dir, f"confusion_matrix_{safe_name}.png")
    cm_dst = os.path.join(evidence_dir, "best_confusion_matrix.png")
    if os.path.exists(cm_src):
        shutil.copy2(cm_src, cm_dst)
        print(f"  [evidence] Best confusion matrix → {cm_dst}")
    else:
        print(f"  WARNING: {cm_src} not found.")

    # Per-class report
    rpt_src = os.path.join(src_dir, f"report_{safe_name}.csv")
    rpt_dst = os.path.join(evidence_dir, "best_per_class_report.csv")
    if os.path.exists(rpt_src):
        shutil.copy2(rpt_src, rpt_dst)
        print(f"  [evidence] Best per-class report → {rpt_dst}")
    else:
        print(f"  WARNING: {rpt_src} not found.")

    # Feature importance
    fi_src = os.path.join(src_dir, f"feature_importance_{safe_name}.png")
    fi_dst = os.path.join(evidence_dir, "best_feature_importance.png")
    if os.path.exists(fi_src):
        shutil.copy2(fi_src, fi_dst)
        print(f"  [evidence] Best feature importance → {fi_dst}")


def write_summary_txt(df_all: pd.DataFrame,
                       best: dict,
                       save_path: str) -> None:
    """Write a human-readable summary for quick reference."""
    lines = [
        "=" * 70,
        "EVIDENCE PACKAGE SUMMARY",
        "Paper: A Hybrid Machine Learning Approach for Ransomware Detection",
        "       in Enterprise Networks",
        "=" * 70,
        "",
        "DATASETS: CICIDS2018 (enterprise attack families) + UNSW-NB15 (network intrusion)",
        "MODELS:   Random Forest | XGBoost | Soft Voting Hybrid (RF + XGBoost)",
        "SPLIT:    80% train / 20% test | random_state=42 | stratified",
        "BALANCE:  SMOTE applied to training set only (no leakage)",
        "",
        "-" * 70,
        "FULL RESULTS TABLE",
        "-" * 70,
    ]

    if not df_all.empty:
        lines.append(df_all.to_string(index=False))

    lines += [
        "",
        "-" * 70,
        "BEST MODEL OVERALL",
        "-" * 70,
    ]

    if best:
        lines.append(f"  Model:           {best.get('model', 'N/A')}")
        lines.append(f"  Dataset:         {best.get('dataset', 'N/A')}")
        for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
            if k in best:
                lines.append(f"  {k:20s}: {best[k]:.4f}")

    lines += [
        "",
        "-" * 70,
        "HYBRID ADVANTAGE (Soft Voting vs best individual model)",
        "-" * 70,
    ]

    if not df_all.empty:
        for dataset in df_all["dataset"].unique():
            sub = df_all[df_all["dataset"] == dataset].copy()
            hybrid_row = sub[sub["model"] == "Soft Voting Hybrid"]
            rf_row     = sub[sub["model"] == "Random Forest"]
            xgb_row    = sub[sub["model"] == "XGBoost"]

            if hybrid_row.empty:
                continue

            hybrid_f1 = hybrid_row["f1_macro"].values[0]
            best_baseline = max(
                rf_row["f1_macro"].values[0] if not rf_row.empty else 0,
                xgb_row["f1_macro"].values[0] if not xgb_row.empty else 0,
            )
            gain = hybrid_f1 - best_baseline
            lines.append(f"  {dataset}:")
            lines.append(f"    Hybrid F1 (macro): {hybrid_f1:.4f}")
            lines.append(f"    Best baseline F1:  {best_baseline:.4f}")
            lines.append(f"    Improvement:       {gain:+.4f}")
            lines.append("")

    lines.append("=" * 70)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [evidence] Summary -> {save_path}")
    print("\n".join(lines))


def build_multi_metric_chart(df_all: pd.DataFrame,
                              evidence_dir: str) -> None:
    """
    4-panel subplot: accuracy, precision, recall, f1_macro.
    Provides a comprehensive visual for the paper's results section.
    """
    if df_all.empty:
        print("  [chart] No data to plot.")
        return

    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels  = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"]

    datasets = df_all["dataset"].unique()
    models   = df_all["model"].unique()
    n_models = len(models)
    x = np.arange(n_models)
    n_datasets = len(datasets)
    bar_width  = 0.7 / n_datasets
    palette = ["#2166AC", "#D6604D", "#4DAC26", "#762A83"][:n_datasets]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()

    for ax, metric, label in zip(axes, metrics, labels):
        for i, (dataset, color) in enumerate(zip(datasets, palette)):
            sub = df_all[df_all["dataset"] == dataset].set_index("model")
            vals = [sub.loc[m, metric] if m in sub.index else 0 for m in models]
            offset = (i - n_datasets / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, vals,
                width=bar_width * 0.88,
                label=dataset,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.4,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, pad=6)
        min_val = df_all[metric].min() if not df_all.empty else 0
        ax.set_ylim(max(0, min_val - 0.08), min(1.06, df_all[metric].max() + 0.10))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.legend(title="Dataset", fontsize=7, title_fontsize=7, framealpha=0.6)

    fig.suptitle(
        "Model Performance Comparison — CICIDS2018 and UNSW-NB15\n"
        "Hybrid ML Approach for Ransomware Detection in Enterprise Networks",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    out = os.path.join(evidence_dir, "comparison_chart_all_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [evidence] Multi-metric chart → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    print("=" * 70)
    print("Model Comparison & Evidence Package Generator")
    print("=" * 70)

    # ── Load metrics from both pipelines ─────────────────────────────────
    print("\n[1/5] Loading saved metrics...")
    df_cicids = load_metrics(
        os.path.join(CICIDS_DIR, "metrics_cicids.csv"), "CICIDS2018"
    )
    df_unsw = load_metrics(
        os.path.join(UNSW_DIR, "metrics_unsw.csv"), "UNSW-NB15"
    )

    available = [df for df in [df_cicids, df_unsw] if not df.empty]
    if not available:
        print("  ERROR: No metrics files found. Run the pipeline scripts first.")
        print("    python scripts/cicids_pipeline.py")
        print("    python scripts/unsw_pipeline.py")
        return

    df_all = pd.concat(available, ignore_index=True)

    # ── Save unified comparison table ─────────────────────────────────────
    print("\n[2/5] Saving unified comparison table...")
    table_path = os.path.join(EVIDENCE_DIR, "comparison_table_all_models.csv")
    df_all.to_csv(table_path, index=False)
    print(f"  [evidence] Comparison table → {table_path}")
    print(f"\n  {'─'*70}")
    print(df_all.to_string(index=False))
    print(f"  {'─'*70}\n")

    # ── Single-metric comparison charts ──────────────────────────────────
    print("\n[3/5] Generating comparison charts...")
    for metric, label in [("f1_macro", "F1 Macro"), ("accuracy", "Accuracy")]:
        save_comparison_chart(
            df_metrics=df_all,
            save_path=os.path.join(EVIDENCE_DIR, f"comparison_chart_{metric}.png"),
            metric=metric,
            title=f"{label} — Model Comparison (CICIDS2018 vs UNSW-NB15)",
        )

    build_multi_metric_chart(df_all, EVIDENCE_DIR)

    # ── Find and copy best model artifacts ───────────────────────────────
    print("\n[4/5] Identifying best model and copying artifacts...")
    best = find_best_model(df_all, metric="f1_macro")
    if best:
        print(f"\n  Best overall model: {best.get('model')} on {best.get('dataset')}")
        print(f"  F1 (macro): {best.get('f1_macro', 0):.4f}")
        copy_best_artifacts(best, EVIDENCE_DIR)

    # ── Write summary ─────────────────────────────────────────────────────
    print("\n[5/5] Writing summary report...")
    write_summary_txt(
        df_all=df_all,
        best=best,
        save_path=os.path.join(EVIDENCE_DIR, "summary_stats.txt"),
    )

    print("\n" + "=" * 70)
    print("Evidence package complete.")
    print(f"All artifacts → {EVIDENCE_DIR}")
    print("=" * 70)
    print("""
  Evidence package contents:
    comparison_table_all_models.csv  ← paste into paper Table 3/4
    comparison_chart_f1_macro.png    ← Figure for Section 5
    comparison_chart_accuracy.png
    comparison_chart_all_metrics.png ← 4-panel subplot
    best_confusion_matrix.png        ← best model CM
    best_per_class_report.csv        ← per-class F1 table
    best_feature_importance.png      ← top features plot
    summary_stats.txt                ← quick reference
""")


if __name__ == "__main__":
    main()
