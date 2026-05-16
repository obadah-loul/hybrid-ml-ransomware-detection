"""
utils/evaluation.py
====================
Shared evaluation and output-generation library.

Produces all Section 5 evidence artifacts:
  - Metrics summary dict  (returned to caller)
  - Classification report → CSV
  - Confusion matrix      → PNG
  - Feature importance    → PNG  (RF built-in + permutation fallback)
  - Comparison chart      → PNG  (called by compare_models.py)

All save paths are passed in by the caller — this module has no hardcoded paths.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.inspection import permutation_importance


# ─────────────────────────────────────────────────────────────────────────────
# 1. Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    model_name: str,
                    dataset_name: str) -> dict:
    """
    Compute a standard metric set and return as a flat dict.
    All average modes are macro (class-balanced) for academic fairness.
    Weighted F1 is also included for completeness.

    Returns
    -------
    dict with keys: model, dataset, accuracy, precision_macro,
                    recall_macro, f1_macro, f1_weighted
    """
    return {
        "model":          model_name,
        "dataset":        dataset_name,
        "accuracy":       round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro",
                                                  zero_division=0), 4),
        "recall_macro":   round(recall_score(y_true, y_pred, average="macro",
                                              zero_division=0), 4),
        "f1_macro":       round(f1_score(y_true, y_pred, average="macro",
                                          zero_division=0), 4),
        "f1_weighted":    round(f1_score(y_true, y_pred, average="weighted",
                                          zero_division=0), 4),
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dict to terminal."""
    print(f"\n  {'─'*44}")
    print(f"  Model:           {metrics['model']}")
    print(f"  Dataset:         {metrics['dataset']}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Precision (mac): {metrics['precision_macro']:.4f}")
    print(f"  Recall (mac):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  {'─'*44}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Classification report → CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_classification_report(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                class_names: list,
                                save_path: str,
                                model_name: str,
                                dataset_name: str) -> pd.DataFrame:
    """
    Generate sklearn classification report as a CSV table.
    Columns: class, precision, recall, f1-score, support

    Saved to save_path. Returns the DataFrame for optional further use.
    """
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    # Convert to DataFrame; drop accuracy/macro avg/weighted avg rows into
    # separate rows at the bottom for clean formatting
    rows = []
    for cls in class_names:
        if cls in report_dict:
            r = report_dict[cls]
            rows.append({
                "class":     cls,
                "precision": round(r["precision"], 4),
                "recall":    round(r["recall"], 4),
                "f1_score":  round(r["f1-score"], 4),
                "support":   int(r["support"]),
            })

    # Append summary rows
    for key in ["macro avg", "weighted avg"]:
        if key in report_dict:
            r = report_dict[key]
            rows.append({
                "class":     key,
                "precision": round(r["precision"], 4),
                "recall":    round(r["recall"], 4),
                "f1_score":  round(r["f1-score"], 4),
                "support":   int(r["support"]),
            })

    df = pd.DataFrame(rows)
    df.insert(0, "model", model_name)
    df.insert(1, "dataset", dataset_name)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  [eval] Classification report saved → {save_path}")

    # Also print to terminal
    print(f"\n  Classification Report — {model_name} on {dataset_name}")
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    ))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confusion matrix → PNG
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           class_names: list,
                           save_path: str,
                           title: str = "Confusion Matrix") -> None:
    """
    Save a normalised confusion matrix heatmap as PNG.

    Normalised by true label (rows sum to 1.0) so class-imbalance does not
    make the matrix unreadable. Raw counts are annotated inside each cell.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig_size = max(8, n * 1.1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.8},
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [eval] Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature importance → PNG
# ─────────────────────────────────────────────────────────────────────────────

def save_feature_importance(model,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              feature_names: list,
                              save_path: str,
                              title: str = "Feature Importance",
                              top_n: int = 20,
                              use_permutation: bool = False) -> pd.DataFrame:
    """
    Save a horizontal bar chart of feature importances as PNG.

    Strategy:
      - For RandomForest or XGBoost: use built-in feature_importances_ (fast).
      - If use_permutation=True or model has no built-in importances:
        use permutation importance on X_test (model-agnostic, slower).
      - For the Hybrid VotingClassifier, extract RF's built-in importances
        (RF is always estimator[0] in our pipeline).

    Returns a DataFrame of importance values (top_n rows).
    """
    importances = None
    method_used = "unknown"

    # ── Try built-in importance ───────────────────────────────────────────
    if not use_permutation:
        # Direct model with feature_importances_
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            method_used = "built-in (Gini impurity)"

        # VotingClassifier — extract RF sub-estimator
        elif hasattr(model, "estimators_"):
            for est in model.estimators_:
                # estimators_ is a list of fitted estimators
                actual_est = est[1] if isinstance(est, tuple) else est
                if hasattr(actual_est, "feature_importances_"):
                    importances = actual_est.feature_importances_
                    method_used = "built-in via RF sub-estimator"
                    break

    # ── Fallback: permutation importance ─────────────────────────────────
    if importances is None or use_permutation:
        print(f"  [importance] Using permutation importance (n_repeats=10)...")
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        importances = result.importances_mean
        method_used = "permutation importance"

    # ── Build DataFrame ───────────────────────────────────────────────────
    imp_df = pd.DataFrame({
        "feature":    feature_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    colors = plt.cm.Blues_r(np.linspace(0.2, 0.85, len(imp_df)))
    bars = ax.barh(
        imp_df["feature"][::-1],
        imp_df["importance"][::-1],
        color=colors,
        edgecolor="none",
        height=0.65,
    )
    ax.set_xlabel(f"Importance score ({method_used})", fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [eval] Feature importance saved → {save_path}  ({method_used})")

    return imp_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Metrics table → CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_table(metrics_list: list, save_path: str) -> pd.DataFrame:
    """
    Save a list of metrics dicts as a CSV table.
    metrics_list : list of dicts from compute_metrics()
    """
    df = pd.DataFrame(metrics_list)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  [eval] Metrics table saved → {save_path}")

    # Terminal display
    print(f"\n  {'─'*80}")
    print(df.to_string(index=False))
    print(f"  {'─'*80}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Model comparison chart → PNG
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_chart(df_metrics: pd.DataFrame,
                           save_path: str,
                           metric: str = "f1_macro",
                           title: str = "Model Comparison") -> None:
    """
    Grouped bar chart: models on x-axis, datasets as hue, metric on y-axis.

    df_metrics must have columns: model, dataset, <metric>
    """
    datasets = df_metrics["dataset"].unique()
    models   = df_metrics["model"].unique()

    n_models  = len(models)
    n_datasets = len(datasets)
    bar_width = 0.7 / n_datasets
    x = np.arange(n_models)

    palette = ["#2166AC", "#D6604D", "#4DAC26", "#762A83"][:n_datasets]

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.8), 5.5))

    for i, (dataset, color) in enumerate(zip(datasets, palette)):
        subset = df_metrics[df_metrics["dataset"] == dataset].set_index("model")
        values = [subset.loc[m, metric] if m in subset.index else 0 for m in models]
        offset = (i - n_datasets / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, values,
            width=bar_width * 0.9,
            label=dataset,
            color=color,
            alpha=0.87,
            edgecolor="white",
            linewidth=0.5,
        )
        # Annotate values on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylim(0, min(1.05, df_metrics[metric].max() + 0.12))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(title="Dataset", fontsize=9, title_fontsize=9, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [eval] Comparison chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full evaluation runner (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(model,
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         class_names: list,
                         feature_names: list,
                         model_name: str,
                         dataset_name: str,
                         output_dir: str) -> dict:
    """
    Run all evaluation steps for a single (model, dataset) pair.

    Saves:
      {output_dir}/confusion_matrix_{model_name}.png
      {output_dir}/classification_report_{model_name}.csv
      {output_dir}/feature_importance_{model_name}.png  (if possible)

    Returns the metrics dict.
    """
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_pred=y_pred, y_true=y_test,
                               model_name=model_name, dataset_name=dataset_name)
    print_metrics(metrics)

    safe_name = model_name.lower().replace(" ", "_")

    save_confusion_matrix(
        y_true=y_test, y_pred=y_pred,
        class_names=class_names,
        save_path=os.path.join(output_dir, f"confusion_matrix_{safe_name}.png"),
        title=f"Confusion Matrix — {model_name} on {dataset_name}",
    )

    save_classification_report(
        y_true=y_test, y_pred=y_pred,
        class_names=class_names,
        save_path=os.path.join(output_dir, f"report_{safe_name}.csv"),
        model_name=model_name,
        dataset_name=dataset_name,
    )

    try:
        save_feature_importance(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            save_path=os.path.join(output_dir, f"feature_importance_{safe_name}.png"),
            title=f"Top Features — {model_name} on {dataset_name}",
        )
    except Exception as e:
        print(f"  [eval] Feature importance skipped for {model_name}: {e}")

    return metrics
