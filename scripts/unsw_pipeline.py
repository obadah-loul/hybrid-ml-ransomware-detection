"""
scripts/unsw_pipeline.py
=========================
Full training and evaluation pipeline for UNSW-NB15.

Research context
----------------
UNSW-NB15 is a modern network intrusion dataset with 9 attack categories
plus Normal traffic. It serves as the second dataset in our paper to
demonstrate cross-dataset generalisability of the hybrid RF + XGBoost
soft voting approach.

Attack categories in UNSW-NB15:
  Normal, Fuzzers, Analysis, Backdoors, DoS, Exploits,
  Generic, Reconnaissance, Shellcode, Worms

The 'Backdoors', 'Shellcode', and 'Worms' categories are most behaviorally
aligned with ransomware network activity (C2 callbacks, payload delivery,
lateral movement). This is discussed in the paper's threat model.

Schema differences from CICIDS2018 (handled automatically):
  - Label column may be 'label' (binary: 0/1) or 'attack_cat' (multiclass)
  - 'Normal' class (not 'BENIGN')
  - Some categorical columns (proto, service, state) are encoded
  - Contains integer and float columns without inf values (CIC-specific issue absent)

Mode selected automatically:
  - If 'attack_cat' column exists → multiclass (9 categories + Normal)
  - If only 'label' binary column → binary (Normal vs Attack)

Outputs saved to results/unsw/ and saved_models/
-------------------------------------------------
  metrics_unsw.csv
  cv_summary_unsw.csv
  confusion_matrix_*.png
  report_*.csv
  feature_importance_*.png
  (saved_models/)
    unsw_random_forest.pkl
    unsw_xgboost.pkl
    unsw_hybrid.pkl
    unsw_scaler.pkl
    unsw_label_encoder.pkl
    unsw_feature_names.pkl

How to run
----------
  cd PAPER_RANSOMWARE_PROJECT
  python scripts/unsw_pipeline.py

Dataset format expected
-----------------------
  data/unsw_nb15/UNSW_NB15_training-set.csv   (or any CSV in that folder)
  data/unsw_nb15/UNSW_NB15_testing-set.csv    (optional)

The script merges all CSVs in data/unsw_nb15/ and performs its own split.
If no CSVs are found, a synthetic demo is generated for code validation only.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import clean_raw, full_preprocess
from utils.evaluation import (
    run_full_evaluation, save_metrics_table, compute_metrics, print_metrics
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_STATE     = 42
TEST_SIZE        = 0.20
SAMPLE_PER_CLASS = 10000       # UNSW-NB15 has ~257k rows; cap per class
CV_FOLDS         = 5

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "unsw")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "saved_models")

# UNSW-NB15 categorical columns that need encoding
CATEGORICAL_COLS = ["proto", "service", "state"]

# UNSW-NB15 columns to always drop (non-feature admin columns)
DROP_COLS = ["id", "srcip", "dstip", "Stime", "Ltime", "attack_cat", "label"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal-encode categorical columns present in UNSW-NB15.
    Uses pandas Categorical codes (stable, reproducible, no leakage risk since
    categories are determined per-column from the full loaded data).
    Unknown categories become -1.
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes.astype(float)
            df[col] = df[col].replace(-1, np.nan)  # treat unknown as missing
    return df


def load_unsw(project_root: str,
              sample_per_class: int) -> tuple:
    """
    Load UNSW-NB15 CSVs from data/unsw_nb15/.

    Auto-detects multiclass vs binary mode.
    Returns (DataFrame_with_label_col, mode_string).

    mode is either 'multiclass' or 'binary'.
    """
    data_dir = os.path.join(project_root, "data", "unsw_nb15")

    if not os.path.exists(data_dir):
        print("  data/unsw_nb15/ not found — using synthetic demo.")
        return _generate_synthetic_unsw(), "multiclass"

    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print("  No CSVs in data/unsw_nb15/ — using synthetic demo.")
        return _generate_synthetic_unsw(), "multiclass"

    frames = []
    print(f"\n  Loading UNSW-NB15 from: {data_dir}")
    for fname in sorted(csv_files):
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df.columns = df.columns.str.strip()
            frames.append(df)
            print(f"    Loaded: {fname} ({len(df):,} rows)")
        except Exception as e:
            print(f"    ERROR: {fname}: {e}")

    if not frames:
        return _generate_synthetic_unsw(), "multiclass"

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)

    # ── Determine mode and build label column ─────────────────────────────
    has_attack_cat = "attack_cat" in combined.columns
    has_label      = "label" in combined.columns

    if has_attack_cat:
        mode = "multiclass"
        print("  Mode: MULTICLASS (using attack_cat column)")

        # Clean attack_cat
        combined["attack_cat"] = combined["attack_cat"].fillna("Normal").str.strip()
        combined["attack_cat"] = combined["attack_cat"].replace("", "Normal")

        # Normalise 'Normal' label
        combined["attack_cat"] = combined["attack_cat"].replace(
            {"normal": "Normal", "NORMAL": "Normal"}
        )

        # Use attack_cat as the target label
        combined["label"] = combined["attack_cat"]

    elif has_label:
        mode = "binary"
        print("  Mode: BINARY (using label column: 0=Normal, 1=Attack)")
        combined["label"] = combined["label"].map({0: "Normal", 1: "Attack"})
        combined["label"] = combined["label"].fillna("Unknown")
        combined = combined[combined["label"] != "Unknown"]

    else:
        raise ValueError("UNSW-NB15: neither 'label' nor 'attack_cat' column found.")

    # ── Encode categorical features ───────────────────────────────────────
    combined = encode_categoricals(combined)

    # ── Drop non-feature columns (keep numeric only + label) ──────────────
    safe_drop = [c for c in DROP_COLS if c in combined.columns and c != "label"]
    combined = combined.drop(columns=safe_drop, errors="ignore")

    num_cols = combined.select_dtypes(include=["number"]).columns.tolist()
    combined = combined[num_cols + ["label"]].copy()

    # ── Sample per class ──────────────────────────────────────────────────
    print(f"\n  Raw class distribution:")
    print(combined["label"].value_counts().to_string())

    frames_sampled = []
    for cls, group in combined.groupby("label"):
        n = min(sample_per_class, len(group))
        frames_sampled.append(group.sample(n=n, random_state=RANDOM_STATE))
        print(f"    {cls:20s}: {len(group):>8,} → sampled {n:>6,}")

    result = pd.concat(frames_sampled, ignore_index=True)
    print(f"\n  Total: {len(result):,} rows, {result['label'].nunique()} classes")
    return result, mode


def _generate_synthetic_unsw(n_per_class: int = 600) -> pd.DataFrame:
    """Synthetic demo for UNSW-NB15 — pipeline validation only."""
    print("\n" + "!"*70)
    print("  WARNING: Using SYNTHETIC DATA — for code validation only.")
    print("  Replace with real UNSW-NB15 CSVs for paper results.")
    print("!"*70 + "\n")

    np.random.seed(RANDOM_STATE)
    classes = ["Normal", "Fuzzers", "Analysis", "Backdoors", "DoS",
               "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms"]
    frames = []
    for i, cls in enumerate(classes):
        X = np.random.randn(n_per_class, 40) + i * 0.6
        df = pd.DataFrame(X, columns=[f"feat_{j}" for j in range(40)])
        df["label"] = cls
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model builders (same hyperparameters as CICIDS for paper consistency)
# ─────────────────────────────────────────────────────────────────────────────

def build_rf(n_classes: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def build_xgb(n_classes: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
        use_label_encoder=False,
    )


def build_hybrid(rf_model, xgb_model) -> VotingClassifier:
    return VotingClassifier(
        estimators=[("random_forest", rf_model), ("xgboost", xgb_model)],
        voting="soft",
        n_jobs=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(model, X: np.ndarray, y: np.ndarray, model_name: str) -> dict:
    print(f"\n  Running {CV_FOLDS}-fold CV for {model_name}...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        model, X, y,
        cv=skf,
        scoring={
            "accuracy":    "accuracy",
            "f1_macro":    "f1_macro",
            "f1_weighted": "f1_weighted",
            "precision":   "precision_macro",
            "recall":      "recall_macro",
        },
        n_jobs=-1,
    )

    summary = {}
    print(f"\n  CV results for {model_name}:")
    for key, values in cv_results.items():
        if key.startswith("test_"):
            metric = key[5:]
            mean, std = np.mean(values), np.std(values)
            summary[metric] = {"mean": round(mean, 4), "std": round(std, 4)}
            print(f"    {metric:20s}: {mean:.4f} ± {std:.4f}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_model(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  [save] {path}")


def save_cv_summary(cv_results_dict: dict, save_path: str) -> None:
    rows = []
    for model_name, metrics in cv_results_dict.items():
        for metric, vals in metrics.items():
            rows.append({
                "model": model_name, "metric": metric,
                "mean": vals["mean"], "std": vals["std"],
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  [save] CV summary → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 70)
    print("UNSW-NB15 Pipeline — Hybrid ML for Enterprise Network Detection")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/7] Loading UNSW-NB15 dataset...")
    df, mode = load_unsw(PROJECT_ROOT, SAMPLE_PER_CLASS)
    dataset_label = f"UNSW-NB15 ({mode})"

    # ── 2. Train/test split ──────────────────────────────────────────────
    print("\n[2/7] Train/test split (80/20, stratified, random_state=42)...")
    X = df.drop(columns=["label"])
    y_raw = df["label"].values

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_raw,
    )
    print(f"  Train: {len(X_train_raw):,}  |  Test: {len(X_test_raw):,}")

    # ── 3. Preprocessing ─────────────────────────────────────────────────
    print("\n[3/7] Preprocessing...")
    prep = full_preprocess(
        X_train=X_train_raw.reset_index(drop=True),
        X_test=X_test_raw.reset_index(drop=True),
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw,
        use_smote=True,
    )

    X_train       = prep["X_train_bal"]
    y_train       = prep["y_train_bal"]
    X_train_scaled = prep["X_train_scaled"]
    y_train_enc    = prep["y_train"]
    X_test        = prep["X_test_scaled"]
    y_test        = prep["y_test"]
    le            = prep["le"]
    scaler        = prep["scaler"]
    feature_names = prep["feature_names"]
    class_names   = list(le.classes_)
    n_classes     = len(class_names)

    print(f"  Classes ({n_classes}): {class_names}")
    print(f"  Final feature count: {len(feature_names)}")

    save_model(scaler,        os.path.join(MODELS_DIR, "unsw_scaler.pkl"))
    save_model(le,            os.path.join(MODELS_DIR, "unsw_label_encoder.pkl"))
    save_model(feature_names, os.path.join(MODELS_DIR, "unsw_feature_names.pkl"))

    # ── 4. Build models ──────────────────────────────────────────────────
    print("\n[4/7] Building models...")
    rf  = build_rf(n_classes)
    xgb = build_xgb(n_classes)

    # ── 5. Cross-validation ──────────────────────────────────────────────
    print("\n[5/7] Cross-validation on training set...")
    cv_results = {
        "Random Forest": run_cv(build_rf(n_classes),  X_train_scaled, y_train_enc, "Random Forest"),
        "XGBoost":       run_cv(build_xgb(n_classes), X_train_scaled, y_train_enc, "XGBoost"),
    }
    save_cv_summary(cv_results, os.path.join(RESULTS_DIR, "cv_summary_unsw.csv"))

    # ── 6. Train final models ────────────────────────────────────────────
    print("\n[6/7] Training final models on full training set (SMOTE-balanced)...")

    print("  Training Random Forest...")
    rf.fit(X_train, y_train)
    save_model(rf, os.path.join(MODELS_DIR, "unsw_random_forest.pkl"))

    print("  Training XGBoost...")
    xgb.fit(X_train, y_train)
    save_model(xgb, os.path.join(MODELS_DIR, "unsw_xgboost.pkl"))

    print("  Building and training Soft Voting Hybrid (RF + XGBoost)...")
    hybrid = build_hybrid(build_rf(n_classes), build_xgb(n_classes))
    hybrid.fit(X_train, y_train)
    save_model(hybrid, os.path.join(MODELS_DIR, "unsw_hybrid.pkl"))

    # ── 7. Evaluate all models ───────────────────────────────────────────
    print("\n[7/7] Evaluating all models on hold-out test set...")
    all_metrics = []

    for model_name, model in [
        ("Random Forest",      rf),
        ("XGBoost",            xgb),
        ("Soft Voting Hybrid", hybrid),
    ]:
        print(f"\n  ── {model_name} ──")
        metrics = run_full_evaluation(
            model=model,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            feature_names=feature_names,
            model_name=model_name,
            dataset_name=dataset_label,
            output_dir=RESULTS_DIR,
        )
        all_metrics.append(metrics)

    save_metrics_table(all_metrics, os.path.join(RESULTS_DIR, "metrics_unsw.csv"))

    print("\n" + "=" * 70)
    print("UNSW-NB15 pipeline complete.")
    print(f"Results → {RESULTS_DIR}")
    print(f"Models  → {MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
