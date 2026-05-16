"""
scripts/cicids_pipeline.py
===========================
Full training and evaluation pipeline for CICIDS2018.

Research context
----------------
Paper: "A Hybrid Machine Learning Approach for Ransomware Detection
        in Enterprise Networks"

CICIDS2018 is NOT a pure ransomware dataset. It is used here as a proxy
for enterprise malicious traffic detection. Attack families such as Bot,
Infiltration, and DoS exhibit network behavioral patterns that overlap with
ransomware C2 communication and lateral movement. This framing is stated
explicitly in the paper's threat model section.

Models trained
--------------
  1. Random Forest          (baseline)
  2. XGBoost                (baseline)
  3. Soft Voting Hybrid     (proposed — RF + XGBoost, soft probability voting)

Outputs saved to results/cicids/ and saved_models/
---------------------------------------------------
  metrics_cicids.csv
  confusion_matrix_random_forest.png
  confusion_matrix_xgboost.png
  confusion_matrix_soft_voting_hybrid.png
  report_random_forest.csv
  report_xgboost.csv
  report_soft_voting_hybrid.csv
  feature_importance_random_forest.png
  feature_importance_xgboost.png
  feature_importance_soft_voting_hybrid.png
  (saved_models/)
    cicids_random_forest.pkl
    cicids_xgboost.pkl
    cicids_hybrid.pkl
    cicids_scaler.pkl
    cicids_label_encoder.pkl
    cicids_feature_names.pkl

How to run
----------
  cd PAPER_RANSOMWARE_PROJECT
  python scripts/cicids_pipeline.py

Dataset format expected
-----------------------
Option A (Kaggle pre-split per-attack CSVs):
  data/cicids2018/Bot.csv
  data/cicids2018/Brute Force -Web.csv
  ... etc.
  Each file has a 'Label' column with the attack name or 'BENIGN'.

Option B (Full daily CSVs from CIC website):
  data/cicids2018/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
  ... etc.

The script auto-detects which format is present.
If neither is found, a synthetic demo dataset is generated for code
validation only — this is clearly flagged and must be replaced with real data.
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
from xgboost import XGBClassifier

# Add project root to path so utils is importable regardless of working dir
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import clean_raw, full_preprocess
from utils.evaluation import (
    run_full_evaluation, save_metrics_table, compute_metrics, print_metrics
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
SAMPLE_PER_CLASS = 5000        # cap per class to keep training practical
CV_FOLDS      = 5

RESULTS_DIR     = os.path.join(PROJECT_ROOT, "results", "cicids")
MODELS_DIR      = os.path.join(PROJECT_ROOT, "saved_models")

# ── Kaggle format mapping (pre-split per-attack CSVs) ──────────────────────
# Each key is the unified class name used in the paper.
# Each value is a list of CSV filenames that contain that attack.
KAGGLE_FILE_MAP = {
    "BENIGN": [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",  # contains BENIGN rows
        "Wednesday-workingHours.pcap_ISCX.csv",
    ],
    "Bot": [
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    ],
    "Brute Force": [
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    ],
    "DDoS": [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ],
    "DoS": [
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    ],
    "Infiltration": [
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    ],
    "PortScan": [
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    ],
}

# Alternative: CICFlowMeter-format attack-named CSVs (Kaggle variant)
KAGGLE_ATTACK_FILES = {
    "BENIGN":       ["BENIGN.csv"],
    "Bot":          ["Bot.csv"],
    "Brute Force":  ["Brute Force -Web.csv", "Brute Force -XSS.csv",
                     "FTP-BruteForce.csv", "SSH-Bruteforce.csv"],
    "DDoS":         ["DDOS attack-HOIC.csv", "DDOS attack-LOIC-UDP.csv",
                     "DDoS attacks-LOIC-HTTP.csv"],
    "DoS":          ["DoS attacks-GoldenEye.csv", "DoS attacks-Hulk.csv",
                     "DoS attacks-SlowHTTPTest.csv", "DoS attacks-Slowloris.csv"],
    "Infiltration": ["Infilteration.csv"],
    "SQL Injection": ["SQL Injection.csv"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_label_col(df: pd.DataFrame) -> str:
    """Return the name of the label column, case-insensitively."""
    for candidate in ["Label", "label", "CLASS", "class", "attack_cat"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No label column found. Columns: {df.columns.tolist()}")


def _load_csv(path: str) -> pd.DataFrame:
    """Load a single CSV with basic cleaning."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def load_cicids_from_attack_csvs(data_dir: str,
                                  file_map: dict,
                                  sample_per_class: int) -> pd.DataFrame:
    """
    Load CICIDS2018 from per-attack CSVs (Kaggle variant).

    For each class, reads all listed CSVs, filters rows to the target class,
    samples up to sample_per_class rows, then concatenates all classes.

    If a file is missing, it is skipped with a warning.
    """
    frames = []
    print(f"\n  Loading CICIDS2018 from per-attack CSVs in: {data_dir}")

    for class_name, filenames in file_map.items():
        class_frames = []
        for fname in filenames:
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                print(f"    SKIP (not found): {fname}")
                continue
            try:
                df = _load_csv(fpath)
                label_col = _find_label_col(df)

                if class_name == "BENIGN":
                    mask = df[label_col].str.strip().str.upper() == "BENIGN"
                else:
                    mask = df[label_col].str.strip().str.contains(
                        class_name, case=False, na=False
                    )

                sub = df[mask].copy()
                sub = sub.drop(columns=[label_col])
                sub["label"] = class_name
                class_frames.append(sub)
            except Exception as e:
                print(f"    ERROR loading {fname}: {e}")

        if not class_frames:
            print(f"    WARNING: No data found for class '{class_name}'")
            continue

        class_df = pd.concat(class_frames, ignore_index=True)

        # Keep only numeric features + label
        num_cols = class_df.select_dtypes(include=["number"]).columns.tolist()
        class_df = class_df[num_cols + ["label"]].copy()

        n_avail = len(class_df)
        n_sample = min(sample_per_class, n_avail)
        class_df = class_df.sample(n=n_sample, random_state=RANDOM_STATE)
        print(f"    {class_name:20s}: {n_avail:>7,} rows → sampled {n_sample:>5,}")
        frames.append(class_df)

    if not frames:
        raise RuntimeError(
            "No CICIDS2018 data loaded. Check data/cicids2018/ contains your CSVs."
        )

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total loaded: {len(combined):,} rows across {len(frames)} classes")
    print(f"  Class distribution:\n{combined['label'].value_counts().to_string()}\n")
    return combined


def load_cicids_from_daily_csvs(data_dir: str, sample_per_class: int) -> pd.DataFrame:
    """
    Load CICIDS2018 from full daily CICFlowMeter CSVs.
    Reads all CSVs in data_dir, extracts Label column, unifies class names,
    and samples up to sample_per_class per class.
    """
    print(f"\n  Loading CICIDS2018 from daily CSVs in: {data_dir}")
    all_frames = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(data_dir, fname)
        try:
            df = _load_csv(fpath)
            label_col = _find_label_col(df)
            df = df.rename(columns={label_col: "label"})
            df["label"] = df["label"].str.strip()
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            df = df[num_cols + ["label"]].copy()
            all_frames.append(df)
            print(f"    Loaded: {fname} ({len(df):,} rows)")
        except Exception as e:
            print(f"    ERROR: {fname}: {e}")

    if not all_frames:
        raise RuntimeError("No CSVs loaded from daily files.")

    combined = pd.concat(all_frames, ignore_index=True)

    # Unify label names
    combined["label"] = combined["label"].str.upper().replace({
        "BENIGN": "BENIGN",
        "BOT": "Bot",
    })

    # Group DoS variants
    dos_mask = combined["label"].str.contains("DOS", case=False, na=False)
    combined.loc[dos_mask, "label"] = "DoS"

    ddos_mask = combined["label"].str.contains("DDOS", case=False, na=False)
    combined.loc[ddos_mask, "label"] = "DDoS"

    brute_mask = combined["label"].str.contains("BRUTE|FTP-BRUTEFORCE|SSH", case=False, na=False)
    combined.loc[brute_mask, "label"] = "Brute Force"

    # Sample per class
    frames = []
    for cls, group in combined.groupby("label"):
        n = min(sample_per_class, len(group))
        frames.append(group.sample(n=n, random_state=RANDOM_STATE))
        print(f"    {cls:25s}: {len(group):>7,} → sampled {n:>5,}")

    result = pd.concat(frames, ignore_index=True)
    print(f"\n  Total: {len(result):,} rows, {result['label'].nunique()} classes")
    return result


def generate_synthetic_demo(n_per_class: int = 800) -> pd.DataFrame:
    """
    Generate a synthetic dataset for pipeline validation ONLY.
    !! This must NOT be used for your actual paper results. !!
    Replace with real CICIDS2018 data.
    """
    print("\n" + "!"*70)
    print("  WARNING: Using SYNTHETIC DATA — for code validation only.")
    print("  Replace with real CICIDS2018 CSVs for your paper results.")
    print("!"*70 + "\n")

    np.random.seed(RANDOM_STATE)
    classes = ["BENIGN", "Bot", "Brute Force", "DDoS", "DoS",
               "Infiltration", "SQL Injection"]
    frames = []
    for i, cls in enumerate(classes):
        n_features = 50
        X = np.random.randn(n_per_class, n_features) + (i * 0.8)
        df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(n_features)])
        df["label"] = cls
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_cicids(project_root: str, sample_per_class: int) -> pd.DataFrame:
    """
    Auto-detect and load CICIDS2018 data.

    Priority:
      1. data/cicids2018/ with per-attack CSVs (KAGGLE_ATTACK_FILES mapping)
      2. data/cicids2018/ with daily CICFlowMeter CSVs
      3. Synthetic demo (validation only)
    """
    data_dir = os.path.join(project_root, "data", "cicids2018")

    if not os.path.exists(data_dir):
        print(f"  data/cicids2018/ not found — falling back to synthetic demo.")
        return generate_synthetic_demo()

    # Check for per-attack CSV format
    has_attack_csvs = any(
        os.path.exists(os.path.join(data_dir, f))
        for flist in KAGGLE_ATTACK_FILES.values()
        for f in flist
    )
    if has_attack_csvs:
        print("  Detected: per-attack CSV format (Kaggle variant)")
        return load_cicids_from_attack_csvs(data_dir, KAGGLE_ATTACK_FILES, sample_per_class)

    # Check for daily CSVs
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if csv_files:
        print("  Detected: daily CICFlowMeter CSV format")
        return load_cicids_from_daily_csvs(data_dir, sample_per_class)

    print("  No CSVs found in data/cicids2018/ — using synthetic demo.")
    return generate_synthetic_demo()


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_rf(n_classes: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight=None,        # SMOTE handles imbalance
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


def build_hybrid(rf_model: RandomForestClassifier,
                  xgb_model: XGBClassifier) -> VotingClassifier:
    """
    Soft Voting Hybrid: RF + XGBoost.
    Both estimators use predict_proba; final prediction = average of probabilities.
    This is the paper's proposed model.
    """
    return VotingClassifier(
        estimators=[
            ("random_forest", rf_model),
            ("xgboost",       xgb_model),
        ],
        voting="soft",
        n_jobs=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation (on training data only)
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(model, X_train_scaled: np.ndarray, y_train: np.ndarray,
           model_name: str) -> dict:
    """
    5-fold stratified cross-validation on the (already-scaled) training set.

    Note: SMOTE is NOT applied inside CV folds here because the training data
    has already been balanced by SMOTE before this call. This is intentional:
    the CV serves as an internal stability check, not the primary evaluation.
    Primary results come from the held-out test set.

    Returns dict of mean ± std for each metric.
    """
    print(f"\n  Running {CV_FOLDS}-fold CV for {model_name}...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        model, X_train_scaled, y_train,
        cv=skf,
        scoring={
            "accuracy":  "accuracy",
            "f1_macro":  "f1_macro",
            "f1_weighted": "f1_weighted",
            "precision": "precision_macro",
            "recall":    "recall_macro",
        },
        n_jobs=-1,
        return_train_score=False,
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
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_model(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  [save] {path}")


def save_cv_summary(cv_results_dict: dict, save_path: str) -> None:
    """Save cross-validation summary as CSV."""
    rows = []
    for model_name, metrics in cv_results_dict.items():
        for metric, vals in metrics.items():
            rows.append({
                "model":  model_name,
                "metric": metric,
                "mean":   vals["mean"],
                "std":    vals["std"],
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
    print("CICIDS2018 Pipeline — Hybrid ML for Enterprise Network Detection")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/7] Loading CICIDS2018 dataset...")
    df = load_cicids(PROJECT_ROOT, SAMPLE_PER_CLASS)

    # ── 2. Train/test split (stratified, 80/20, before any preprocessing) ─
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

    # ── 3. Preprocessing (fit on train only) ─────────────────────────────
    print("\n[3/7] Preprocessing (imputation → feature selection → scaling → SMOTE)...")
    prep = full_preprocess(
        X_train=X_train_raw.reset_index(drop=True),
        X_test=X_test_raw.reset_index(drop=True),
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw,
        use_smote=True,
    )

    X_train = prep["X_train_bal"]     # SMOTE-balanced, scaled
    y_train = prep["y_train_bal"]
    X_train_scaled = prep["X_train_scaled"]   # scaled but not SMOTE (for CV stability check)
    y_train_enc    = prep["y_train"]
    X_test  = prep["X_test_scaled"]
    y_test  = prep["y_test"]
    le      = prep["le"]
    scaler  = prep["scaler"]
    feature_names = prep["feature_names"]
    class_names   = list(le.classes_)
    n_classes     = len(class_names)

    print(f"  Classes ({n_classes}): {class_names}")
    print(f"  Final feature count: {len(feature_names)}")

    # Save scaler + encoder + feature names for deployment / reproducibility
    save_model(scaler,        os.path.join(MODELS_DIR, "cicids_scaler.pkl"))
    save_model(le,            os.path.join(MODELS_DIR, "cicids_label_encoder.pkl"))
    save_model(feature_names, os.path.join(MODELS_DIR, "cicids_feature_names.pkl"))

    # ── 4. Build models ──────────────────────────────────────────────────
    print("\n[4/7] Building models...")
    rf  = build_rf(n_classes)
    xgb = build_xgb(n_classes)

    # ── 5. Cross-validation ──────────────────────────────────────────────
    print("\n[5/7] Cross-validation on training set...")
    # Use scaled (non-SMOTE) data for CV — gives honest stability estimate
    cv_results = {}
    cv_results["Random Forest"]      = run_cv(build_rf(n_classes),  X_train_scaled, y_train_enc, "Random Forest")
    cv_results["XGBoost"]            = run_cv(build_xgb(n_classes), X_train_scaled, y_train_enc, "XGBoost")

    save_cv_summary(cv_results, os.path.join(RESULTS_DIR, "cv_summary_cicids.csv"))

    # ── 6. Train final models on full (SMOTE-balanced) training set ──────
    print("\n[6/7] Training final models on full training set (SMOTE-balanced)...")

    print("  Training Random Forest...")
    rf.fit(X_train, y_train)
    save_model(rf, os.path.join(MODELS_DIR, "cicids_random_forest.pkl"))

    print("  Training XGBoost...")
    xgb.fit(X_train, y_train)
    save_model(xgb, os.path.join(MODELS_DIR, "cicids_xgboost.pkl"))

    print("  Building and training Soft Voting Hybrid (RF + XGBoost)...")
    # Build hybrid from fresh estimators (not the already-fitted ones)
    # VotingClassifier re-fits its own clones internally
    rf_for_hybrid  = build_rf(n_classes)
    xgb_for_hybrid = build_xgb(n_classes)
    hybrid = build_hybrid(rf_for_hybrid, xgb_for_hybrid)
    hybrid.fit(X_train, y_train)
    save_model(hybrid, os.path.join(MODELS_DIR, "cicids_hybrid.pkl"))

    # ── 7. Evaluate all models on hold-out test set ──────────────────────
    print("\n[7/7] Evaluating all models on hold-out test set...")
    all_metrics = []

    models_to_eval = [
        ("Random Forest",      rf),
        ("XGBoost",            xgb),
        ("Soft Voting Hybrid", hybrid),
    ]

    for model_name, model in models_to_eval:
        print(f"\n  ── {model_name} ──")
        metrics = run_full_evaluation(
            model=model,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            feature_names=feature_names,
            model_name=model_name,
            dataset_name="CICIDS2018",
            output_dir=RESULTS_DIR,
        )
        all_metrics.append(metrics)

    # Save unified metrics table
    save_metrics_table(all_metrics, os.path.join(RESULTS_DIR, "metrics_cicids.csv"))

    print("\n" + "=" * 70)
    print("CICIDS2018 pipeline complete.")
    print(f"Results → {RESULTS_DIR}")
    print(f"Models  → {MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
