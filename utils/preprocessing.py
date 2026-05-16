"""
utils/preprocessing.py
=======================
Shared preprocessing library for:
  - A Hybrid Machine Learning Approach for Ransomware Detection in Enterprise Networks
  - Datasets: CICIDS2018 and UNSW-NB15

All functions are stateless and pure (no side effects).
Callers own the train/test split; this module never sees the full dataset
during fit — preventing data leakage.

Key design decisions
--------------------
1. inf → NaN → per-column MEDIAN imputation (not zero, not mean).
   CICIDS2018 flow-rate features produce inf when flow duration == 0.
   Zero replacement creates a massive artificial spike; median is honest.

2. Variance threshold (0.01) removes constant/near-zero-variance columns
   that contribute no discriminative signal.

3. Pearson |r| > 0.95 correlation filter removes one of each redundant pair.
   Applied AFTER variance filter. Both thresholds are fit on TRAIN only,
   then the same column mask is applied to TEST — no leakage.

4. StandardScaler is fit on TRAIN only.

5. SMOTE is applied to TRAIN only, after scaling.
   k_neighbors is reduced automatically when any class has very few samples.

6. LabelEncoder maps string labels to integers.
   The encoder is returned to the caller for inverse_transform on reports.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw data cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_raw(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Standardise a raw DataFrame before any ML processing.

    Steps:
      - Strip whitespace from column names
      - Drop rows that are entirely NaN
      - Replace inf / -inf with NaN (will be imputed later)
      - Keep only numeric columns + the label column
      - Remove duplicate rows (identical feature vectors with identical label)

    Parameters
    ----------
    df        : Raw DataFrame as loaded from CSV.
    label_col : Name of the target column (already renamed to 'label'
                by the caller before passing in).

    Returns
    -------
    Cleaned DataFrame with label column preserved.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Drop fully-empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Coerce inf → NaN in numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Keep numeric features + label only
    keep_cols = numeric_cols + ([label_col] if label_col in df.columns else [])
    df = df[keep_cols].copy()

    # Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  [clean] Dropped {dropped} duplicate rows.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Label encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(y_raw: np.ndarray) -> tuple:
    """
    Fit a LabelEncoder on y_raw and return (y_encoded, fitted_encoder).

    The encoder is returned so callers can use le.classes_ for display
    and le.inverse_transform() for reporting.
    """
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return y, le


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature selection (fit on train, apply to both)
# ─────────────────────────────────────────────────────────────────────────────

def fit_feature_selector(X_train: pd.DataFrame,
                          var_threshold: float = 0.01,
                          corr_threshold: float = 0.95) -> dict:
    """
    Fit a two-stage feature selector on training data only.

    Stage 1: VarianceThreshold — removes near-constant features.
    Stage 2: Pearson correlation filter — removes one column from each
             highly correlated pair (|r| > corr_threshold).

    Returns a dict with:
      'variance_mask'  : boolean array for stage-1 column selection
      'variance_cols'  : column names surviving stage 1
      'drop_corr'      : list of column names to drop in stage 2
      'final_cols'     : final column list (surviving both stages)
      'n_original'     : number of input features
      'n_final'        : number of final features
    """
    n_original = X_train.shape[1]

    # ── Stage 1: variance filter ──────────────────────────────────────────
    vt = VarianceThreshold(threshold=var_threshold)
    vt.fit(X_train)
    variance_mask = vt.get_support()
    variance_cols = X_train.columns[variance_mask].tolist()
    X_stage1 = X_train[variance_cols]

    n_after_var = len(variance_cols)
    print(f"  [feature_select] Variance filter: {n_original} → {n_after_var} features "
          f"({n_original - n_after_var} removed)")

    # ── Stage 2: correlation filter ───────────────────────────────────────
    corr_matrix = X_stage1.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    drop_corr = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    final_cols = [c for c in variance_cols if c not in drop_corr]

    n_final = len(final_cols)
    print(f"  [feature_select] Correlation filter: {n_after_var} → {n_final} features "
          f"({len(drop_corr)} removed, |r| > {corr_threshold})")

    return {
        "variance_mask": variance_mask,
        "variance_cols": variance_cols,
        "drop_corr": drop_corr,
        "final_cols": final_cols,
        "n_original": n_original,
        "n_final": n_final,
    }


def apply_feature_selector(X: pd.DataFrame, selector: dict) -> pd.DataFrame:
    """
    Apply a fitted feature selector to any DataFrame (train or test).
    Uses only the 'final_cols' list from fit_feature_selector().
    """
    available = [c for c in selector["final_cols"] if c in X.columns]
    missing = [c for c in selector["final_cols"] if c not in X.columns]
    if missing:
        print(f"  [feature_select] WARNING: {len(missing)} expected columns missing from data.")
    return X[available].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Imputation (fit on train, apply to both)
# ─────────────────────────────────────────────────────────────────────────────

def fit_imputer(X_train: pd.DataFrame) -> pd.Series:
    """
    Compute per-column median on training data.
    Returns a Series of medians (the 'fitted imputer').
    """
    return X_train.median()


def apply_imputer(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """
    Fill NaN values using the training medians.
    Columns in X that are not in medians are filled with 0 (safe fallback).
    """
    return X.fillna(medians).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Scaling (fit on train, apply to both)
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on training data. Returns fitted scaler."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted scaler to data."""
    return scaler.transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SMOTE oversampling (train only)
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Apply SMOTE to balance the training set.

    k_neighbors is automatically reduced to (min_class_count - 1)
    when any class has very few samples, preventing SMOTE from failing.

    If any class has only 1 sample (k would be 0), SMOTE is skipped
    and the original data is returned with a warning.

    Returns (X_balanced, y_balanced).
    """
    class_counts = np.bincount(y_train)
    min_count = class_counts.min()
    k = min(5, min_count - 1)

    if k < 1:
        print(f"  [smote] Skipped — smallest class has {min_count} sample(s). "
              "Using original distribution.")
        return X_train, y_train

    print(f"  [smote] Applying SMOTE (k_neighbors={k}) ...")
    before = len(y_train)
    smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"  [smote] Training samples: {before} → {len(y_bal)}")
    return X_bal, y_bal


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full preprocessing pipeline (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def full_preprocess(X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train_raw: np.ndarray,
                    y_test_raw: np.ndarray,
                    use_smote: bool = True) -> dict:
    """
    Execute the complete preprocessing pipeline on a pre-split dataset.

    Order of operations (leakage-free):
      1. Impute NaN using TRAIN medians → apply to both splits
      2. Feature selection fit on TRAIN → apply to both splits
      3. Scale fit on TRAIN → apply to both splits
      4. Encode labels (fit on TRAIN labels only)
      5. SMOTE on TRAIN only (after scaling)

    Parameters
    ----------
    X_train      : Training features (numeric, NaN allowed, inf already replaced)
    X_test       : Test features
    y_train_raw  : Training labels as strings
    y_test_raw   : Test labels as strings
    use_smote    : Whether to apply SMOTE (default True)

    Returns
    -------
    dict with keys:
      X_train_bal, y_train_bal  — balanced, scaled training arrays
      X_test_scaled             — scaled test array
      y_train, y_test           — encoded integer label arrays
      le                        — fitted LabelEncoder
      scaler                    — fitted StandardScaler
      selector                  — feature selector dict
      feature_names             — list of final feature names
    """
    print("\n  ── Preprocessing pipeline ──")

    # 1. Imputation
    medians = fit_imputer(X_train)
    X_train_imp = apply_imputer(X_train, medians)
    X_test_imp  = apply_imputer(X_test,  medians)

    # 2. Feature selection (fit on train only)
    selector = fit_feature_selector(X_train_imp)
    X_train_sel = apply_feature_selector(X_train_imp, selector)
    X_test_sel  = apply_feature_selector(X_test_imp,  selector)
    feature_names = selector["final_cols"]

    # 3. Label encoding
    # Fit on train labels; test labels are transformed with the same encoder.
    # This is safe because both splits come from the same label universe.
    all_labels = np.concatenate([y_train_raw, y_test_raw])
    le = LabelEncoder()
    le.fit(all_labels)
    y_train = le.transform(y_train_raw)
    y_test  = le.transform(y_test_raw)

    # 4. Scaling (fit on train only)
    X_train_arr = X_train_sel.values
    X_test_arr  = X_test_sel.values
    scaler = fit_scaler(X_train_arr)
    X_train_scaled = apply_scaler(X_train_arr, scaler)
    X_test_scaled  = apply_scaler(X_test_arr,  scaler)

    # 5. SMOTE (train only)
    if use_smote:
        X_train_bal, y_train_bal = apply_smote(X_train_scaled, y_train)
    else:
        X_train_bal, y_train_bal = X_train_scaled, y_train
        print("  [smote] Skipped by caller request.")

    print(f"  ── Preprocessing complete. Final feature count: {len(feature_names)} ──\n")

    return {
        "X_train_bal":    X_train_bal,
        "y_train_bal":    y_train_bal,
        "X_train_scaled": X_train_scaled,
        "y_train":        y_train,
        "X_test_scaled":  X_test_scaled,
        "y_test":         y_test,
        "le":             le,
        "scaler":         scaler,
        "selector":       selector,
        "feature_names":  feature_names,
        "medians":        medians,
    }
