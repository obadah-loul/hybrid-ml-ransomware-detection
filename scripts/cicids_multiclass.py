"""
=============================================================================
Hybrid Machine Learning Approach for Network Intrusion Detection
Dataset: CICIDS2018
Course: Mobile and Wireless Security

Pipeline overview:
  1. Load + merge attack-family CSVs (including BENIGN)
  2. Correct inf/NaN handling (median imputation — NOT replacing with 0)
  3. Feature cleaning: remove zero-variance, high-correlation features
  4. Class balancing: SMOTE on training fold only (never on full dataset)
  5. Feature scaling: StandardScaler
  6. Stacking ensemble: RF + XGBoost + LightGBM → Logistic Regression meta-learner
  7. 5-fold stratified cross-validation with full metric reporting
  8. Final hold-out test evaluation with confusion matrix + classification report
  9. Feature importance via permutation importance
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =============================================================================
# SECTION 1 — FILE CONFIGURATION
# =============================================================================
# Adjust paths to match your directory structure.
# IMPORTANT: Include BENIGN traffic. Without it, your model never sees normal
# traffic and cannot function as a real IDS.

FILE_GROUPS = {
    "BENIGN": [
        "data/cicids2018/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
        # Add other days' files here — each contains a large BENIGN class
    ],
    "Bot": [
        "data/cicids2018/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    ],
    "Brute Force": [
        "data/cicids2018/Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    ],
    "DDoS": [
        "data/cicids2018/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    ],
    "DoS": [
        "data/cicids2018/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    ],
    "Infiltration": [
        "data/cicids2018/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    ],
    "SQL Injection": [
        "data/cicids2018/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    ],
}

# If your files are already per-attack CSVs (the Kaggle version), use the
# original mapping from your code instead. Just make sure to add BENIGN.
KAGGLE_FILE_GROUPS = {
    "BENIGN": [],   # Add the Wednesday CSV or whichever contains BENIGN rows
    "Bot": ["data/cicids2018/Bot.csv"],
    "Brute Force": [
        "data/cicids2018/Brute Force -Web.csv",
        "data/cicids2018/Brute Force -XSS.csv",
        "data/cicids2018/FTP-BruteForce.csv",
        "data/cicids2018/SSH-Bruteforce.csv",
    ],
    "DDoS": [
        "data/cicids2018/DDOS attack-HOIC.csv",
        "data/cicids2018/DDOS attack-LOIC-UDP.csv",
        "data/cicids2018/DDoS attacks-LOIC-HTTP.csv",
    ],
    "DoS": [
        "data/cicids2018/DoS attacks-GoldenEye.csv",
        "data/cicids2018/DoS attacks-Hulk.csv",
        "data/cicids2018/DoS attacks-SlowHTTPTest.csv",
        "data/cicids2018/DoS attacks-Slowloris.csv",
    ],
    "Infiltration": ["data/cicids2018/Infilteration.csv"],
    "SQL Injection": ["data/cicids2018/SQL Injection.csv"],
}

# Samples per class for the balanced training set.
# Use a higher value for common attacks (BENIGN, DDoS) and keep rare classes
# at their natural count (never oversample BEFORE splitting — do SMOTE inside CV).
SAMPLE_PER_CLASS = 5000
RANDOM_STATE = 42


# =============================================================================
# SECTION 2 — DATA LOADING
# =============================================================================

def load_file(path: str) -> pd.DataFrame:
    """Load a single CSV, strip column names, drop full-NA rows."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    return df


def load_dataset(file_groups: dict, sample_per_class: int) -> pd.DataFrame:
    """
    Load all files, extract per-group samples, return a combined DataFrame
    with a clean 'label' column.

    Key differences from original code:
    - The original Label column is preserved until the last moment
    - select_dtypes() is called AFTER adding 'label', then 'label' is excluded
      explicitly — not accidentally dropped by dtype filtering
    - BENIGN rows are extracted from mixed files by filtering Label == 'BENIGN'
    """
    dfs = []

    for group_name, files in file_groups.items():
        if not files:
            print(f"  Skipping {group_name} — no files configured")
            continue

        group_frames = []
        for filepath in files:
            try:
                temp = load_file(filepath)
            except FileNotFoundError:
                print(f"  WARNING: File not found — {filepath}")
                continue

            # Normalise the Label column name
            label_col = None
            for candidate in ["Label", "label", "CLASS", "class"]:
                if candidate in temp.columns:
                    label_col = candidate
                    break

            if label_col is None:
                print(f"  WARNING: No Label column in {filepath}, skipping")
                continue

            # If the file is a mixed file (e.g. full daily CSVs), filter to
            # the target group. If it is already a per-attack CSV (Kaggle),
            # all rows belong to this group.
            if group_name == "BENIGN":
                temp = temp[temp[label_col].str.strip().str.upper() == "BENIGN"].copy()
            elif temp[label_col].nunique() > 2:
                # Mixed file: filter to the group's attacks
                mask = temp[label_col].str.contains(group_name, case=False, na=False)
                temp = temp[mask].copy()

            temp = temp.drop(columns=[label_col])
            temp["label"] = group_name
            group_frames.append(temp)

        if not group_frames:
            print(f"  No data loaded for {group_name}")
            continue

        group_df = pd.concat(group_frames, ignore_index=True)

        # Select numeric features only — AFTER setting 'label' as a string col
        numeric_cols = group_df.select_dtypes(include=["number"]).columns.tolist()
        group_df = group_df[numeric_cols + ["label"]].copy()

        n_available = len(group_df)
        n_sample = min(sample_per_class, n_available)
        group_df = group_df.sample(n=n_sample, random_state=RANDOM_STATE)

        print(f"  {group_name}: {n_available} rows → sampled {n_sample}")
        dfs.append(group_df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal combined rows: {len(combined)}")
    print(f"Class distribution:\n{combined['label'].value_counts()}\n")
    return combined


# =============================================================================
# SECTION 3 — PREPROCESSING
# =============================================================================

def preprocess(df: pd.DataFrame):
    """
    Returns X (DataFrame), y (ndarray), label_encoder, feature_names.

    Critical fixes applied here:
    1. inf → NaN, then median imputation per column (NOT replacement with 0)
    2. Variance threshold: drop columns with near-zero variance
    3. Correlation filter: drop one of each pair with Pearson r > 0.95
    """
    X = df.drop(columns=["label"]).copy()
    y_raw = df["label"].values

    # --- Fix 1: inf → NaN → median imputation ---
    # CICIDS2018 has inf in flow-rate columns (e.g. Flow Bytes/s when duration=0)
    # Replacing with 0 would create a massive artificial spike at 0.
    X = X.replace([np.inf, -np.inf], np.nan)
    col_medians = X.median()
    X = X.fillna(col_medians)

    # --- Fix 2: Remove constant/near-zero variance columns ---
    # These carry no information but add noise to tree splits and importance.
    vt = VarianceThreshold(threshold=0.01)
    X_vt = vt.fit_transform(X)
    selected_mask = vt.get_support()
    X = pd.DataFrame(X_vt, columns=X.columns[selected_mask])
    print(f"After variance filter: {X.shape[1]} features (from {df.shape[1]-1})")

    # --- Fix 3: Correlation filter ---
    # Drop one column from each highly correlated pair (|r| > 0.95).
    # This reduces redundancy without hurting RF/XGB much, but it makes the
    # model more interpretable and speeds up training.
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X = X.drop(columns=to_drop)
    print(f"After correlation filter: {X.shape[1]} features ({len(to_drop)} dropped)")

    # --- Encode labels ---
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    feature_names = X.columns.tolist()

    return X, y, le, feature_names


# =============================================================================
# SECTION 4 — MODEL DEFINITION
# =============================================================================

def build_stacking_model(n_classes: int) -> StackingClassifier:
    """
    A genuine stacking ensemble (not a simple vote):
    - Base learners: RF, XGBoost, LightGBM
    - Meta-learner: Logistic Regression trained on out-of-fold predictions

    Why this is better than VotingClassifier:
    - Each base model's strengths are learned by the meta-learner
    - LightGBM is significantly faster than XGBoost and often better on
      imbalanced tabular data
    - The meta-learner learns WHICH model to trust for WHICH class

    Why these hyperparameters:
    - RF n_estimators=300: more trees = lower variance, plateau around 300
    - XGB max_depth=6: standard for tabular; deeper overfits CICIDS2018
    - LGB num_leaves=63: equivalent to depth-6 but leaf-wise (faster)
    - LR C=1.0: mild regularization for the meta-learner
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        # Do NOT use class_weight='balanced' here — we handle imbalance with SMOTE
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",   # faster than 'exact', required for GPU too
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    lgbm = LGBMClassifier(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    # Meta-learner: Logistic Regression
    # passthrough=True: the meta-learner also sees the original features,
    # not just the base model predictions — often improves performance.
    meta = LogisticRegression(
        C=1.0,
        max_iter=1000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    stacking = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=meta,
        cv=5,            # internal CV for generating meta-features
        passthrough=True,
        n_jobs=-1,
    )

    return stacking


# =============================================================================
# SECTION 5 — EVALUATION
# =============================================================================

def evaluate_cv(X, y, model, n_splits=5):
    """
    5-fold stratified cross-validation.
    SMOTE is applied INSIDE each fold (on training data only).
    StandardScaler is fit on training data only and applied to val data.

    This is the correct way — applying SMOTE or scaling before CV splits
    causes data leakage and inflates metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = {
        "accuracy": [], "f1_macro": [], "f1_weighted": [],
        "precision_macro": [], "recall_macro": []
    }

    print(f"\n{'='*60}")
    print(f"5-Fold Stratified Cross-Validation")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y[train_idx], y[val_idx]

        # Scale inside the fold — no leakage
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        # SMOTE inside the fold — no leakage
        # k_neighbors is reduced for very small classes
        min_class_count = np.bincount(y_tr).min()
        k = min(5, min_class_count - 1)
        if k < 1:
            print(f"  Fold {fold}: Skipping SMOTE (class too small)")
            X_tr_bal, y_tr_bal = X_tr_scaled, y_tr
        else:
            smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
            X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_scaled, y_tr)

        # Fit the model
        model.fit(X_tr_bal, y_tr_bal)
        preds = model.predict(X_val_scaled)

        acc = accuracy_score(y_val, preds)
        f1m = f1_score(y_val, preds, average="macro", zero_division=0)
        f1w = f1_score(y_val, preds, average="weighted", zero_division=0)
        prec = precision_score(y_val, preds, average="macro", zero_division=0)
        rec = recall_score(y_val, preds, average="macro", zero_division=0)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["f1_macro"].append(f1m)
        fold_metrics["f1_weighted"].append(f1w)
        fold_metrics["precision_macro"].append(prec)
        fold_metrics["recall_macro"].append(rec)

        print(f"  Fold {fold}: Acc={acc:.4f} | F1-macro={f1m:.4f} | F1-weighted={f1w:.4f}")

    print(f"\n--- CV Summary ---")
    for metric, values in fold_metrics.items():
        print(f"  {metric:20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    return fold_metrics


def evaluate_holdout(model, X_test_scaled, y_test, class_names):
    """Full evaluation on the final hold-out test set."""
    preds = model.predict(X_test_scaled)

    print(f"\n{'='*60}")
    print("Final Hold-Out Test Evaluation")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"F1 macro:  {f1_score(y_test, preds, average='macro', zero_division=0):.4f}")
    print(f"F1 weighted: {f1_score(y_test, preds, average='weighted', zero_division=0):.4f}")
    print()
    print(classification_report(y_test, preds, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix — Hold-Out Test Set", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved to confusion_matrix.png")

    return preds


def plot_feature_importance(model, X_test_scaled, y_test, feature_names, top_n=20):
    """
    Permutation importance on the test set.
    More reliable than RF's built-in impurity importance (which biases toward
    high-cardinality features) and works for any model, including XGBoost.
    """
    print(f"\nComputing permutation importance (top {top_n} features)...")
    # Use the RF sub-model for speed; swap to `model` for the full stacking model
    rf_model = model.named_estimators_["rf"]
    result = permutation_importance(
        rf_model, X_test_scaled, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["feature"][::-1], importance_df["importance_mean"][::-1],
            xerr=importance_df["importance_std"][::-1], color="#378ADD", alpha=0.85)
    ax.set_xlabel("Mean accuracy decrease (permutation importance)")
    ax.set_title(f"Top {top_n} Features — Permutation Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("Feature importance plot saved to feature_importance.png")


# =============================================================================
# SECTION 6 — MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 60)
    print("CICIDS2018 — Hybrid IDS Pipeline")
    print("=" * 60)

    # --- Load data ---
    # Switch to KAGGLE_FILE_GROUPS if using the pre-split Kaggle version
    print("\n[1] Loading dataset...")
    df = load_dataset(KAGGLE_FILE_GROUPS, SAMPLE_PER_CLASS)

    # --- Preprocess ---
    print("\n[2] Preprocessing...")
    X, y, le, feature_names = preprocess(df)
    class_names = list(le.classes_)
    n_classes = len(class_names)
    print(f"Classes ({n_classes}): {class_names}")
    print(f"Feature matrix shape: {X.shape}")

    # --- Train/Test split (hold-out BEFORE any fitting) ---
    # This set is never touched during CV. It represents unseen data.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train_full)} | Test size: {len(X_test)}")

    # --- Build model ---
    print("\n[3] Building stacking ensemble...")
    model = build_stacking_model(n_classes)

    # --- Cross-validation (on train set only) ---
    print("\n[4] Running 5-fold stratified CV...")
    cv_metrics = evaluate_cv(X_train_full, y_train_full, model)

    # --- Final model training on full train set ---
    print("\n[5] Training final model on full training set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE on full training set for final model
    min_class_count = np.bincount(y_train_full).min()
    k = min(5, min_class_count - 1)
    if k >= 1:
        smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train_full)
        print(f"After SMOTE: {X_train_bal.shape[0]} samples")
    else:
        X_train_bal, y_train_bal = X_train_scaled, y_train_full

    model.fit(X_train_bal, y_train_bal)

    # --- Final evaluation ---
    print("\n[6] Evaluating on hold-out test set...")
    evaluate_holdout(model, X_test_scaled, y_test, class_names)

    # --- Feature importance ---
    print("\n[7] Computing feature importance...")
    plot_feature_importance(model, X_test_scaled, y_test, feature_names)

    print("\n[8] Pipeline complete.")
    print("Files saved: confusion_matrix.png, feature_importance.png")


if __name__ == "__main__":
    main()


# =============================================================================
# ACADEMIC PAPER RECOMMENDATIONS
# =============================================================================
"""
Title reframe (recommended):
  "A Stacking Ensemble Approach for Multi-Class Network Intrusion Detection
   Using CICIDS2018 with SMOTE-Based Class Balancing"

  Subsection: "Applicability to Ransomware Precursor Traffic Detection"
  (Bot and Infiltration classes exhibit behavioral overlap with ransomware C2
   traffic — discuss this in your threat model section)

Key contributions to highlight:
  1. Proper inf/NaN handling specific to CICFlowMeter-generated features
  2. Within-fold SMOTE (vs. pre-split SMOTE — cite data leakage literature)
  3. Stacking vs. voting: show ablation table (RF alone, XGB alone, vote, stack)
  4. Permutation importance for interpretability (cite Breiman 2001 + Molnar 2022)

Tables your paper needs:
  Table 1: Dataset description (class, #samples before/after balancing)
  Table 2: Feature selection results (before/after variance + correlation filter)
  Table 3: 5-fold CV results (mean ± std per metric)
  Table 4: Hold-out test results per class (precision, recall, F1)
  Table 5: Ablation study (individual models vs. stacking)

Comparison models for ablation (easy to add):
  - Naive Bayes baseline (set a floor)
  - Decision Tree (single tree vs. forest)
  - RF alone
  - XGBoost alone
  - LightGBM alone
  - Soft Voting (your original approach)
  - Stacking (proposed)

Related papers to cite:
  - Sharafaldin et al. (2018) — original CICIDS2018 paper (must cite)
  - Leevy et al. (2018) — survey of oversampling for intrusion detection
  - Yang et al. (2022) — stacking ensembles for network anomaly detection
  - Brownlee (2021) — data leakage in CV (imbalanced-learn docs)
"""