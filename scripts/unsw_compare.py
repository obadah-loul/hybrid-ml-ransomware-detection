import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from xgboost import XGBClassifier

# Load train + test
train_df = pd.read_csv("data/UNSW_NB15/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("data/UNSW_NB15/UNSW_NB15_testing-set.csv")

df = pd.concat([train_df, test_df], ignore_index=True)

# Drop missing values
df = df.dropna()

# Labels
y = df["label"]

# Features
X = df.drop(columns=["label", "attack_cat"])

# Convert categorical columns
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
rf = RandomForestClassifier(n_estimators=50, random_state=42)

xgb = XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss"
)

hybrid = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    voting="soft"
)

models = {
    "Random Forest": rf,
    "XGBoost": xgb,
    "Proposed Hybrid": hybrid
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    if name == "Proposed Hybrid":
        cm = confusion_matrix(y_test, pred)
        print("\nConfusion Matrix:")
        print(cm)

    acc = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted"
    )

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")