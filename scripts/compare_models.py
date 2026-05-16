import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier

# Load files
files = [
    "data/Bot.csv",
    "data/Brute Force -Web.csv",
    "data/Brute Force -XSS.csv"
]

dfs = []
for f in files:
    temp = pd.read_csv(f)
    temp = temp.sample(n=min(5000, len(temp)), random_state=42)
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

# Clean
df.columns = df.columns.str.strip()
df = df.dropna()

# Features / labels
y = df["Label"]
X = df.drop(columns=["Label"])
X = X.select_dtypes(include=["number"])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

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
    eval_metric="mlogloss"
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

    acc = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted"
    )

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")