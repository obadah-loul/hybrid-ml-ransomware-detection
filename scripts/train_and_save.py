import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

# Split features/labels
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

model = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    voting="soft"
)

# Train
model.fit(X_train, y_train)

# Save model + columns + encoder
joblib.dump(model, "saved_models/hybrid_model.pkl")
joblib.dump(list(X.columns), "saved_models/columns.pkl")
joblib.dump(le, "saved_models/label_encoder.pkl")

print("Model saved successfully.")