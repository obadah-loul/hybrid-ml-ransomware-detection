import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Load files
files = [
    "data/Bot.csv",
    "data/Brute Force -Web.csv",
    "data/Brute Force -XSS.csv"
]

dfs = []

for f in files:
    df = pd.read_csv(f).sample(n=min(5000, len(pd.read_csv(f))), random_state=42)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Clean
df.columns = df.columns.str.strip()
df = df.dropna()

# Split X y
y = df["Label"]
X = df.drop(columns=["Label"])
X = X.select_dtypes(include=["number"])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
rf = RandomForestClassifier(n_estimators=50, random_state=42)

xgb = XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    eval_metric='mlogloss'
)

hybrid = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft'
)

# Train
hybrid.fit(X_train, y_train)

# Predict
pred = hybrid.predict(X_test)

# Output
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))