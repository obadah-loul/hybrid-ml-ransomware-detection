import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

train_df = pd.read_csv("data/UNSW_NB15/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("data/UNSW_NB15/UNSW_NB15_testing-set.csv")

df = pd.concat([train_df, test_df], ignore_index=True).dropna()

y = df["label"]
X = df.drop(columns=["label", "attack_cat"])
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
xgb = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, eval_metric="logloss")

hybrid = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    voting="soft"
)

hybrid.fit(X_train, y_train)
pred = hybrid.predict(X_test)

print(confusion_matrix(y_test, pred))