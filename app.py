from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved files
model = joblib.load("saved_models/hybrid_model.pkl")
columns = joblib.load("saved_models/columns.pkl")
encoder = joblib.load("saved_models/label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    # Keep numeric columns only
    df = df.select_dtypes(include=["number"])

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    preds = model.predict(df)
    labels = encoder.inverse_transform(preds)

    counts = pd.Series(labels).value_counts().to_dict()

    return render_template("result.html", counts=counts)

if __name__ == "__main__":
    app.run(debug=True)