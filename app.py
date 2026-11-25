from pathlib import Path

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "hdi_model.pkl"

model = joblib.load(MODEL_PATH)

# Keep the feature names and order exactly as used during model training.
FEATURE_COLUMNS = [
    "Life Expectancy at Birth (2021)",
    "Mean Years of Schooling (2021)",
    "Expected Years of Schooling (2021)",
    "Gross National Income Per Capita (2021)",
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        life = float(request.form["life"])
        mean = float(request.form["mean"])
        expected = float(request.form["expected"])
        gni = float(request.form["gni"])

        if life <= 0 or mean < 0 or expected < 0 or gni < 0:
            raise ValueError

        # DataFrame preserves the feature names expected by the trained model.
        features = pd.DataFrame(
            [[life, mean, expected, gni]],
            columns=FEATURE_COLUMNS
        )
        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted HDI Group: {prediction}",
            result=True,
            values=request.form
        )

    except (ValueError, TypeError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid numeric values. Life expectancy must be greater than 0; the other values cannot be negative.",
            values=request.form
        )


if __name__ == "__main__":
    app.run(debug=False)
