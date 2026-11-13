from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="House Price Prediction API")
#model = joblib.load("../best_model.pkl")

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(features: dict):
    x = np.array([list(features.values())]).reshape(1, -1)
    prediction = model.predict(x)[0]
    return {"predicted_price": float(prediction * 100000)}
