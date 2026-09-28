"""Serve a selected MLflow text pipeline with a minimal Flask API."""
import os
import mlflow
import mlflow.pyfunc
from flask import Flask, jsonify, request

mlflow.set_tracking_uri("file:./mlruns")
app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        uri = os.getenv("MODEL_URI")
        if not uri:
            raise RuntimeError("Set MODEL_URI to a URI printed by select_best.py.")
        _model = mlflow.pyfunc.load_model(uri)
    return _model


@app.post("/predict")
def predict():
    body = request.get_json(silent=True) or {}
    texts = body.get("texts")
    if not isinstance(texts, list) or not texts or not all(isinstance(text, str) for text in texts):
        return jsonify(error="Provide a nonempty list of strings in 'texts'."), 400
    predictions = get_model().predict(texts)
    return jsonify(predictions=[int(value) for value in predictions])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
