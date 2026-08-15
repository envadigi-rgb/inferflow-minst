from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def health():
    return jsonify(status="ok")

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    pixels = payload.get("pixels")

    if pixels is None or len(pixels) != 64:
        return jsonify(error="expected 'pixels': list of 64 floats (8x8 image, 0-16 range)"), 400

    x = np.array(pixels, dtype=float).reshape(1, -1)
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0][pred]

    return jsonify(prediction=int(pred), confidence=round(float(proba), 4))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
