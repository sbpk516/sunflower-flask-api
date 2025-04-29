# app_flask.py

import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load the trained model once when the server starts
model = joblib.load('sunflower_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Sunflower Height Prediction API!"

@app.route('/predict', methods=['GET'])
def predict():
    # Get 'sunlight_hours' from URL query parameters
    sunlight_hours = request.args.get('sunlight_hours')

    if sunlight_hours is None:
        return jsonify({"error": "Please provide 'sunlight_hours' as a query parameter."}), 400

    try:
        sunlight_hours = float(sunlight_hours)
    except ValueError:
        return jsonify({"error": "Invalid input. 'sunlight_hours' must be a number."}), 400

    # Prepare the input for prediction
    X_new = np.array([[sunlight_hours]])
    predicted_height = model.predict(X_new)[0]

    # Return prediction
    return jsonify({
        "sunlight_hours": sunlight_hours,
        "predicted_height_cm": round(predicted_height, 2)
    })

if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
