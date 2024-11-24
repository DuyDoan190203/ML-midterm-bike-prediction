from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the saved model
with open("tuned_random_forest.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data
    data = request.get_json()

    # Map the input to match the training features
    features = [
        data['instant'], data['season'], data['yr'], data['mnth'], data['hr'],
        data['holiday'], data['weekday'], data['workingday'], data['weathersit'],
        data['temp'], data['atemp'], data['hum'], data['windspeed'],
        data['day'], data['day_of_week']
    ]

    print("Features received:", features)

    # Apply the scaler
    scaled_features = scaler.transform([features])

    # Predict using the trained model
    prediction = model.predict(scaled_features)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
