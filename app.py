from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load the trained model
model_filename = "gas_leak_model.h5"  # Ensure this path is correct
try:
    model = tf.keras.models.load_model(model_filename)
    print(f"Model {model_filename} loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json  # Receive input data as JSON
        
        # Validate the presence of input data
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in input data"}), 400
        
        # Convert input to NumPy array and reshape for prediction
        features = np.array(data["features"]).reshape(1, -1)
        
        # Ensure the model's input shape matches the shape of 'features'
        print(f"Input features: {features}")
        
        # Make prediction
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class with highest probability

        return jsonify({"prediction": int(predicted_class)})
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)
