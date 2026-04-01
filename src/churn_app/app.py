from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)
MODEL_PATH = 'models/churn_model.pkl'

# Load model at startup
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    input_df = pd.DataFrame([data])

    # Simple prediction
    prediction = model.predict(input_df)
    return jsonify({
        "churn_prediction": int(prediction[0]),
        "status": "Success"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
