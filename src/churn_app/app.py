from flask import Flask, request, jsonify
import pandas as pd
# Import your logic here once created
# from .db_handler import get_customer_data
# from .prediction_service import predict_churn

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # This will eventually connect to RDS and the ML model
    data = request.get_json()
    return jsonify({"message": "Prediction endpoint reached", "input": data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
