"""Inference pipeline (Features -> Prediction)"""
# src/customer_crunch/classification/predict.py
import os
import sys
import joblib
import pandas as pd

# This injection guarantees Python can map the root package layout no matter where the runner initiates execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

def predict_single_customer(customer_data, model_path="saved_models/churn_pipeline.joblib"):
    """
    Accepts a dictionary of customer features, validates schema integrity,
    and runs inference through the serialized pipeline.
    """
    # 1. Structural Check: Ensure model binary exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Production model artifact missing at: {model_path}. Run train.py first!")
        
    # 2. De-serialize Pipeline Artifact
    pipeline = joblib.load(model_path)
    
    # 3. Transform input payload into structural DataFrame
    input_df = pd.DataFrame([customer_data])
    
    # 4. Data Validation and Integrity Guardrails
    required_columns = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    
    # Schema Verification
    for col in required_columns:
        if col not in input_df.columns:
            raise ValueError(f"❌ Data Integrity Violation: Missing required structural column '{col}'")
            
    # Range Boundary Checks
    if not (0 <= input_df['Age'].iloc[0] <= 120):
        raise ValueError(f"❌ Input Boundary Fault: Unrealistic age dimension encountered ({input_df['Age'].iloc[0]})")
    if not (0 <= input_df['CreditScore'].iloc[0] <= 850):
        raise ValueError(f"❌ Input Boundary Fault: Credit Score outside real limits ({input_df['CreditScore'].iloc[0]})")

    # 5. Execute Pipeline Inference
    probability = pipeline.predict_proba(input_df)[0][1]  # Extract true probability vector for Class 1 (Churn)
    prediction = pipeline.predict(input_df)[0]            # Binary target class output
    
    return {
        "churn_probability": float(probability),
        "prediction": int(prediction),
        "status": "High Risk" if prediction == 1 else "Low Risk"
    }

if __name__ == "__main__":
    print("🧪 Executing Data Integrity and Inference Smoke Tests...")
    
    # Target Test Profile: A high-risk segment match
    test_customer = {
        'CreditScore': 500,
        'Geography': 'Germany',
        'Gender': 'Female',
        'Age': 52,
        'Tenure': 2,
        'Balance': 125000.0,
        'NumOfProducts': 3,
        'HasCrCard': 0,
        'IsActiveMember': 0,
        'EstimatedSalary': 90000.0
    }
    
    try:
        result = predict_single_customer(test_customer)
        print("\n✅ Inference Engine Test Passed!")
        print(f" -> Assessment Status: {result['status']}")
        print(f" -> Mathematical Churn Probability: {result['churn_probability'] * 100:.2f}%")
    except Exception as e:
        print(f"❌ Inference Test Failed: {str(e)}")