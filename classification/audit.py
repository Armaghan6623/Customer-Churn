# src/customer_crunch/classification/audit.py
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

class ModelAuditor:
    def __init__(self, model_path="saved_models/churn_pipeline.joblib", data_path="data/raw/Churn_Modelling kaggel.csv"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at {model_path}. Run train.py first!")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ Raw data not found at {data_path}!")
            
        self.pipeline = joblib.load(model_path)
        self.df = pd.read_csv(data_path)
        
    def run_bias_and_fairness_audit(self):
        print("\n🔒 STARTING COMPLIANCE & FAIRNESS AUDIT...")
        
        # 1. Generate predictions across the full dataset
        X = self.df.drop(columns=['CustomerId', 'Surname', 'Exited', 'RowNumber'], errors='ignore')
        y_true = self.df['Exited']
        y_pred = self.pipeline.predict(X)
        
        # Add predictions back temporarily to slice the metrics
        audit_df = self.df.copy()
        audit_df['Predictions'] = y_pred
        
        # 2. Slice Data by Protected Attribute: Gender
        female_group = audit_df[audit_df['Gender'] == 'Female']
        male_group = audit_df[audit_df['Gender'] == 'Male']
        
        # Calculate Churn Prediction Rates
        female_churn_rate = female_group['Predictions'].mean()
        male_churn_rate = male_group['Predictions'].mean()
        
        print(f"📊 Female Predicted Churn Rate: {female_churn_rate * 100:.2f}%")
        print(f"📊 Male Predicted Churn Rate: {male_churn_rate * 100:.2f}%")
        
        # Calculate Disparate Impact Ratio (DIR)
        # Using Male as reference group and Female as unprivileged/monitored group
        dir_ratio = female_churn_rate / male_churn_rate if male_churn_rate > 0 else 0
        print(f"⚖️ Disparate Impact Ratio (DIR): {dir_ratio:.4f}")
        
        if 0.80 <= dir_ratio <= 1.25:
            print("✅ PASS: The model falls within the ethically acceptable fairness boundary (0.80 - 1.25).")
        else:
            print("⚠️ WARNING: Potential demographic bias detected! Review feature distributions.")
            
        # 3. Slice Performance Auditing by Country
        print("\n🌍 GEOGRAPHIC ERROR DISTRIBUTION AUDIT:")
        for country in audit_df['Geography'].unique():
            country_df = audit_df[audit_df['Geography'] == country]
            cm = confusion_matrix(country_df['Exited'], country_df['Predictions'])
            # Extract False Positives (FP) and False Negatives (FN)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
            
            total_errors = fp + fn
            error_rate = (total_errors / len(country_df)) * 100
            print(f" -> {country}: Error Rate = {error_rate:.2f}% ({total_errors} misclassifications)")

if __name__ == "__main__":
    try:
        auditor = ModelAuditor()
        auditor.run_bias_and_fairness_audit()
    except Exception as e:
        print(f"❌ Audit failed: {str(e)}")
