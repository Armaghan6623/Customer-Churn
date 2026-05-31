"""Model training logic (XGBoost/LightGBM)"""
# src/customer_crunch/classification/train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_customer_churn_model(data_path, save_dir):
    # 1. Load Dataset
    print(f"📥 Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Separate Target and Features
    # We drop metadata columns that don't help the model learn general structural patterns
    X = df.drop(columns=['CustomerId', 'Surname', 'Exited', 'RowNumber'], errors='ignore')
    y = df['Exited']  # 1 = Churn (Exited), 0 = Stayed
    
    # 3. Define Feature Types based on your Kaggle Dataset columns
    categorical_features = ['Geography', 'Gender']
    numerical_features = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    
    # 4. Create Preprocessing Pipeline
    # This automatically scales numbers and transforms text categories cleanly
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # 5. Handle Class Imbalance (Calculate Ratio of Retained vs Churned users)
    num_retained = (y == 0).sum()
    num_churned = (y == 1).sum()
    scale_weight = num_retained / num_churned
    print(f"⚖️ Handling class imbalance. Scale position weight ratio: {scale_weight:.2f}")
    
    # 6. Build Complete End-to-End Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            scale_pos_weight=scale_weight, 
            random_state=42, 
            eval_metric='logloss'
        ))
    ])
    
    # 7. Stratified Train/Test Split (Ensures equal churn ratio in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 8. Train the Engine
    print("🚀 Training CustomerCrunch Classification Model (XGBoost)...")
    model_pipeline.fit(X_train, y_train)
    
    # 9. Evaluate Performance
    predictions = model_pipeline.predict(X_test)
    print("\n📊 Model Classification Report:")
    print(classification_report(y_test, predictions))
    
    # 10. Serialize and Save Pipeline Artifact
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'churn_pipeline.joblib')
    joblib.dump(model_pipeline, model_path)
    print(f"✅ Success! Pipeline serialized and saved to: {model_path}")

if __name__ == "__main__":
    # Ensure your data folder has the file renamed or match this name
    raw_data_path = "data/raw/Churn_Modelling kaggel.csv"
    output_directory = "saved_models"
    
    if os.path.exists(raw_data_path):
        train_customer_churn_model(raw_data_path, output_directory)
    else:
        print(f"❌ Data file not found! Make sure your Kaggle dataset is at: {raw_data_path}")