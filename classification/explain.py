# src/customer_crunch/classification/explain.py
import os
import sys
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Prevents terminal from crashing by saving the graph silently
import matplotlib.pyplot as plt

# Ensure Python can find 'src' root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

def generate_customer_explanation(customer_data, model_path="saved_models/churn_pipeline.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Trained model not found at {model_path}. Run train.py first!")
        
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps['preprocessor']
    xgb_model = pipeline.named_steps['classifier']
    
    input_df = pd.DataFrame([customer_data])
    X_transformed = preprocessor.transform(input_df)
    
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_features = cat_encoder.get_feature_names_out(['Geography', 'Gender']).tolist()
    num_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    all_feature_names = num_features + encoded_cat_features

    explainer = shap.TreeExplainer(xgb_model)
    explanation_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Impact_Score (SHAP)': explainer.shap_values(X_transformed)[0]
    })
    
    explanation_df['Absolute_Impact'] = explanation_df['Impact_Score (SHAP)'].abs()
    explanation_df = explanation_df.sort_values(by='Absolute_Impact', ascending=False).drop(columns=['Absolute_Impact'])
    
    return explanation_df

if __name__ == "__main__":
    print("🧠 Initializing Explainable AI (SHAP) Engine & Visualizer...")
    
    sample_customer = {
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
        # 1. Print Text Report
        impact_report = generate_customer_explanation(sample_customer)
        print("\n🎯 Top Drivers (Text Report):")
        print(impact_report.head(4))

        # 2. Generate and Save the Visual Graph
        pipeline = joblib.load("saved_models/churn_pipeline.joblib")
        preprocessor = pipeline.named_steps['preprocessor']
        xgb_model = pipeline.named_steps['classifier']
        
        input_df = pd.DataFrame([sample_customer])
        X_transformed = preprocessor.transform(input_df)
        
        cat_encoder = preprocessor.named_transformers_['cat']
        encoded_cat_features = cat_encoder.get_feature_names_out(['Geography', 'Gender']).tolist()
        num_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        all_feature_names = num_features + encoded_cat_features

        explainer = shap.TreeExplainer(xgb_model)
        explanation_obj = explainer(X_transformed)
        explanation_obj.feature_names = all_feature_names

        # 1. Generate Waterfall Plot
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation_obj[0], show=False)
        plt.tight_layout()
        plt.savefig("shap_waterfall.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Waterfall plot saved as: 'shap_waterfall.png'")
        
        # 2. Generate Force-like plot (matplotlib-safe)
        # NOTE: `shap.plots.force` is often HTML-based and can fail depending on SHAP/matplotlib versions.
        # To keep the same output filename and provide a similar "impact" visualization, we plot the top-k
        # signed SHAP values for this single observation.
        force_k = 10
        sv = explanation_obj[0].values  # SHAP values for the single row
        abs_order = sv.argsort()[::-1]
        top_idx = abs_order[:force_k]
        top_features = [all_feature_names[i] for i in top_idx]
        top_values = sv[top_idx]

        plt.figure(figsize=(12, 6))
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in top_values]  # green=positive, red=negative
        plt.barh(range(len(top_values)), top_values[::-1], color=colors[::-1])
        plt.yticks(range(len(top_values)), top_features[::-1])
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("SHAP value (impact on model output)")
        plt.title(f"Top {force_k} feature impacts for the selected customer")
        plt.tight_layout()
        plt.savefig("shap_force.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Force-like (signed SHAP) plot saved as: 'shap_force.png'")

        
        # 3. Generate Bar Plot (Summary of impacts)
        plt.figure(figsize=(10, 6))
        shap.plots.bar(explanation_obj, show=False)
        plt.tight_layout()
        plt.savefig("shap_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Bar plot saved as: 'shap_bar.png'")
        
        print(f"\n🎨 SUCCESS: All SHAP visualization graphs generated!")
        
    except Exception as e:
        print(f"❌ SHAP visualization failed: {str(e)}")