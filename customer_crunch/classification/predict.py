"""Inference pipeline — applies feature engineering then runs the model."""
from __future__ import annotations

import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from classification.dataset_config import DatasetConfig, KAGGLE_BANK_CHURN


def _load_artifact(model_path: str) -> tuple:
    """Return (pipeline, config, artifact_dict) from a saved artifact."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact missing at: {model_path}. Run train.py first."
        )
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"], artifact.get("config", KAGGLE_BANK_CHURN), artifact
    return artifact, KAGGLE_BANK_CHURN, {}


def _apply_feature_engineering(
    df: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:
    """Apply the same feature engineering used at training time, if recorded."""
    if not artifact.get("feature_engineering", False):
        return df
    try:
        from classification.feature_engineering import engineer_features
        df = engineer_features(df)
    except ImportError:
        pass
    return df


def predict_single_customer(
    customer_data: dict,
    model_path: str = "saved_models/churn_pipeline.joblib",
    config: DatasetConfig = None,
) -> dict:
    """Run inference for a single customer record.

    Parameters
    ----------
    customer_data : Dict of raw feature name → value (pre-engineering).
    model_path    : Path to the joblib artifact produced by train.py.
    config        : Optional DatasetConfig override.

    Returns
    -------
    dict with keys: churn_probability, prediction, status,
                    optimal_threshold (if stored), risk_tier
    """
    pipeline, artifact_config, artifact = _load_artifact(model_path)
    cfg = config or artifact_config

    # Validate raw input bounds
    cfg.validate_row(customer_data)

    # Build dataframe and apply feature engineering
    input_df = pd.DataFrame([customer_data])
    input_df = _apply_feature_engineering(input_df, artifact)

    # Keep only columns the pipeline expects
    numeric_features     = artifact.get("numeric_features",     cfg.numeric_features)
    categorical_features = artifact.get("categorical_features", cfg.categorical_features)
    expected_cols = [c for c in numeric_features + categorical_features
                     if c in input_df.columns]
    input_df = input_df[expected_cols]

    probability = float(pipeline.predict_proba(input_df)[0][1])
    prediction  = int(pipeline.predict(input_df)[0])

    # Use stored optimal threshold if available (business-calibrated)
    opt_threshold = artifact.get("business_metrics", {}).get("optimal_threshold", 0.5)
    business_pred = int(probability >= opt_threshold)
    risk_tier = (
        "High Risk"   if probability >= 0.70 else
        "Medium Risk" if probability >= 0.40 else
        "Low Risk"
    )

    return {
        "churn_probability":  probability,
        "prediction":         prediction,
        "business_prediction":business_pred,
        "optimal_threshold":  opt_threshold,
        "status":             "High Risk" if prediction == 1 else "Low Risk",
        "risk_tier":          risk_tier,
    }


if __name__ == "__main__":
    print("🧪 Inference smoke test (Kaggle bank-churn schema)...")
    test_customer = {
        "CreditScore": 500,
        "Geography":   "Germany",
        "Gender":      "Female",
        "Age":          52,
        "Tenure":        2,
        "Balance":    125000.0,
        "NumOfProducts": 3,
        "HasCrCard":     0,
        "IsActiveMember":0,
        "EstimatedSalary": 90000.0,
    }
    try:
        result = predict_single_customer(test_customer)
        print(f"✅ Status       : {result['status']}")
        print(f"   Risk tier    : {result['risk_tier']}")
        print(f"   Churn prob   : {result['churn_probability']*100:.2f}%")
        print(f"   Opt threshold: {result['optimal_threshold']:.2f}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
