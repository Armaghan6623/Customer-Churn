"""Inference pipeline — dataset-agnostic via DatasetConfig stored in the artifact."""
import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from classification.dataset_config import DatasetConfig, KAGGLE_BANK_CHURN


def _load_artifact(model_path: str) -> tuple:
    """Return (pipeline, config) from a saved artifact.

    Supports both the new dict format ``{"pipeline": ..., "config": ...}``
    and the legacy format where the file is the raw pipeline object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact missing at: {model_path}. Run train.py first."
        )
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"], artifact.get("config", KAGGLE_BANK_CHURN)
    # Legacy: artifact is the pipeline itself — assume Kaggle schema
    return artifact, KAGGLE_BANK_CHURN


def predict_single_customer(
    customer_data: dict,
    model_path: str = "saved_models/churn_pipeline.joblib",
    config: DatasetConfig = None,
) -> dict:
    """Run inference for a single customer record.

    Parameters
    ----------
    customer_data:
        Dict of feature name → value.
    model_path:
        Path to the joblib artifact produced by train.py.
    config:
        Optional DatasetConfig override.  When None the config embedded
        in the artifact is used (or KAGGLE_BANK_CHURN for legacy files).
    """
    pipeline, artifact_config = _load_artifact(model_path)
    cfg = config or artifact_config

    # Validate schema and bounds using the config
    cfg.validate_row(customer_data)

    input_df = pd.DataFrame([customer_data])

    probability = float(pipeline.predict_proba(input_df)[0][1])
    prediction = int(pipeline.predict(input_df)[0])

    return {
        "churn_probability": probability,
        "prediction": prediction,
        "status": "High Risk" if prediction == 1 else "Low Risk",
    }


if __name__ == "__main__":
    print("🧪 Running inference smoke test (Kaggle bank-churn schema)...")

    test_customer = {
        "CreditScore": 500,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 52,
        "Tenure": 2,
        "Balance": 125000.0,
        "NumOfProducts": 3,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 90000.0,
    }

    try:
        result = predict_single_customer(test_customer)
        print(f"✅ Status: {result['status']}")
        print(f"   Churn probability: {result['churn_probability'] * 100:.2f}%")
    except Exception as e:
        print(f"❌ Test failed: {e}")
