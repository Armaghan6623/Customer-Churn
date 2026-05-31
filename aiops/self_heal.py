# src/customer_crunch/aiops/self_heal.py
import os
import sys
import time
import joblib
import warnings
from typing import Optional

# Silences the runpy namespace synchronization warning cleanly
warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")

# Guarantees root package folder visibility across nested modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


def dummy_retrain_callback():
    """Simulates loading train.py, refitting an XGBoost pipeline on new data,

    and returning the newly serialized binary model dictionary artifact.
    """
    print("\n⏳ [Callback Engine] Re-ingesting fresh target data matrix...")
    time.sleep(1)
    print("⏳ [Callback Engine] Minimizing gradient descent loss function parameters...")
    time.sleep(1.5)
    print("✅ [Callback Engine] Model optimized. New performance metrics meet deployment baseline.")
    
    # Return a dummy structural dictionary mimicking your training pipeline artifact payload
    return {"model_type": "XGBoostClassifier", "status": "retrained_fresh", "version": 2.0}


def self_heal_if_needed(
    drift_report: dict,
    retrain_callback,
    drift_threshold: int = 1,
    model_path: Optional[str] = None,
):
    """Simple self-healing mechanism.

    If drift_report indicates enough drifted features, call `retrain_callback`.

    - drift_threshold: minimum number of drifted features required to trigger retraining.
    - retrain_callback: function with signature `() -> Any` that retrains and returns artifacts.
    - model_path (optional): if provided and retrain_callback returns a model object, persist it.
    """
    drift_features = drift_report.get("drift_features", [])
    drift_count = len(drift_features)

    if drift_count < drift_threshold:
        return {
            "status": "skipped",
            "reason": f"drift_count={drift_count} < drift_threshold={drift_threshold}",
            "drift_features": drift_features,
        }

    # Trigger retraining
    print(f"\n🚨 SELF-HEAL TRIGGER ACTIVE: Drift count ({drift_count}) matches or exceeds threshold ({drift_threshold}).")
    artifact = retrain_callback()

    # Optionally persist to a path (if your retrain_callback returns a joblib-serializable object)
    if model_path is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(artifact, model_path)
        print(f"💾 Freshly optimized model artifact successfully written out to: {model_path}")

    return {
        "status": "retrained",
        "drift_features": drift_features,
        "drift_count": drift_count,
    }


if __name__ == "__main__":
    print("🛡️ Initializing AIOps Self-Healing Engine Smoke Verification...")
    print("=========================================================================")
    
    # 🧪 Scenario A: Test report where drift is below threshold bounds
    print("\n📋 Scenario A: Testing low-intensity baseline drift (Threshold = 2)")
    stable_report = {
        "alpha": 0.05,
        "total_features": 1,
        "drift_features": ["EstimatedSalary"],  # Only 1 feature drifted
        "details": [{"feature": "EstimatedSalary", "drift_detected": True}]
    }
    
    result_a = self_heal_if_needed(
        drift_report=stable_report,
        retrain_callback=dummy_retrain_callback,
        drift_threshold=2,
        model_path="saved_models/retrained_chk_test.joblib"
    )
    print(f"Execution Output Status: {result_a}")
    
    # 🧪 Scenario B: Test report where drift triggers full background recovery
    print("\n📋 Scenario B: Testing high-intensity demographic drift (Threshold = 1)")
    severe_report = {
        "alpha": 0.05,
        "total_features": 2,
        "drift_features": ["Age", "Balance"],  # 2 features drifted
        "details": [
            {"feature": "Age", "drift_detected": True},
            {"feature": "Balance", "drift_detected": True}
        ]
    }
    
    result_b = self_heal_if_needed(
        drift_report=severe_report,
        retrain_callback=dummy_retrain_callback,
        drift_threshold=1,
        model_path="saved_models/retrained_chk_test.joblib"
    )
    print(f"Execution Output Status: {result_b}")
    print("=========================================================================")