import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Any

from scipy.stats import ks_2samp


@dataclass
class DriftResult:
    feature: str
    p_value: float
    statistic: float


def _to_1d_array(x):
    arr = np.asarray(x)
    arr = arr.reshape(-1)
    return arr


def ks_drift_test(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Two-sample Kolmogorov–Smirnov test for drift.

    Returns dict with p_value and a boolean flag `drift_detected`.
    """
    ref = _to_1d_array(reference.dropna())
    cur = _to_1d_array(current.dropna())

    if len(ref) < 2 or len(cur) < 2:
        return {
            "feature": feature_name,
            "p_value": 1.0,
            "statistic": 0.0,
            "drift_detected": False,
            "reason": "Not enough samples for KS test",
        }

    res = ks_2samp(ref, cur)

    return {
        "feature": feature_name,
        "p_value": float(res.pvalue),
        "statistic": float(res.statistic),
        "drift_detected": bool(res.pvalue < alpha),
        "alpha": alpha,
    }


def monitor_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Monitor drift for a set of features."""
    results = []
    for col in feature_cols:
        results.append(
            ks_drift_test(
                reference=reference_df[col],
                current=current_df[col],
                feature_name=col,
                alpha=alpha,
            )
        )

    drifted = [r for r in results if r["drift_detected"]]

    return {
        "alpha": alpha,
        "total_features": len(feature_cols),
        "drift_features": [r["feature"] for r in drifted],
        "details": results,
    }

# src/customer_crunch/aiops/drift.py
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Guarantees root package folder visibility across nested modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

class DataDriftMonitor:
    def __init__(self, reference_data_path="漏hurn_Modelling kaggel.csv"):
        self.reference_data_path = reference_data_path
        
    def load_reference_data(self):
        """Loads the baseline data the model originally trained on."""
        if not os.path.exists(self.reference_data_path):
            raise FileNotFoundError(f"❌ Reference baseline data missing at: {self.reference_data_path}")
        return pd.read_csv(self.reference_data_path)

    def analyze_drift(self, current_data_df, alpha=0.05):
        """
        Executes a Two-Sample Kolmogorov-Smirnov Test on numerical features
        to mathematically detect distribution anomalies.
        """
        ref_df = self.load_reference_data()
        
        numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
        drift_report = {}
        drift_detected_global = False
        
        print("\n🔍 Running Statistical Kolmogorov-Smirnov Data Drift Analysis...")
        print("=========================================================================")
        print(f"{'Feature':<18} | {'KS-Statistic':<12} | {'p-value':<10} | {'Status':<10}")
        print("=========================================================================")
        
        for feature in numerical_features:
            if feature not in current_data_df.columns:
                continue
                
            # Extract historical baseline vs new runtime distributions
            baseline_distribution = ref_df[feature].dropna()
            current_distribution = current_data_df[feature].dropna()
            
            # Execute the 2-sample Kolmogorov-Smirnov test
            ks_stat, p_value = ks_2samp(baseline_distribution, current_distribution)
            
            # If p-value is smaller than our alpha threshold, drift is statistically significant
            has_drift = p_value < alpha
            status = "🚨 DRIFTED" if has_drift else "✅ STABLE"
            
            if has_drift:
                drift_detected_global = True
                
            drift_report[feature] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": has_drift
            }
            
            print(f"{feature:<18} | {ks_stat:<12.4f} | {p_value:<10.4e} | {status}")
            
        print("=========================================================================")
        
        if drift_detected_global:
            print("⚠️ ALERT: Significant structural data drift identified! Triggering pipeline alert.")
        else:
            print("🚀 Operational Integrity Normal: Data distributions remain stable.")
            
        return drift_detected_global, drift_report

if __name__ == "__main__":
    # Smoke test to simulate different system monitoring conditions
    monitor = DataDriftMonitor()
    
    # Condition 1: Simulate normal incoming user traffic (No drift)
    print("🧪 TEST CASE 1: Simulating Normal Production Incoming Traffic...")
    try:
        raw_df = pd.read_csv("Churn_Modelling kaggel.csv")
        # Sample a random slice to act as current incoming traffic
        normal_traffic = raw_df.sample(n=1000, random_state=42)
        monitor.analyze_drift(normal_traffic)
    except Exception as e:
        print(f"Test 1 Failed: {str(e)}")
        
    # Condition 2: Simulate an out-of-bounds demographic drift (e.g., age or balance shift)
    print("\n🧪 TEST CASE 2: Simulating Macroeconomic Shift (Drifted Age Demographic)...")
    try:
        drifted_traffic = raw_df.sample(n=1000, random_state=101).copy()
        # Simulate a sudden market change where the bank's average user profile ages by 8 years
        drifted_traffic['Age'] = drifted_traffic['Age'] + 8
        monitor.analyze_drift(drifted_traffic)
    except Exception as e:
        print(f"Test 2 Failed: {str(e)}")