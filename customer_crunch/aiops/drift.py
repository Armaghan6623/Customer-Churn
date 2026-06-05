"""Data drift detection using the two-sample Kolmogorov-Smirnov test.

Provides two interfaces:
  - Functional API  : ks_drift_test(), monitor_drift()  — used by MLOpsAgent
  - Class API       : DataDriftMonitor                  — used for standalone runs
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from scipy.stats import ks_2samp

# Ensure root package is importable when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    feature: str
    p_value: float
    statistic: float


def _to_1d_array(x) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _resolve_data_path() -> str:
    """Find the reference dataset across known project layouts."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "customer_churn_dataset.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "customer_churn_dataset.csv"),
        os.path.join(os.getcwd(), "customer_crunch", "data", "customer_churn_dataset.csv"),
        os.path.join(os.getcwd(), "customer_crunch", "data", "raw", "customer_churn_dataset.csv"),
        "/app/customer_crunch/data/customer_churn_dataset.csv",
        "/app/customer_crunch/data/raw/customer_churn_dataset.csv",
        "/app/data/customer_churn_dataset.csv",
        "/app/data/raw/customer_churn_dataset.csv",
        "customer_crunch/data/customer_churn_dataset.csv",
        "customer_crunch/data/raw/customer_churn_dataset.csv",
        "data/customer_churn_dataset.csv",
        "data/raw/customer_churn_dataset.csv",
        # legacy filenames
        "/app/customer_crunch/data/raw/Churn_Modelling kaggel.csv",
        "/app/data/raw/Churn_Modelling kaggel.csv",
        "data/raw/Churn_Modelling kaggel.csv",
        "Churn_Modelling kaggel.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "customer_crunch/data/customer_churn_dataset.csv"


# ---------------------------------------------------------------------------
# Functional API  (used by MLOpsAgent)
# ---------------------------------------------------------------------------

def ks_drift_test(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Two-sample KS test for a single feature.

    Returns a dict with p_value, statistic, and drift_detected flag.
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
    feature_cols: List[str],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Run KS drift test across all specified feature columns."""
    results = [
        ks_drift_test(reference_df[col], current_df[col], col, alpha)
        for col in feature_cols
        if col in reference_df.columns and col in current_df.columns
    ]
    drifted = [r for r in results if r["drift_detected"]]
    return {
        "alpha": alpha,
        "total_features": len(results),
        "drift_features": [r["feature"] for r in drifted],
        "details": results,
    }


# ---------------------------------------------------------------------------
# Class API  (standalone scripts and smoke tests)
# ---------------------------------------------------------------------------

class DataDriftMonitor:
    """Standalone drift monitor — loads reference data and runs KS tests."""

    def __init__(
        self,
        reference_data_path: Optional[str] = None,
        numeric_features: Optional[List[str]] = None,
    ):
        self.reference_data_path = reference_data_path or _resolve_data_path()
        self._numeric_features = numeric_features

    def load_reference_data(self) -> pd.DataFrame:
        if not os.path.exists(self.reference_data_path):
            raise FileNotFoundError(
                f"Reference data not found: {self.reference_data_path}"
            )
        return pd.read_csv(self.reference_data_path)

    def analyze_drift(
        self,
        current_data_df: pd.DataFrame,
        alpha: float = 0.05,
    ):
        """Run KS drift analysis and print a formatted report.

        Returns (drift_detected_globally: bool, drift_report: dict).
        """
        ref_df = self.load_reference_data()

        if self._numeric_features:
            numerical_features = self._numeric_features
        else:
            numerical_features = [
                c for c in ref_df.select_dtypes(include="number").columns
                if c in current_data_df.columns
            ]

        print("\n🔍 KS Data Drift Analysis")
        print("=" * 65)
        print(f"{'Feature':<20} {'KS-Stat':>10} {'p-value':>12} {'Status':>12}")
        print("-" * 65)

        drift_report: Dict[str, Any] = {}
        drift_detected_global = False

        for feature in numerical_features:
            if feature not in current_data_df.columns:
                continue
            ks_stat, p_value = ks_2samp(
                ref_df[feature].dropna(),
                current_data_df[feature].dropna(),
            )
            has_drift = p_value < alpha
            if has_drift:
                drift_detected_global = True
            drift_report[feature] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": has_drift,
            }
            status = "🚨 DRIFTED" if has_drift else "✅ STABLE"
            print(f"{feature:<20} {ks_stat:>10.4f} {p_value:>12.4e} {status:>12}")

        print("=" * 65)
        if drift_detected_global:
            print("⚠️  ALERT: Significant drift detected — consider retraining.")
        else:
            print("✅  Distributions stable.")
        return drift_detected_global, drift_report


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    monitor = DataDriftMonitor()
    print(f"Reference data: {monitor.reference_data_path}")

    print("\n🧪 TEST 1: Normal traffic (no drift expected)")
    try:
        raw_df = pd.read_csv(monitor.reference_data_path)
        monitor.analyze_drift(raw_df.sample(n=1000, random_state=42))
    except Exception as e:
        print(f"Test 1 failed: {e}")

    print("\n🧪 TEST 2: Shifted Age +8 (drift expected)")
    try:
        drifted = raw_df.sample(n=1000, random_state=101).copy()
        drifted["Age"] = drifted["Age"] + 8
        monitor.analyze_drift(drifted)
    except Exception as e:
        print(f"Test 2 failed: {e}")
