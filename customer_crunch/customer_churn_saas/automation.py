import os
from typing import Dict, Any


def collect_telemetry(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stub telemetry collector.

    In a real deployment, this would emit events to a monitoring system.
    """
    payload = dict(payload)
    payload.setdefault("telemetry_version", 1)
    payload.setdefault("cwd", os.getcwd())
    return payload


def ks_test_drift(old_values, new_values) -> Dict[str, Any]:
    """Lightweight drift tracker using KS test.

    Note: This is a minimal implementation; for production consider
    streaming histograms / efficient approximations.
    """
    from scipy.stats import ks_2samp

    res = ks_2samp(old_values, new_values)
    return {
        "ks_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "drift_detected": bool(res.pvalue < 0.05),
    }
