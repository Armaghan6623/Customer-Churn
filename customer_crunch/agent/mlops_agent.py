"""MLOps agent: drift monitoring, self-healing retrain, and telemetry."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import joblib
import pandas as pd

_AGENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

from classification.dataset_config import DatasetConfig, KAGGLE_BANK_CHURN


def _resolve_data_path(*parts: str) -> str:
    """Resolve the reference dataset path across common project layouts.

    Tries the new canonical filename first (customer_churn_dataset.csv),
    then falls back to the legacy Kaggle filename for local dev environments.
    """
    canonical_names = [
        "customer_churn_dataset.csv",
        "Churn_Modelling kaggel.csv",
    ]
    base_dirs = [
        os.path.join(_AGENT_ROOT, "data"),
        os.path.join(_AGENT_ROOT, "data", "raw"),
        os.path.join(os.getcwd(), "customer_crunch", "data"),
        os.path.join(os.getcwd(), "customer_crunch", "data", "raw"),
        os.path.join(os.getcwd(), "data"),
        os.path.join(os.getcwd(), "data", "raw"),
        "/app/customer_crunch/data",
        "/app/customer_crunch/data/raw",
        "/app/data",
    ]
    for base in base_dirs:
        for name in canonical_names:
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p
    # fallback — canonical path used in the Docker image
    return os.path.join(_AGENT_ROOT, "data", "customer_churn_dataset.csv")


def _load_config_from_artifact(model_path: str) -> DatasetConfig:
    """Extract DatasetConfig from a saved artifact, or fall back to the Kaggle default."""
    try:
        artifact = joblib.load(model_path)
        if isinstance(artifact, dict) and "config" in artifact:
            return artifact["config"]
    except Exception:
        pass
    return KAGGLE_BANK_CHURN


class MLOpsAgent:
    """Monitors data drift and triggers retraining when thresholds are exceeded.

    The set of features monitored for drift is derived from the DatasetConfig
    embedded in the model artifact, so the agent works with any dataset.
    """

    def __init__(
        self,
        model_path: str,
        reference_data_path: Optional[str] = None,
        config: Optional[DatasetConfig] = None,
    ):
        self.model_path = model_path
        # Load config from artifact if not explicitly provided
        self._config: DatasetConfig = config or _load_config_from_artifact(model_path)
        self.reference_data_path = reference_data_path or _resolve_data_path()
        self._log: list[str] = []

    @property
    def NUMERIC_FEATURES(self) -> List[str]:
        """Drift-monitored features — driven by the dataset config."""
        return self._config.effective_drift_features

    def _log_line(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {msg}"
        self._log.append(line)

    def get_event_log(self) -> str:
        return "\n".join(self._log[-50:]) if self._log else "No MLOps events yet."

    def _load_reference(self) -> pd.DataFrame:
        if not os.path.exists(self.reference_data_path):
            raise FileNotFoundError(
                f"Reference dataset not found: {self.reference_data_path}"
            )
        return pd.read_csv(self.reference_data_path)

    def _sample_current(self, n: int = 800, seed: int = 42) -> pd.DataFrame:
        ref = self._load_reference()
        n = min(n, len(ref))
        return ref.sample(n=n, random_state=seed)

    def run_drift_scan(
        self,
        alpha: float = 0.05,
        simulate_drift: bool = False,
        sample_size: int = 800,
    ) -> Dict[str, Any]:
        from aiops.drift import monitor_drift

        ref = self._load_reference()
        current = self._sample_current(n=sample_size)

        if simulate_drift:
            current = current.copy()
            current["Age"] = current["Age"] + 10
            self._log_line("Simulated drift: shifted Age +10 years for demo.")

        report = monitor_drift(
            reference_df=ref,
            current_df=current,
            feature_cols=[c for c in self.NUMERIC_FEATURES if c in ref.columns],
            alpha=alpha,
        )
        self._log_line(
            f"Drift scan complete — {len(report['drift_features'])}/{report['total_features']} "
            f"features drifted (alpha={alpha})."
        )
        return report

    @staticmethod
    def format_drift_report(report: Dict[str, Any]) -> str:
        lines = [
            f"**Drift summary** (alpha={report.get('alpha', 0.05)})",
            f"- Features checked: {report.get('total_features', 0)}",
            f"- Drifted: {len(report.get('drift_features', []))}",
        ]
        if report.get("drift_features"):
            lines.append(f"- **Alert features:** {', '.join(report['drift_features'])}")
        else:
            lines.append("- **Status:** distributions stable")

        lines.append("\n| Feature | KS stat | p-value | Drift? |")
        lines.append("|---|---:|---:|:---:|")
        for row in report.get("details", []):
            flag = "yes" if row.get("drift_detected") else "no"
            lines.append(
                f"| {row.get('feature', '?')} | {row.get('statistic', 0):.4f} | "
                f"{row.get('p_value', 1):.4e} | {flag} |"
            )
        return "\n".join(lines)

    def _retrain_callback(self) -> Dict[str, str]:
        from classification.train import train_customer_churn_model

        save_dir = os.path.dirname(self.model_path) or "saved_models"
        os.makedirs(save_dir, exist_ok=True)
        self._log_line(f"Retraining model from {self.reference_data_path} ...")
        # Pass the dataset config so the retrained model matches the current schema
        out_path = train_customer_churn_model(
            self.reference_data_path, save_dir, config=self._config
        )
        self._log_line(f"Model saved to {out_path}")
        return {"status": "retrained", "path": out_path}

    def run_self_heal(
        self,
        drift_report: Dict[str, Any],
        drift_threshold: int = 1,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        from aiops.self_heal import self_heal_if_needed

        if dry_run:
            drifted = drift_report.get("drift_features", [])
            would_trigger = len(drifted) >= drift_threshold
            return {
                "status": "dry_run",
                "would_retrain": would_trigger,
                "drift_features": drifted,
                "drift_threshold": drift_threshold,
            }

        callback: Callable[[], Any] = self._retrain_callback
        result = self_heal_if_needed(
            drift_report=drift_report,
            retrain_callback=callback,
            drift_threshold=drift_threshold,
            model_path=None,
        )
        self._log_line(f"Self-heal result: {result.get('status')} — {result}")
        return result

    def run_full_cycle(
        self,
        drift_threshold: int = 1,
        alpha: float = 0.05,
        simulate_drift: bool = False,
        dry_run: bool = False,
    ) -> str:
        report = self.run_drift_scan(alpha=alpha, simulate_drift=simulate_drift)
        heal = self.run_self_heal(
            drift_report=report,
            drift_threshold=drift_threshold,
            dry_run=dry_run,
        )
        parts = [
            self.format_drift_report(report),
            "",
            "**Self-heal**",
            f"- Status: `{heal.get('status')}`",
        ]
        if heal.get("drift_features") is not None:
            parts.append(f"- Drift features: {heal.get('drift_features')}")
        if heal.get("would_retrain") is not None:
            parts.append(f"- Would retrain: {heal.get('would_retrain')}")
        parts.append("\n**Event log (recent)**\n" + self.get_event_log())
        return "\n".join(parts)
