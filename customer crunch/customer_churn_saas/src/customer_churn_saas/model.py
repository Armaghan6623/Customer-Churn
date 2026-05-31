"""Model entry points for the SaaS layout."""

import os
import sys
from dataclasses import dataclass
from typing import Optional

# Resolve monorepo root (local) vs Space root (HF deploy bundle).
_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_crunch_root() -> str:
    if os.path.isdir(os.path.join(_APP_ROOT, "classification")):
        return _APP_ROOT
    parent = os.path.dirname(_APP_ROOT)
    if os.path.isdir(os.path.join(parent, "classification")):
        return parent
    return _APP_ROOT


_CRUNCH_ROOT = _resolve_crunch_root()
if _CRUNCH_ROOT not in sys.path:
    sys.path.insert(0, _CRUNCH_ROOT)


@dataclass
class ChurnModelConfig:
    churn_pipeline_path: str = "saved_models/churn_pipeline.joblib"
    sarima_artifact_path: str = "saved_models/forecast_model.joblib"
    xgb_artifact_path: str = "saved_models/xgb_forecast_model.joblib"


class ChurnSaaSModel:
    def __init__(self, config: Optional[ChurnModelConfig] = None):
        self.config = config or ChurnModelConfig()

    def predict_customer_churn_probability(self, customer_features: dict) -> dict:
        from classification.predict import predict_single_customer

        return predict_single_customer(
            customer_features,
            model_path=self.config.churn_pipeline_path,
        )

    def generate_future_forecast(self, steps: int = 6):
        from forecasting.model import ChurnSARIMAForecaster
        from forecasting.trends import plot_forecasts
        from forecasting.xgb_model import ChurnXGBForecaster

        data_path = os.path.join(_CRUNCH_ROOT, "monthly_churn_trends.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(_CRUNCH_ROOT, "data/raw/monthly_churn_trends.csv")
        out_path = self.config.sarima_artifact_path.replace(
            "forecast_model.joblib", "forecast_plot.png"
        )
        if out_path == self.config.sarima_artifact_path:
            out_path = "saved_models/forecast_plot.png"

        plot_forecasts(
            data_path=data_path,
            out_path=out_path,
            steps=steps,
            sarima_artifact=self.config.sarima_artifact_path,
            xgb_artifact=self.config.xgb_artifact_path,
        )

        model_dir = os.path.dirname(self.config.sarima_artifact_path) or "saved_models"
        sarima = ChurnSARIMAForecaster(model_dir=model_dir)
        sarima_df = sarima.forecast_future_steps(steps=steps)

        xgb = ChurnXGBForecaster(
            model_dir=model_dir, test_months=6, lags=12, horizon_steps=steps
        )
        xgb_df = xgb.forecast_future_steps(steps=steps)

        return {
            "sarima": sarima_df,
            "xgb": xgb_df,
            "plot_path": out_path,
        }
