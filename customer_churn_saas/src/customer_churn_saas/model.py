"""Model entry points for the SaaS layout.

This repo already contains a working implementation under src/customer_crunch.
We re-expose compatible interfaces here.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChurnModelConfig:
    churn_pipeline_path: str = "saved_models/churn_pipeline.joblib"
    sarima_artifact_path: str = "saved_models/forecast_model.joblib"
    xgb_artifact_path: str = "saved_models/xgb_forecast_model.joblib"


class ChurnSaaSModel:
    def __init__(self, config: Optional[ChurnModelConfig] = None):
        self.config = config or ChurnModelConfig()

    def predict_customer_churn_probability(self, customer_features: dict) -> dict:
        from src.customer_crunch.classification.predict import predict_single_customer

        return predict_single_customer(
            customer_features,
            model_path=self.config.churn_pipeline_path,
        )

    def generate_future_forecast(self, steps: int = 6):
        # Always (re)generate plot so the UI has forecast_plot.png
        from src.customer_crunch.forecasting.trends import plot_forecasts

        out_path = "saved_models/forecast_plot.png"
        plot_forecasts(
            out_path=out_path,
            steps=steps,
            sarima_artifact=self.config.sarima_artifact_path,
            xgb_artifact=self.config.xgb_artifact_path,
        )

        from src.customer_crunch.forecasting.model import ChurnSARIMAForecaster
        from src.customer_crunch.forecasting.xgb_model import ChurnXGBForecaster

        sarima = ChurnSARIMAForecaster()
        sarima_df = sarima.forecast_future_steps(steps=steps)

        xgb = ChurnXGBForecaster(test_months=6, lags=12, horizon_steps=steps)
        xgb_df = xgb.forecast_future_steps(steps=steps)

        return {
            "sarima": sarima_df,
            "xgb": xgb_df,
            "plot_path": out_path,
        }

