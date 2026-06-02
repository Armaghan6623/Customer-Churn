"""Plot forecasting results (historical + SARIMAX vs XGBoost forecasts).

Creates:
- saved_models/forecast_plot.png (default)

Run:
python -m src.customer_crunch.forecasting.plot_forecasts
"""

import os

from src.customer_crunch.forecasting.trends import plot_forecasts


if __name__ == "__main__":
    out_path = os.path.join("saved_models", "forecast_plot.png")
    plot_forecasts(out_path=out_path)

