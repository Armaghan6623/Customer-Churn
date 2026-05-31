import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Guarantees root directory visibility across nested sub-packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

warnings.filterwarnings("ignore")


def _load_series(data_path: str, target_col: str = "ChurnCount") -> pd.Series:
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        if target_col not in df.columns:
            raise KeyError(f"target_col='{target_col}' not found. Available: {list(df.columns)}")
        return df[target_col]

    # Fallback: same synthetic structure as model.py
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    date_range = pd.date_range(start="2021-01-01", end="2025-12-01", freq="MS")
    N = len(date_range)
    trend = np.linspace(100, 180, N)
    seasonality = 35 * np.sin(2 * np.pi * np.arange(N) / 12)
    irregularity = np.random.normal(0, 10, N)
    churn_count = np.round(trend + seasonality + irregularity).astype(int)
    churn_count = np.clip(churn_count, 10, None)
    df = pd.DataFrame({"ChurnCount": churn_count}, index=date_range)
    df.index.name = "Date"
    df.to_csv(data_path)
    return df[target_col]


def _sarima_artifact_forecast(artifact_path: str, steps: int = 6) -> pd.Series:
    payload = joblib.load(artifact_path)
    last_date = payload["history_last_index"]
    history_series = payload["history_last_values"]

    # Recreate results container
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        history_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.smooth(payload["results_param"])
    forecast_res = results.get_forecast(steps=steps)

    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1), periods=steps, freq="MS"
    )
    return pd.Series(np.round(forecast_res.predicted_mean.values).astype(int), index=future_dates)


def _xgb_artifact_forecast(artifact_path: str, steps: int = 6) -> pd.Series:
    payload = joblib.load(artifact_path)
    model = payload["model"]
    lags = int(payload["lags"])
    last_window = payload["last_window"]
    last_date = payload["history_last_index"]

    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1), periods=steps, freq="MS"
    )

    window = last_window.astype(float).copy()
    preds = []

    for dt in future_dates:
        month = int(dt.month)
        x_row = np.concatenate([window, np.array([month], dtype=float)], axis=0)
        x_row = x_row.reshape(1, -1)
        y_hat = float(model.predict(x_row)[0])
        preds.append(y_hat)
        window = np.concatenate([window[1:], np.array([y_hat], dtype=float)], axis=0)

    return pd.Series(np.round(np.array(preds)).astype(int), index=future_dates)


def plot_forecasts(
    data_path: str = "data/raw/monthly_churn_trends.csv",
    sarima_artifact: str = "saved_models/forecast_model.joblib",
    xgb_artifact: str = "saved_models/xgb_forecast_model.joblib",
    out_path: str = "saved_models/forecast_plot.png",
    steps: int = 6,
):
    series = _load_series(data_path)
    series = series.sort_index()

    if not os.path.exists(sarima_artifact):
        raise FileNotFoundError(f"Missing SARIMA artifact: {sarima_artifact}. Run model.py first.")
    if not os.path.exists(xgb_artifact):
        raise FileNotFoundError(f"Missing XGB artifact: {xgb_artifact}. Run model.py first.")

    sarima_fc = _sarima_artifact_forecast(sarima_artifact, steps=steps)
    xgb_fc = _xgb_artifact_forecast(xgb_artifact, steps=steps)

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, label="Historical", linewidth=2)
    plt.plot(sarima_fc.index, sarima_fc.values, marker="o", label="SARIMAX forecast")
    plt.plot(xgb_fc.index, xgb_fc.values, marker="o", label="XGBoost forecast")

    plt.title("Churn Forecast (SARIMAX vs XGBoost)")
    plt.xlabel("Date")
    plt.ylabel("ChurnCount")
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"✅ Forecast graph saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    plot_forecasts()

