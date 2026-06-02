# src/customer_crunch/forecasting/model.py
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Silently plots without halting terminal runtime
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Guarantees root directory visibility across nested sub-packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

warnings.filterwarnings("ignore")  # Suppress technical convergence optimization warnings


def portable_sarima_payload(final_results, series):
    """Serialize SARIMA state using plain Python types (pandas-version safe)."""
    tail = series[-24:]
    return {
        "results_param": final_results.params.to_dict(),
        "history_last_index": series.index[-1].isoformat(),
        "history_last_values": tail.astype(float).tolist(),
        "history_index": [d.isoformat() for d in tail.index],
    }


def load_sarima_payload(artifact_path):
    """Load SARIMA artifact saved in portable or legacy layout."""
    payload = joblib.load(artifact_path)

    if isinstance(payload.get("results_param"), dict):
        params = pd.Series(payload["results_param"])
        history_series = pd.Series(
            payload["history_last_values"],
            index=pd.to_datetime(payload["history_index"]),
        )
        last_date = pd.Timestamp(payload["history_last_index"])
        return params, history_series, last_date

    params = payload["results_param"]
    history_series = payload["history_last_values"]
    last_date = payload["history_last_index"]
    return params, history_series, last_date


class ChurnSARIMAForecaster:
    def __init__(self, model_dir="saved_models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "forecast_model.joblib")
        self.model_results = None

    def load_or_create_timeline(self, data_path="data/raw/monthly_churn_trends.csv"):
        """
        Loads continuous monthly chronological logs. If missing, automatically compiles
        a 5-year historical timeline mapping macro trends, seasonality, and irregularities.
        """
        if os.path.exists(data_path):
            print(f"📊 Loading historical timeline dataset from: {data_path}")
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df

        print("⚠️ 'monthly_churn_trends.csv' not found. Synthesizing a 5-year macro sequence...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        # Build 60 months of historical sequence (Jan 2021 to Dec 2025)
        date_range = pd.date_range(start="2021-01-01", end="2025-12-01", freq="MS")
        N = len(date_range)

        # Mathematical components initialization
        trend = np.linspace(100, 180, N)                             # Structural upward bank scale growth
        seasonality = 35 * np.sin(2 * np.pi * np.arange(N) / 12)     # Annual 12-month cyclical wave
        irregularity = np.random.normal(0, 10, N)                    # Stochastic random noise

        churn_count = np.round(trend + seasonality + irregularity).astype(int)
        # Force natural lower bound integrity
        churn_count = np.clip(churn_count, 10, None)

        df = pd.DataFrame(data={"ChurnCount": churn_count}, index=date_range)
        df.index.name = 'Date'
        df.to_csv(data_path)
        print(f"💾 Completed data aggregation and saved matrix to: {data_path}")
        return df

    def train(self, data_path="data/raw/monthly_churn_trends.csv", target_col="ChurnCount"):
        """
        Fits a SARIMA(1, 1, 1) x (1, 1, 1)_12 parametric stochastic process model
        to capture trend differences and cyclical monthly variances.
        """
        print("\n📈 Initializing Classical SARIMA Time-Series Pipeline...")
        df = self.load_or_create_timeline(data_path)
        series = df[target_col]

        # Split out-of-time evaluation window (Last 6 months kept strictly for backtesting)
        test_months = 6
        train_series = series.iloc[:-test_months]
        test_series = series.iloc[-test_months:]

        # Define SARIMA Hyperparameters:
        # (p,d,q) = (1,1,1) -> Local Auto-Regression, Differencing, and Moving Averages
        # (P,D,Q)12 = (1,1,1)12 -> 12-Month Seasonal Auto-Regression, Differencing, and Moving Averages
        print("⚙️ Computing Maximum Likelihood Estimates for SARIMA parameters...")
        model = SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self.model_results = model.fit(disp=False)
        print("✅ Statistical convergence complete.")

        # Generate Backtest Predictions
        predictions = self.model_results.predict(
            start=test_series.index[0],
            end=test_series.index[-1],
            dynamic=True,
        )

        # Calculate Validation Loss
        mae = np.mean(np.abs(test_series - predictions))
        rmse = np.sqrt(np.mean((test_series - predictions) ** 2))

        print(f" -> Backtest Mean Absolute Error (MAE): {mae:.2f} churned users")
        print(f" -> Backtest Root Mean Squared Error (RMSE): {rmse:.2f} churned users")

        # Re-fit on full sequence history to prepare maximum horizon capability
        final_model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        final_results = final_model.fit(disp=False)

        payload = portable_sarima_payload(final_results, series)
        joblib.dump(payload, self.model_path)
        print(f"💾 SARIMA Model artifact successfully serialized to: {self.model_path}")

    def forecast_future_steps(self, steps=6):
        """
        Extrapolates out-of-sample future points beyond historical bounds
        using the saved mathematical weights.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Forecast artifact missing at '{self.model_path}'. Execute training first!")

        params, history_series, last_date = load_sarima_payload(self.model_path)

        model = SARIMAX(
            history_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.smooth(params)

        # Predict out-of-sample forecast steps
        forecast_res = results.get_forecast(steps=steps)
        forecast_means = forecast_res.predicted_mean

        # Create corresponding future calendar timestamp indices
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=steps, freq="MS")

        return pd.DataFrame(
            data={"Forecasted_Churn": np.round(forecast_means.values).astype(int)}, index=future_dates
        )


if __name__ == "__main__":
    # Compare SARIMAX vs XGBoost using the same last-6-month backtest split.

    # ----------------------
    # SARIMAX
    sarima_forecaster = ChurnSARIMAForecaster()
    sarima_forecaster.train()

    print("\n=========================================================")
    print("SARIMAX 6-month future forecast:")
    print("=========================================================")
    print(sarima_forecaster.forecast_future_steps(steps=6))

    # ----------------------
    # XGBoost
    try:
        from src.customer_crunch.forecasting.xgb_model import ChurnXGBForecaster
    except Exception:
        from .xgb_model import ChurnXGBForecaster

    xgb_forecaster = ChurnXGBForecaster(test_months=6, lags=12, horizon_steps=6)
    xgb_forecaster.train()

    print("\n=========================================================")
    print("XGBoost 6-month future forecast:")
    print("=========================================================")
    print(xgb_forecaster.forecast_future_steps(steps=6))

