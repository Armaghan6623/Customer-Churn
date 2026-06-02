import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

# Guarantees root directory visibility across nested sub-packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

warnings.filterwarnings("ignore")


class ChurnXGBForecaster:
    """Lag-feature supervised forecaster using XGBoost."""

    def __init__(
        self,
        model_dir: str = "saved_models",
        lags: int = 12,
        test_months: int = 6,
        horizon_steps: int = 6,
    ):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "xgb_forecast_model.joblib")

        self.lags = lags
        self.test_months = test_months
        self.horizon_steps = horizon_steps

        self.model = None
        self.metrics_ = None

    @staticmethod
    def _make_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
        # Simple calendar features that help XGB learn seasonality.
        return pd.DataFrame(
            {
                "month": index.month.astype(int),
            },
            index=index,
        )

    def load_or_create_timeline(self, data_path: str = "data/raw/monthly_churn_trends.csv"):
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            return df

        # Fallback: synthesize same structure as SARIMA module
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
        return df

    def _make_lag_supervised(self, series: pd.Series):
        # For each time t, build features from y[t-lags ... t-1], and target y[t]
        y = series.astype(float)
        feats = []
        targets = []
        feat_index = []

        for t in range(self.lags, len(y)):
            feats.append(y.iloc[t - self.lags : t].values)
            targets.append(y.iloc[t])
            feat_index.append(y.index[t])

        X_lags = np.vstack(feats)  # shape: (n_samples, lags)
        y_target = np.array(targets, dtype=float)
        idx = pd.DatetimeIndex(feat_index)

        # Add calendar features
        time_feats = self._make_time_features(idx)

        # Combine
        X = np.concatenate([X_lags, time_feats.values], axis=1)
        feature_names = [f"lag_{i}" for i in range(1, self.lags + 1)] + ["month"]
        X_df = pd.DataFrame(X, columns=feature_names, index=idx)

        return X_df, y_target

    def _iterative_forecast(self, last_window: np.ndarray, future_time_index: pd.DatetimeIndex, base_month_features: bool = True):
        # last_window: array of length lags, representing the most recent lags in chronological order
        window = last_window.astype(float).copy()
        preds = []

        for dt in future_time_index:
            month = int(dt.month)
            x_row = np.concatenate([window, np.array([month], dtype=float)], axis=0)
            x_row = x_row.reshape(1, -1)
            y_hat = float(self.model.predict(x_row)[0])
            preds.append(y_hat)

            # update window: drop oldest, append prediction
            window = np.concatenate([window[1:], np.array([y_hat], dtype=float)], axis=0)

        return np.array(preds, dtype=float)

    def train(self, data_path: str = "data/raw/monthly_churn_trends.csv", target_col: str = "ChurnCount"):
        print("\n🔧 Initializing XGBoost Time-Series Pipeline (lag-feature supervised)...")
        df = self.load_or_create_timeline(data_path)
        series = df[target_col]

        # Chronological split: reserve last `test_months` months for evaluation
        test_end_idx = len(series) - self.test_months
        train_series = series.iloc[:test_end_idx]
        test_series = series.iloc[test_end_idx:]

        # Build supervised datasets using lags
        X_train, y_train = self._make_lag_supervised(train_series)

        # For test targets, we need to include lags from train tail as well.
        # Build combined window for feature creation, but evaluate only on the test portion.
        combined_series = pd.concat([train_series, test_series])
        X_all, y_all = self._make_lag_supervised(combined_series)

        test_mask = X_all.index.isin(test_series.index)
        X_test = X_all.loc[test_mask]
        y_test = y_all[test_mask]

        # Model training
        self.model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
        )

        self.model.fit(X_train, y_train)

        # Backtest predictions (one-step ahead using actual lag features for MAE/RMSE)
        preds = self.model.predict(X_test)
        mae = float(np.mean(np.abs(y_test - preds)))
        rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))

        print(f" -> Backtest Mean Absolute Error (MAE): {mae:.2f} churned users")
        print(f" -> Backtest Root Mean Squared Error (RMSE): {rmse:.2f} churned users")

        # Serialize for future iterative forecasting
        payload = {
            "model": self.model,
            "lags": self.lags,
            "last_window": series.values[-self.lags:],
            "history_last_index": series.index[-1],
            "test_months": self.test_months,
        }
        joblib.dump(payload, self.model_path)

        self.metrics_ = {"mae": mae, "rmse": rmse}
        print(f"📦 XGB Model artifact successfully serialized to: {self.model_path}")

    def forecast_future_steps(self, steps: int = 6):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Forecast artifact missing at '{self.model_path}'. Execute training first!")

        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        lags = int(payload["lags"])

        last_date = payload["history_last_index"]
        last_window = payload["last_window"]

        future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        preds = self._iterative_forecast(last_window=last_window, future_time_index=future_dates)

        return pd.DataFrame(
            data={"Forecasted_Churn": np.round(preds).astype(int)},
            index=future_dates,
        )


if __name__ == "__main__":
    forecaster = ChurnXGBForecaster()
    forecaster.train()
    print("\n📈 Executing 6-Month Forward-Horizon Iterative Forecast Test...")
    print("=========================================================")
    print(forecaster.forecast_future_steps(steps=6))
    print("=========================================================")

