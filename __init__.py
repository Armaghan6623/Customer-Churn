# src/customer_crunch/forecasting/model.py
import os
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Guarantee system path mapping to root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

class ChurnForecaster:
    def __init__(self, model_dir="saved_models/forecasting"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "xgboost_forecaster.joblib")
        self.model = None

    def create_time_features(self, df, target_column='ChurnCount'):
        """
        Transforms a sequential DataFrame with a DatetimeIndex into a matrix 
        of lag, rolling statistical, and calendar features.
        """
        df = df.copy().sort_index()
        
        # 1. Calendar/Temporal features
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        # 2. Lag Features (Looking back in time)
        df['Lag_1'] = df[target_column].shift(1)
        df['Lag_2'] = df[target_column].shift(2)
        df['Lag_3'] = df[target_column].shift(3)
        
        # 3. Rolling Window Features (Capturing momentum/trends)
        df['Rolling_Mean_3'] = df[target_column].shift(1).rolling(window=3).mean()
        df['Rolling_Std_3'] = df[target_column].shift(1).rolling(window=3).std()
        
        # Drop rows with NaN values created by shifting/rolling features
        df = df.dropna()
        return df

    def train(self, data_path="data/raw/monthly_churn_trends.csv", target_column='ChurnCount'):
        """
        Loads time-series data, runs deterministic feature engineering, 
        splits data chronologically, and trains the XGBoost Regressor.
        """
        print("📈 Initializing Time-Series Forecasting Training Pipeline...")
        
        if not os.path.exists(data_path):
            # Fallback: Generate synthetic temporal data if your file isn't placed yet
            print("⚠️ Data file not found. Generating operational synthetic time-series history...")
            date_range = pd.date_range(start="2021-01-01", end="2026-04-01", freq="ME")
            synthetic_churn = 150 + np.sin(np.arange(len(date_range))) * 40 + np.random.normal(0, 15, len(date_range))
            df = pd.DataFrame(data={target_column: synthetic_churn}, index=date_range)
        else:
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Build features matrix
        featured_df = self.create_time_features(df, target_column)
        
        # Split into X and y
        X = featured_df.drop(columns=[target_column])
        y = featured_df[target_column]
        
        # Chronological Split (Last 6 months kept strictly for out-of-time evaluation)
        test_size = 6
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
        
        # Define and fit the forecasting model
        self.model = XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            objective='reg:squarederror',
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model performance
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
        
        print(f"✅ Training Complete. Out-of-Time Validation Metrics:")
        print(f" -> Mean Absolute Error (MAE): {mae:.2f} churned users")
        print(f" -> Root Mean Squared Error (RMSE): {rmse:.2f} churned users")
        
        # Save model artifact
        joblib.dump(self.model, self.model_path)
        print(f"💾 Forecasting artifact serialized successfully to: {self.model_path}")
        
    def forecast_future_steps(self, historical_data, steps=6, target_column='ChurnCount'):
        """
        Performs recursive multi-step forecasting into the future 
        using dynamically updating lag properties.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                raise FileNotFoundError("❌ Trained forecasting model artifact missing. Run train first!")

        # Start with the absolute latest sequence of historical points
        current_data = historical_data.copy().sort_index()
        future_forecasts = []
        future_dates = pd.date_range(start=current_data.index[-1] + pd.offsets.MonthEnd(), periods=steps, freq="ME")

        for next_date in future_dates:
            # Build features for the single next chronological step
            featured_df = self.create_time_features(current_data, target_column)
            
            # Extract the very last engineered row to predict the next step
            X_next = featured_df.drop(columns=[target_column]).iloc[[-1]]
            
            # Execute step prediction
            next_pred = float(self.model.predict(X_next)[0])
            future_forecasts.append(next_pred)
            
            # Append the predicted value back into the timeline recursively to feed the next lag step
            new_row = pd.DataFrame(data={target_column: [next_pred]}, index=[next_date])
            current_data = pd.concat([current_data, new_row])
            
        return pd.DataFrame(data={"Forecasted_Churn": future_forecasts}, index=future_dates)

if __name__ == "__main__":
    forecaster = ChurnForecaster()
    # Run standalone smoke test training loop
    forecaster.train()
    
    # Generate mock history to verify recursive forecasting engine functionality
    print("\n🔮 Testing recursive 6-month future forecasting window...")
    mock_history = pd.DataFrame(
        data={"ChurnCount": [160, 145, 170, 185, 190, 155]}, 
        index=pd.date_range(start="2025-11-01", end="2026-04-01", freq="ME")
    )
    predictions = forecaster.forecast_future_steps(mock_history, steps=6)
    print(predictions)