"""Dataset-agnostic configuration schema for the churn pipeline.

Drop this file next to train.py and pass a DatasetConfig instance to
train_customer_churn_model() / predict_single_customer() so the system
works with any binary-classification dataset — not just the Kaggle bank
churn CSV.

Usage example (custom dataset):
    from classification.dataset_config import DatasetConfig

    cfg = DatasetConfig(
        target_col="Churn",
        drop_cols=["CustomerID", "Phone"],
        numeric_features=["tenure", "MonthlyCharges", "TotalCharges"],
        categorical_features=["Contract", "InternetService", "PaymentMethod"],
        feature_bounds={"tenure": (0, 100), "MonthlyCharges": (0, 200)},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DatasetConfig:
    """Describes the structure of a binary-churn dataset.

    Attributes
    ----------
    target_col:
        Name of the binary target column (1 = churn, 0 = retained).
    drop_cols:
        Columns to discard before training (IDs, free-text, etc.).
    numeric_features:
        Columns to pass through StandardScaler.
    categorical_features:
        Columns to pass through OneHotEncoder.
    feature_bounds:
        Optional per-feature (min, max) validation ranges used at
        inference time.  Keys must match column names.
    drift_features:
        Subset of numeric_features to monitor for distribution drift.
        Defaults to all numeric_features when empty.
    model_filename:
        Filename (not path) for the serialised pipeline artifact.
    """

    target_col: str = "Exited"
    drop_cols: List[str] = field(default_factory=lambda: ["CustomerId", "Surname", "RowNumber"])
    numeric_features: List[str] = field(
        default_factory=lambda: [
            "CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
        ]
    )
    categorical_features: List[str] = field(
        default_factory=lambda: ["Geography", "Gender"]
    )
    feature_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "Age": (0, 120),
            "CreditScore": (0, 850),
        }
    )
    drift_features: List[str] = field(default_factory=list)
    model_filename: str = "churn_pipeline.joblib"

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def all_feature_cols(self) -> List[str]:
        """All feature columns expected at inference time."""
        return self.numeric_features + self.categorical_features

    @property
    def effective_drift_features(self) -> List[str]:
        """Numeric features to monitor; falls back to all numeric_features."""
        return self.drift_features if self.drift_features else self.numeric_features

    def validate_row(self, row: dict) -> None:
        """Raise ValueError for missing columns or out-of-bound values."""
        missing = [c for c in self.all_feature_cols if c not in row]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for col, (lo, hi) in self.feature_bounds.items():
            if col in row:
                val = float(row[col])
                if not (lo <= val <= hi):
                    raise ValueError(
                        f"Column '{col}' value {val} is outside allowed range [{lo}, {hi}]."
                    )


# ---------------------------------------------------------------------------
# Built-in presets — add your own here or construct DatasetConfig inline
# ---------------------------------------------------------------------------

KAGGLE_BANK_CHURN = DatasetConfig()  # default values match the Kaggle dataset

TELCO_CHURN = DatasetConfig(
    target_col="Churn",
    drop_cols=["customerID"],
    numeric_features=["tenure", "MonthlyCharges", "TotalCharges"],
    categorical_features=[
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ],
    feature_bounds={"tenure": (0, 100), "MonthlyCharges": (0, 200)},
    model_filename="telco_churn_pipeline.joblib",
)
