"""Feature engineering for churn prediction.

Addresses research gaps identified in:
  "Prediction of Customer Churn in Financial Sectors using ML" (2025)

Gap — Technology-driven and behavioral factors are underexplored:
  Raw features (Age, Balance, Tenure…) carry limited signal on their own.
  Interaction and ratio features expose relationships that tree models
  can learn more efficiently and that also improve interpretability.

Implemented features
--------------------
TenurePerProduct    : average tenure per product held — high values
                      suggest long-term, diversified customers (lower risk)
BalanceSalaryRatio  : balance relative to salary — captures financial
                      stress; very low or very high ratios are informative
AgeGroup            : ordinal binning [18-30 → 1, 30-40 → 2, …] — encodes
                      life-stage risk segments identified in the literature
IsZeroBalance       : flag for customers with zero balance — strong churn
                      signal in banking datasets
IsHighValue         : flag for high-balance, multi-product customers —
                      important segment for targeted retention
ProductsPerYear     : NumOfProducts / (Tenure + 1) — rapid product
                      acquisition may signal exploratory / at-risk behaviour
CreditScoreGroup    : ordinal binning of CreditScore into risk tiers

All transformations are implemented as a sklearn-compatible Transformer
so they plug directly into a Pipeline or ColumnTransformer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Pure-function helpers (usable outside sklearn)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with engineered columns appended.

    Safe to call on raw input — missing source columns are handled
    gracefully (feature is simply not added).
    """
    out = df.copy()

    # -- Tenure per product --------------------------------------------------
    if "Tenure" in out.columns and "NumOfProducts" in out.columns:
        out["TenurePerProduct"] = out["Tenure"] / (out["NumOfProducts"] + 1)

    # -- Balance / salary ratio -----------------------------------------------
    if "Balance" in out.columns and "EstimatedSalary" in out.columns:
        out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + 1)

    # -- Age group (ordinal life-stage buckets) --------------------------------
    if "Age" in out.columns:
        out["AgeGroup"] = pd.cut(
            out["Age"],
            bins=[0, 30, 40, 50, 60, 200],
            labels=[1, 2, 3, 4, 5],
            right=True,
        ).astype(float)

    # -- Zero-balance flag ----------------------------------------------------
    if "Balance" in out.columns:
        out["IsZeroBalance"] = (out["Balance"] == 0).astype(int)

    # -- High-value customer flag ---------------------------------------------
    if "Balance" in out.columns and "NumOfProducts" in out.columns:
        balance_median = out["Balance"].median()
        out["IsHighValue"] = (
            (out["Balance"] > balance_median) & (out["NumOfProducts"] >= 2)
        ).astype(int)

    # -- Products acquired per year -------------------------------------------
    if "NumOfProducts" in out.columns and "Tenure" in out.columns:
        out["ProductsPerYear"] = out["NumOfProducts"] / (out["Tenure"] + 1)

    # -- Credit score group (risk tiers) --------------------------------------
    if "CreditScore" in out.columns:
        out["CreditScoreGroup"] = pd.cut(
            out["CreditScore"],
            bins=[0, 500, 600, 700, 800, 1000],
            labels=[1, 2, 3, 4, 5],
            right=True,
        ).astype(float)

    return out


# ---------------------------------------------------------------------------
# sklearn Transformer wrapper
# ---------------------------------------------------------------------------

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that appends engineered features.

    Parameters
    ----------
    add_features : list[str] | None
        Which engineered features to add.  None means all of them.
    """

    ALL_FEATURES = [
        "TenurePerProduct",
        "BalanceSalaryRatio",
        "AgeGroup",
        "IsZeroBalance",
        "IsHighValue",
        "ProductsPerYear",
        "CreditScoreGroup",
    ]

    def __init__(self, add_features: list | None = None):
        self.add_features = add_features  # None = add all

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        if isinstance(X, np.ndarray):
            raise ValueError("ChurnFeatureEngineer requires a pandas DataFrame input.")
        out = engineer_features(X)
        if self.add_features is not None:
            keep = list(X.columns) + [f for f in self.add_features if f in out.columns]
            out = out[keep]
        return out

    def get_feature_names_out(self, input_features=None):
        dummy = pd.DataFrame(
            columns=["CreditScore", "Age", "Tenure", "Balance",
                     "NumOfProducts", "HasCrCard", "IsActiveMember",
                     "EstimatedSalary"],
            data=[[600, 40, 5, 75000, 2, 1, 1, 100000]],
        )
        return list(engineer_features(dummy).columns)


# ---------------------------------------------------------------------------
# List of new numeric columns produced — used by DatasetConfig / train.py
# ---------------------------------------------------------------------------

ENGINEERED_NUMERIC_FEATURES = [
    "TenurePerProduct",
    "BalanceSalaryRatio",
    "AgeGroup",
    "IsZeroBalance",
    "IsHighValue",
    "ProductsPerYear",
    "CreditScoreGroup",
]
