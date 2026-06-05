"""customer_crunch.preprocessing.datacleaning

Utility functions for cleaning tabular customer churn datasets.

Covers:
- Missing value handling
- Duplicate removal
- Data consistency / schema enforcement

This module is intentionally framework-agnostic (pandas only) so it can be reused
in training pipelines, forecasting pipelines, and batch jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class CleaningConfig:
    """Configuration controlling cleaning behavior."""

    # Missing values
    missing_string_values: Tuple[str, ...] = (
        "",
        "na",
        "n/a",
        "none",
        "null",
        "nan",
        "missing",
        "unknown",
    )

    # If True, drop rows where all feature columns are missing.
    drop_all_missing_rows: bool = True

    # Dedup
    duplicate_keep: str = "first"  # pandas: "first" | "last" | False

    # Consistency
    # Columns that should be treated as categorical (will be coerced to string)
    categorical_columns: Optional[Sequence[str]] = None

    # Columns that should be treated as numeric
    numeric_columns: Optional[Sequence[str]] = None

    # Column-wise bounds: {"col": (min_inclusive, max_inclusive)}
    # Values outside bounds become NaN (then handled by missing-value strategy).
    numeric_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None

    # Columns that should be parsed as datetimes
    datetime_columns: Optional[Sequence[str]] = None


def _normalize_missing_markers(df: pd.DataFrame, missing_values: Iterable[str]) -> pd.DataFrame:
    """Replace common string markers and whitespace-only strings with np.nan."""
    df = df.copy()

    # Trim whitespace for object/string columns
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        df[c] = df[c].astype("string")
        df[c] = df[c].str.strip()

    df = df.replace({m: np.nan for m in missing_values})
    return df


def _coerce_numeric(df: pd.DataFrame, numeric_columns: Sequence[str]) -> pd.DataFrame:
    """Coerce specified columns to numeric type."""
    df = df.copy()
    for c in numeric_columns:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _apply_bounds(df: pd.DataFrame, bounds: Dict[str, Tuple[Optional[float], Optional[float]]]) -> pd.DataFrame:
    """Apply min/max bounds to numeric columns, setting out-of-bounds values to NaN."""
    df = df.copy()
    for c, (lower, upper) in bounds.items():
        if c not in df.columns:
            continue
        if lower is not None:
            df.loc[df[c] < lower, c] = np.nan
        if upper is not None:
            df.loc[df[c] > upper, c] = np.nan
    return df


def _coerce_datetime(df: pd.DataFrame, datetime_columns: Sequence[str]) -> pd.DataFrame:
    """Coerce specified columns to datetime type."""
    df = df.copy()
    for c in datetime_columns:
        if c not in df.columns:
            continue
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    return df


def _coerce_categorical(df: pd.DataFrame, categorical_columns: Sequence[str]) -> pd.DataFrame:
    """Coerce specified columns to categorical (string) type."""
    df = df.copy()
    for c in categorical_columns:
        if c not in df.columns:
            continue
        # Keep NaN as NaN (string dtype with pd.NA)
        df[c] = df[c].astype("string")
    return df


def _missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a missing value report for a dataframe."""
    rep = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .rename("missing_count")
        .to_frame()
    )
    rep["missing_percent"] = (rep["missing_count"] / max(len(df), 1)) * 100.0
    return rep


def clean_dataset(
    df: pd.DataFrame,
    config: CleaningConfig,
    *,
    target_columns: Optional[Sequence[str]] = None,
    required_columns: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Clean a dataset according to provided configuration.

    Parameters
    ----------
    df:
        Input dataframe.
    config:
        Cleaning rules and configuration.
    target_columns:
        Optional columns for your prediction target. Missing targets can optionally
        be dropped.
    required_columns:
        Columns that must be present (non-missing) after cleaning.

    Returns
    -------
    cleaned_df, artifacts
        cleaned_df: cleaned dataset.
        artifacts: dict containing simple reports (missing report).

    Raises
    ------
    ValueError:
        If required or target columns are not found in dataframe.
    """

    if df is None or len(df) == 0:
        return df, {"missing_before": pd.DataFrame(), "missing_after": pd.DataFrame()}

    cleaned = df.copy()

    # 1) Standardize missing markers
    cleaned = _normalize_missing_markers(cleaned, config.missing_string_values)
    missing_before = _missing_value_report(cleaned)

    # 2) Enforce schema/coercions
    categorical_cols = list(config.categorical_columns or [])
    numeric_cols = list(config.numeric_columns or [])
    datetime_cols = list(config.datetime_columns or [])

    # If columns are not specified, infer based on dtype.
    # For stability, we only coerce when config provided or inference is safe.
    if config.numeric_columns is not None:
        cleaned = _coerce_numeric(cleaned, numeric_cols)

    if config.numeric_bounds:
        cleaned = _apply_bounds(cleaned, config.numeric_bounds)

    if config.datetime_columns is not None:
        cleaned = _coerce_datetime(cleaned, datetime_cols)

    if config.categorical_columns is not None:
        cleaned = _coerce_categorical(cleaned, categorical_cols)

    # 3) Drop rows with all missing values (useful for sparse ingest)
    if config.drop_all_missing_rows:
        cleaned = cleaned.dropna(axis=0, how="all")

    # 4) Handle duplicates
    # Duplicate definition: entire row match (common default).
    # If you need subset dedup, extend here with a config.
    cleaned = cleaned.drop_duplicates(keep=config.duplicate_keep, ignore_index=True)

    # 5) Missing value handling (simple, robust strategies)
    # - Numeric: median
    # - Categorical: mode (first)
    # - Datetime: forward-fill then back-fill (per-column)

    # Identify numeric columns after coercion
    if config.numeric_columns is None:
        numeric_cols = cleaned.select_dtypes(include=["number", "bool"]).columns.tolist()

    # Identify categorical columns if not provided
    if config.categorical_columns is None:
        categorical_cols = cleaned.select_dtypes(include=["object", "string"]).columns.tolist()

    # Datetime
    if config.datetime_columns is None:
        datetime_cols = cleaned.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    # Numeric impute
    for c in numeric_cols:
        if c not in cleaned.columns:
            continue
        med = cleaned[c].median(skipna=True)
        if pd.isna(med):
            # if all missing, keep as NaN
            continue
        cleaned[c] = cleaned[c].fillna(med)

    # Categorical impute
    for c in categorical_cols:
        if c not in cleaned.columns:
            continue
        mode = cleaned[c].mode(dropna=True)
        if len(mode) == 0:
            continue
        cleaned[c] = cleaned[c].fillna(mode.iloc[0])

    # Datetime impute
    for c in datetime_cols:
        if c not in cleaned.columns:
            continue
        # forward fill then back fill
        cleaned[c] = cleaned[c].sort_index().ffill().bfill()

    # 6) Required columns consistency
    if required_columns is not None:
        for c in required_columns:
            if c not in cleaned.columns:
                raise ValueError(f"Required column '{c}' not found in dataframe.")
        cleaned = cleaned.dropna(subset=list(required_columns)).reset_index(drop=True)

    # 7) Target handling (optional)
    if target_columns is not None:
        for c in target_columns:
            if c not in cleaned.columns:
                raise ValueError(f"Target column '{c}' not found in dataframe.")
        cleaned = cleaned.dropna(subset=list(target_columns)).reset_index(drop=True)

    # Final missing report
    missing_after = _missing_value_report(cleaned)

    artifacts = {
        "missing_before": missing_before,
        "missing_after": missing_after,
    }
    return cleaned, artifacts


def get_cleaning_pipeline_artifacts(
    artifacts: Dict[str, pd.DataFrame],
    *,
    max_rows: int = 50,
) -> Dict[str, List[Dict[str, float]]]:
    """Convert artifacts dataframes to JSON-serializable dicts.

    Parameters
    ----------
    artifacts:
        Dictionary of artifact dataframes.
    max_rows:
        Maximum number of rows to include in output.

    Returns
    -------
    dict:
        Dictionary with the same keys, but dataframes converted to list of dicts.
    """

    out: Dict[str, List[Dict[str, float]]] = {}
    for k, v in artifacts.items():
        if isinstance(v, pd.DataFrame):
            out[k] = (
                v.head(max_rows)
                .reset_index()
                .rename(columns={"index": "column"})
                .to_dict(orient="records")
            )
    return out
