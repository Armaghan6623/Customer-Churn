"""Model training — XGBoost with SMOTE, stratified CV, and business metrics.

Addresses two research gaps identified in:
  "Customer Churn Prediction: A Systematic Review" (2025)

  Gap 1 — Class imbalance: adds SMOTE oversampling + stratified k-fold CV
           (scale_pos_weight alone is insufficient for heavily skewed datasets)

  Gap 4 — Business metrics: reports EMP, profit curve, ROI alongside
           standard precision/recall/F1 after every training run.
"""
from __future__ import annotations

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier

from classification.dataset_config import DatasetConfig, KAGGLE_BANK_CHURN
from classification.business_metrics import (
    cost_benefit_matrix,
    expected_maximum_profit,
    roi_of_retention,
    format_business_report,
    plot_profit_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# SMOTE is optional — gracefully skipped if imbalanced-learn is not installed
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False
    ImbPipeline = Pipeline  # fall back silently


def _build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def _run_stratified_cv(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline,
    n_splits: int = 5,
) -> dict:
    """5-fold stratified cross-validation. Returns mean ± std for key metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["f1", "roc_auc", "average_precision", "precision", "recall"]
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    summary = {}
    for metric in scoring:
        key = f"test_{metric}"
        summary[metric] = {
            "mean": round(float(results[key].mean()), 4),
            "std":  round(float(results[key].std()),  4),
        }
    return summary


def train_customer_churn_model(
    data_path: str,
    save_dir: str,
    config: DatasetConfig = KAGGLE_BANK_CHURN,
    use_smote: bool = True,
    cv_splits: int = 5,
    clv: float = 200.0,
    offer_cost: float = 20.0,
) -> str:
    """Train a churn pipeline with SMOTE, stratified CV, and business metrics.

    Parameters
    ----------
    data_path   : Path to the CSV dataset.
    save_dir    : Directory to save the serialised artifact.
    config      : DatasetConfig schema — defaults to Kaggle bank churn.
    use_smote   : Apply SMOTE oversampling to the training split.
                  Automatically disabled if imbalanced-learn is not installed.
    cv_splits   : Number of stratified CV folds (0 = skip CV).
    clv         : Customer Lifetime Value used for business metric calculation.
    offer_cost  : Cost of sending one retention offer.

    Returns
    -------
    str : Absolute path to the saved model artifact.
    """
    print(f"📥  Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # ── Feature / target split ────────────────────────────────────────────
    X = df.drop(columns=config.drop_cols + [config.target_col], errors="ignore")
    y = df[config.target_col]

    known_cols = [c for c in config.all_feature_cols if c in X.columns]
    X = X[known_cols]

    numeric_features     = [c for c in config.numeric_features     if c in X.columns]
    categorical_features = [c for c in config.categorical_features if c in X.columns]

    # ── Class imbalance report ────────────────────────────────────────────
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    imbalance_ratio = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"⚖️   Class distribution — retained: {n_neg:,}  |  churned: {n_pos:,}"
          f"  |  ratio: {imbalance_ratio:.2f}:1")

    scale_weight = imbalance_ratio  # kept for XGBoost internal balancing

    # ── Train / test split (stratified) ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Build pipeline ────────────────────────────────────────────────────
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    classifier   = XGBClassifier(
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    smote_applied = False
    if use_smote and _SMOTE_AVAILABLE:
        # SMOTE operates in the transformed (numeric) space.
        # We preprocess first, then oversample, then fit the classifier.
        print("🔄  Applying SMOTE oversampling to training split...")
        preprocessor.fit(X_train)
        X_train_t = preprocessor.transform(X_train)
        X_test_t  = preprocessor.transform(X_test)

        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train_t, y_train)

        n_synth = len(y_resampled) - len(y_train)
        print(f"   Synthetic minority samples added: {n_synth:,}")
        print(f"   Post-SMOTE distribution — retained: {(y_resampled==0).sum():,}"
              f"  |  churned: {(y_resampled==1).sum():,}")

        classifier.fit(X_resampled, y_resampled)
        smote_applied = True

        # Wrap in sklearn Pipeline for consistent predict/predict_proba API
        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   classifier),
        ])
        # Preprocessor already fitted; mark it so pipeline.predict works correctly
        # We rebuild a clean pipeline carrying the fitted objects
        from sklearn.base import clone
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),   # already fitted
            ("classifier",   classifier),     # already fitted
        ])

    elif use_smote and not _SMOTE_AVAILABLE:
        print("⚠️   imbalanced-learn not installed — SMOTE skipped."
              " Run: pip install imbalanced-learn")
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   classifier),
        ])
        X_train_t = None
        X_test_t  = None
    else:
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   classifier),
        ])
        X_train_t = None
        X_test_t  = None

    # ── Fit (if SMOTE was not applied, train the whole pipeline) ─────────
    if not smote_applied:
        print("🚀  Training XGBoost pipeline...")
        final_pipeline.fit(X_train, y_train)

    # ── Stratified cross-validation ───────────────────────────────────────
    if cv_splits > 1:
        print(f"\n📐  Running {cv_splits}-fold stratified cross-validation...")
        cv_pipeline = Pipeline([
            ("preprocessor", _build_preprocessor(numeric_features, categorical_features)),
            ("classifier",   XGBClassifier(
                scale_pos_weight=scale_weight,
                random_state=42,
                eval_metric="logloss",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
            )),
        ])
        cv_results = _run_stratified_cv(X, y, cv_pipeline, n_splits=cv_splits)
        print(f"{'Metric':<20} {'Mean':>8} {'Std':>8}")
        print("-" * 38)
        for metric, vals in cv_results.items():
            print(f"  {metric:<18} {vals['mean']:>8.4f} ±{vals['std']:.4f}")
    else:
        cv_results = {}

    # ── Hold-out test evaluation ──────────────────────────────────────────
    print("\n📊  Hold-out test set evaluation:")
    if smote_applied:
        y_pred      = classifier.predict(X_test_t)
        y_prob      = classifier.predict_proba(X_test_t)[:, 1]
    else:
        y_pred      = final_pipeline.predict(X_test)
        y_prob      = final_pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr  = average_precision_score(y_test, y_prob)
    print(f"  AUC-ROC : {auc_roc:.4f}")
    print(f"  AUC-PR  : {auc_pr:.4f}   (more informative under class imbalance)")

    # ── Business metrics ──────────────────────────────────────────────────
    print("\n💰  Computing business-oriented metrics...")
    cb         = cost_benefit_matrix(clv=clv, offer_cost=offer_cost)
    emp_result = expected_maximum_profit(np.array(y_test), y_prob, cb=cb)
    roi_result = roi_of_retention(
        np.array(y_test), y_prob,
        opt_threshold=emp_result["optimal_threshold"],
        cb=cb,
    )

    report = format_business_report(emp_result, roi_result)
    # Print plain-text version
    plain = report.replace("**", "").replace("##", "").replace("|", " | ")
    print(plain)

    # Save profit curve plot
    os.makedirs(save_dir, exist_ok=True)
    profit_plot_path = os.path.join(save_dir, "profit_curve.png")
    plot_profit_curve(
        emp_result["curve_df"],
        opt_threshold=emp_result["optimal_threshold"],
        emp=emp_result["emp"],
        save_path=profit_plot_path,
    )
    print(f"📈  Profit curve saved to: {profit_plot_path}")

    # ── Save artifact ─────────────────────────────────────────────────────
    model_path = os.path.join(save_dir, config.model_filename)
    joblib.dump(
        {
            "pipeline":           final_pipeline,
            "config":             config,
            "smote_applied":      smote_applied,
            "cv_results":         cv_results,
            "business_metrics": {
                "emp":              emp_result["emp"],
                "emp_per_customer": emp_result["emp_per_customer"],
                "optimal_threshold":emp_result["optimal_threshold"],
                "roi_pct":          roi_result["roi_pct"],
                "lift_over_random": emp_result["lift_over_random"],
                "auc_roc":          round(auc_roc, 4),
                "auc_pr":           round(auc_pr,  4),
            },
        },
        model_path,
    )
    print(f"\n✅  Pipeline saved to: {model_path}")
    return model_path


if __name__ == "__main__":
    raw_data_path    = "data/raw/Churn_Modelling kaggel.csv"
    output_directory = "saved_models"

    if os.path.exists(raw_data_path):
        train_customer_churn_model(
            raw_data_path,
            output_directory,
            use_smote=True,
            cv_splits=5,
            clv=200.0,
            offer_cost=20.0,
        )
    else:
        print(f"❌  Data file not found at: {raw_data_path}")
