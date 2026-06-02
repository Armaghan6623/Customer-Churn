"""Model training — full MLOps pipeline.

Addresses research gaps from two 2025 papers:

Paper 1 — "A Systematic Review of Recent Advances, Trends, and Challenges"
  Gap 1  Class imbalance   → SMOTE oversampling + scale_pos_weight
  Gap 4  Business metrics  → EMP, profit curve, ROI after every run

Paper 2 — "Prediction of Customer Churn in Financial Sectors using ML"
  Gap 2  Feature quality   → engineered interaction/ratio features
  Gap 5  Digital behaviour → AgeGroup, BalanceSalaryRatio, IsHighValue
  Gap 3  Trustworthy AI    → MLflow experiment tracking for auditability
  Gap 1  Model accuracy    → RandomizedSearchCV hyperparameter tuning
                           → stratified k-fold cross-validation
"""
from __future__ import annotations

import os
import warnings
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from classification.dataset_config import DatasetConfig, KAGGLE_BANK_CHURN
from classification.feature_engineering import (
    engineer_features,
    ENGINEERED_NUMERIC_FEATURES,
)
from classification.business_metrics import (
    cost_benefit_matrix,
    expected_maximum_profit,
    roi_of_retention,
    format_business_report,
    plot_profit_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Optional dependencies (graceful fallback) ─────────────────────────────

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ── Hyperparameter search space ───────────────────────────────────────────

XGBOOST_PARAM_DIST = {
    "classifier__n_estimators":    [100, 200, 300, 400, 500],
    "classifier__max_depth":       [3, 4, 5, 6, 7, 8],
    "classifier__learning_rate":   [0.01, 0.03, 0.05, 0.08, 0.1],
    "classifier__subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
    "classifier__min_child_weight":[1, 3, 5, 7],
    "classifier__gamma":           [0, 0.1, 0.2, 0.3],
    "classifier__reg_alpha":       [0, 0.01, 0.1, 1.0],
    "classifier__reg_lambda":      [0.5, 1.0, 1.5, 2.0],
}


def _build_preprocessor(
    numeric_features: list,
    categorical_features: list,
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )


def _run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline,
    n_splits: int = 5,
) -> dict:
    """Stratified k-fold CV — returns mean ± std for five metrics."""
    cv      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["f1", "roc_auc", "average_precision", "precision", "recall"]
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    summary = {}
    for m in scoring:
        vals = results[f"test_{m}"]
        summary[m] = {"mean": round(float(vals.mean()), 4),
                      "std":  round(float(vals.std()),  4)}
    return summary


def _apply_smote(preprocessor, X_train, y_train, X_test):
    """Fit preprocessor, apply SMOTE on training split, return transformed arrays."""
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train_t, y_train)

    n_synth = len(y_res) - len(y_train)
    print(f"   Synthetic minority samples added: {n_synth:,}")
    print(f"   Post-SMOTE — retained: {(y_res==0).sum():,}  |"
          f"  churned: {(y_res==1).sum():,}")
    return X_res, y_res, X_test_t


def train_customer_churn_model(
    data_path: str,
    save_dir: str,
    config: DatasetConfig = KAGGLE_BANK_CHURN,
    use_smote: bool = True,
    use_feature_engineering: bool = True,
    tune_hyperparams: bool = True,
    n_search_iter: int = 20,
    cv_splits: int = 5,
    clv: float = 200.0,
    offer_cost: float = 20.0,
    mlflow_experiment: str = "customer_churn",
) -> str:
    """Full training pipeline.

    Parameters
    ----------
    data_path              : CSV dataset path.
    save_dir               : Directory for model artifact and plots.
    config                 : DatasetConfig schema.
    use_smote              : Apply SMOTE (requires imbalanced-learn).
    use_feature_engineering: Add interaction/ratio features before training.
    tune_hyperparams       : Run RandomizedSearchCV (n_search_iter trials).
    n_search_iter          : Number of random hyperparameter combinations.
    cv_splits              : Folds for stratified CV (0 = skip).
    clv                    : Customer Lifetime Value for business metrics.
    offer_cost             : Retention offer cost per contacted customer.
    mlflow_experiment      : MLflow experiment name (None = skip tracking).

    Returns
    -------
    str : Absolute path to saved model artifact.
    """
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"📥  Loading dataset: {data_path}")
    print(f"{'='*60}")

    df = pd.read_csv(data_path)

    # ── Feature engineering ───────────────────────────────────────────────
    if use_feature_engineering:
        print("\n🔧  Applying feature engineering...")
        df_eng = engineer_features(
            df.drop(columns=config.drop_cols, errors="ignore")
        )
        # Restore the target column
        df_eng[config.target_col] = df[config.target_col].values

        added = [f for f in ENGINEERED_NUMERIC_FEATURES if f in df_eng.columns]
        print(f"   Added features: {added}")

        numeric_features = [
            c for c in (config.numeric_features + added)
            if c in df_eng.columns
        ]
        categorical_features = [
            c for c in config.categorical_features if c in df_eng.columns
        ]
        X = df_eng.drop(
            columns=config.drop_cols + [config.target_col] + ["Exited"],
            errors="ignore",
        )[numeric_features + categorical_features]
        y = df_eng[config.target_col]
    else:
        X = df.drop(columns=config.drop_cols + [config.target_col], errors="ignore")
        y = df[config.target_col]
        known = [c for c in config.all_feature_cols if c in X.columns]
        X = X[known]
        numeric_features     = [c for c in config.numeric_features     if c in X.columns]
        categorical_features = [c for c in config.categorical_features if c in X.columns]
        added = []

    # ── Class imbalance report ────────────────────────────────────────────
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    ratio = n_neg / n_pos if n_pos else 1.0
    print(f"\n⚖️   Class distribution — retained: {n_neg:,} | churned: {n_pos:,}"
          f" | ratio: {ratio:.2f}:1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Build base pipeline ───────────────────────────────────────────────
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    base_clf = XGBClassifier(
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   base_clf),
    ])

    # ── Hyperparameter tuning ─────────────────────────────────────────────
    best_params: dict = {}
    if tune_hyperparams:
        print(f"\n🔍  Hyperparameter tuning ({n_search_iter} random trials, "
              f"{cv_splits}-fold stratified CV, scoring=f1)...")
        cv_tune = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=XGBOOST_PARAM_DIST,
            n_iter=n_search_iter,
            scoring="f1",
            cv=cv_tune,
            n_jobs=-1,
            random_state=42,
            verbose=0,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_params = {k.replace("classifier__", ""): v
                       for k, v in search.best_params_.items()}
        print(f"   Best params: {best_params}")
        print(f"   Best CV F1 : {search.best_score_:.4f}")
        final_pipeline = search.best_estimator_
    else:
        final_pipeline = base_pipeline

    # ── SMOTE (fit/transform outside Pipeline to avoid data leakage) ──────
    smote_applied = False
    X_test_t = None

    if use_smote and _SMOTE_AVAILABLE and not tune_hyperparams:
        # When tuning, sklearn CV handles splits internally — SMOTE inside
        # CV would cause leakage.  Apply SMOTE only for the final fit pass.
        print("\n🔄  Applying SMOTE to training split...")
        X_res, y_res, X_test_t = _apply_smote(
            preprocessor, X_train, y_train, X_test
        )
        final_pipeline.named_steps["classifier"].fit(X_res, y_res)
        smote_applied = True
    elif use_smote and not _SMOTE_AVAILABLE:
        print("⚠️   imbalanced-learn not installed — SMOTE skipped. "
              "Run: pip install imbalanced-learn")
    elif not tune_hyperparams:
        print("\n🚀  Training pipeline (no SMOTE, no tuning)...")
        final_pipeline.fit(X_train, y_train)

    # If tuning was done, the best_estimator_ is already fitted.
    # ── Evaluation ───────────────────────────────────────────────────────
    print("\n📊  Hold-out test evaluation:")
    if smote_applied and X_test_t is not None:
        y_pred = final_pipeline.named_steps["classifier"].predict(X_test_t)
        y_prob = final_pipeline.named_steps["classifier"].predict_proba(X_test_t)[:, 1]
    else:
        y_pred = final_pipeline.predict(X_test)
        y_prob = final_pipeline.predict_proba(X_test)[:, 1]

    report_str = classification_report(y_test, y_pred)
    print(report_str)

    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr  = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    rec     = recall_score(y_test, y_pred)
    acc     = float((y_pred == np.array(y_test)).mean())

    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc_roc:.4f}")
    print(f"  AUC-PR   : {auc_pr:.4f}  (more informative under class imbalance)")

    # ── Stratified CV (on tuned model params, fresh pipeline) ────────────
    cv_results: dict = {}
    if cv_splits > 1:
        print(f"\n📐  {cv_splits}-fold stratified cross-validation...")
        # Re-build a fresh pipeline with the best found params for CV
        clf_params = {k: v for k, v in base_clf.get_params().items()}
        if best_params:
            clf_params.update(best_params)
        cv_clf = XGBClassifier(**clf_params, random_state=42, eval_metric="logloss")
        cv_pipe = Pipeline([
            ("preprocessor", _build_preprocessor(numeric_features, categorical_features)),
            ("classifier",   cv_clf),
        ])
        cv_results = _run_cv(X, y, cv_pipe, n_splits=cv_splits)
        print(f"  {'Metric':<22} {'Mean':>8}  {'Std':>8}")
        print("  " + "-" * 42)
        for m, v in cv_results.items():
            print(f"  {m:<22} {v['mean']:>8.4f}  ±{v['std']:.4f}")

    # ── Business metrics ──────────────────────────────────────────────────
    print("\n💰  Business-oriented metrics...")
    cb         = cost_benefit_matrix(clv=clv, offer_cost=offer_cost)
    emp_result = expected_maximum_profit(np.array(y_test), y_prob, cb=cb)
    roi_result = roi_of_retention(
        np.array(y_test), y_prob,
        opt_threshold=emp_result["optimal_threshold"], cb=cb,
    )
    biz_report = format_business_report(emp_result, roi_result)
    plain = biz_report.replace("**", "").replace("##", "").replace("|", " | ")
    print(plain)

    os.makedirs(save_dir, exist_ok=True)
    profit_plot = os.path.join(save_dir, "profit_curve.png")
    plot_profit_curve(
        emp_result["curve_df"],
        opt_threshold=emp_result["optimal_threshold"],
        emp=emp_result["emp"],
        save_path=profit_plot,
    )
    print(f"📈  Profit curve → {profit_plot}")

    # ── MLflow tracking ───────────────────────────────────────────────────
    if _MLFLOW_AVAILABLE and mlflow_experiment:
        print(f"\n📝  Logging to MLflow experiment: '{mlflow_experiment}'")
        try:
            mlflow.set_experiment(mlflow_experiment)
            with mlflow.start_run(run_name="xgboost_churn_train"):
                # Parameters
                mlflow.log_param("use_smote",              smote_applied)
                mlflow.log_param("use_feature_engineering",use_feature_engineering)
                mlflow.log_param("tune_hyperparams",        tune_hyperparams)
                mlflow.log_param("n_search_iter",           n_search_iter)
                mlflow.log_param("cv_splits",               cv_splits)
                mlflow.log_param("imbalance_ratio",         round(ratio, 2))
                mlflow.log_param("clv",                     clv)
                mlflow.log_param("offer_cost",              offer_cost)
                mlflow.log_param("engineered_features",
                                 json.dumps(added))
                if best_params:
                    for k, v in best_params.items():
                        mlflow.log_param(f"best_{k}", v)

                # Standard metrics
                mlflow.log_metric("accuracy",    acc)
                mlflow.log_metric("f1",          f1)
                mlflow.log_metric("precision",   prec)
                mlflow.log_metric("recall",      rec)
                mlflow.log_metric("auc_roc",     auc_roc)
                mlflow.log_metric("auc_pr",      auc_pr)

                # Business metrics
                mlflow.log_metric("emp",               emp_result["emp"])
                mlflow.log_metric("emp_per_customer",  emp_result["emp_per_customer"])
                mlflow.log_metric("optimal_threshold", emp_result["optimal_threshold"])
                mlflow.log_metric("roi_pct",           roi_result["roi_pct"])
                mlflow.log_metric("lift_over_random",  emp_result["lift_over_random"])

                # CV metrics
                for m, v in cv_results.items():
                    mlflow.log_metric(f"cv_{m}_mean", v["mean"])
                    mlflow.log_metric(f"cv_{m}_std",  v["std"])

                # Artefacts
                mlflow.log_artifact(profit_plot, artifact_path="plots")
                mlflow.sklearn.log_model(final_pipeline, "model")

            print("   MLflow run logged successfully.")
        except Exception as mlf_err:
            print(f"   ⚠️  MLflow logging failed (non-fatal): {mlf_err}")
    elif not _MLFLOW_AVAILABLE:
        print("⚠️   mlflow not installed — tracking skipped. "
              "Run: pip install mlflow")

    # ── Save artifact ─────────────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 1)
    model_path = os.path.join(save_dir, config.model_filename)
    artifact = {
        "pipeline":             final_pipeline,
        "config":               config,
        "smote_applied":        smote_applied,
        "feature_engineering":  use_feature_engineering,
        "engineered_features":  added,
        "numeric_features":     numeric_features,
        "categorical_features": categorical_features,
        "best_params":          best_params,
        "cv_results":           cv_results,
        "business_metrics": {
            "emp":               emp_result["emp"],
            "emp_per_customer":  emp_result["emp_per_customer"],
            "optimal_threshold": emp_result["optimal_threshold"],
            "roi_pct":           roi_result["roi_pct"],
            "lift_over_random":  emp_result["lift_over_random"],
            "auc_roc":           round(auc_roc, 4),
            "auc_pr":            round(auc_pr,  4),
            "f1":                round(f1,      4),
            "accuracy":          round(acc,     4),
            "recall":            round(rec,     4),
            "precision":         round(prec,    4),
        },
    }
    joblib.dump(artifact, model_path)
    print(f"\n✅  Pipeline saved → {model_path}  ({elapsed}s)")
    return model_path


if __name__ == "__main__":
    _DATA = "data/raw/Churn_Modelling kaggel.csv"
    _SAVE = "saved_models"

    if os.path.exists(_DATA):
        train_customer_churn_model(
            data_path=_DATA,
            save_dir=_SAVE,
            use_smote=True,
            use_feature_engineering=True,
            tune_hyperparams=True,
            n_search_iter=20,
            cv_splits=5,
            clv=200.0,
            offer_cost=20.0,
        )
    else:
        print(f"❌  Data file not found: {_DATA}")
