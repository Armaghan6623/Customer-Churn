"""Model audit — fairness, geographic error distribution, and business metrics.

Addresses two research gaps identified in:
  "Customer Churn Prediction: A Systematic Review" (2025)

  Gap 4 — Business metrics: reports EMP, profit curve, ROI alongside
           the standard fairness / error-distribution audit.
"""
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from classification.business_metrics import (
    cost_benefit_matrix,
    expected_maximum_profit,
    roi_of_retention,
    format_business_report,
    plot_profit_curve,
)


class ModelAuditor:
    """Run a comprehensive audit: fairness + geographic errors + business metrics."""

    def __init__(
        self,
        model_path: str = "saved_models/churn_pipeline.joblib",
        data_path: str = None,
        clv: float = 200.0,
        offer_cost: float = 20.0,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at {model_path}. Run train.py first.")

        # Resolve data path — try canonical name first, then legacy name
        if data_path is None:
            for candidate in [
                os.path.join("customer_crunch", "data", "customer_churn_dataset.csv"),
                os.path.join("data", "customer_churn_dataset.csv"),
                "/app/customer_crunch/data/customer_churn_dataset.csv",
                "/app/data/customer_churn_dataset.csv",
                os.path.join("data", "raw", "Churn_Modelling kaggel.csv"),
                os.path.join("customer_crunch", "data", "raw", "Churn_Modelling kaggel.csv"),
            ]:
                if os.path.exists(candidate):
                    data_path = candidate
                    break
            if data_path is None:
                raise FileNotFoundError(
                    "Reference dataset not found. Expected "
                    "customer_crunch/data/customer_churn_dataset.csv"
                )

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ Raw data not found at {data_path}.")

        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact
        self.df       = pd.read_csv(data_path)
        self.clv      = clv
        self.offer_cost = offer_cost

        # Show stored training business metrics if present
        if isinstance(artifact, dict) and "business_metrics" in artifact:
            bm = artifact["business_metrics"]
            print("📦  Stored training-time business metrics:")
            for k, v in bm.items():
                print(f"    {k}: {v}")
            print()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _prepare(self):
        """Return (X, y_true, y_pred, y_prob, audit_df)."""
        X = self.df.drop(
            columns=["CustomerId", "Surname", "Exited", "RowNumber"],
            errors="ignore",
        )
        y_true = self.df["Exited"].values

        y_pred = self.pipeline.predict(X)
        y_prob = self.pipeline.predict_proba(X)[:, 1]

        audit_df = self.df.copy()
        audit_df["_pred"]  = y_pred
        audit_df["_prob"]  = y_prob
        return X, y_true, y_pred, y_prob, audit_df

    # ------------------------------------------------------------------ #
    # Public audit methods                                                 #
    # ------------------------------------------------------------------ #

    def run_fairness_audit(self, audit_df: pd.DataFrame, y_true, y_pred) -> dict:
        """Disparate Impact Ratio and overall classification report."""
        print("\n🔒  FAIRNESS AUDIT")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=["Retained", "Churned"]))

        female_rate = audit_df[audit_df["Gender"] == "Female"]["_pred"].mean()
        male_rate   = audit_df[audit_df["Gender"] == "Male"]["_pred"].mean()
        dir_ratio   = female_rate / male_rate if male_rate > 0 else 0.0

        print(f"  Female predicted churn rate : {female_rate*100:.2f}%")
        print(f"  Male   predicted churn rate : {male_rate*100:.2f}%")
        print(f"  Disparate Impact Ratio (DIR): {dir_ratio:.4f}")
        if 0.80 <= dir_ratio <= 1.25:
            print("  ✅ PASS — within acceptable fairness boundary (0.80–1.25)")
        else:
            print("  ⚠️  WARNING — potential demographic bias detected")

        return {"female_rate": female_rate, "male_rate": male_rate, "dir": dir_ratio}

    def run_geographic_audit(self, audit_df: pd.DataFrame) -> dict:
        """Error rates sliced by geography."""
        print("\n🌍  GEOGRAPHIC ERROR AUDIT")
        print("=" * 60)
        geo_results = {}
        for country in sorted(audit_df["Geography"].unique()):
            sub = audit_df[audit_df["Geography"] == country]
            cm  = confusion_matrix(sub["Exited"], sub["_pred"])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0
            err_rate = (fp + fn) / len(sub) * 100
            print(f"  {country:<12}  error={err_rate:.2f}%  "
                  f"TP={tp} FP={fp} FN={fn} TN={tn}")
            geo_results[country] = {"error_rate": err_rate, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        return geo_results

    def run_business_audit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_dir: str = "saved_models",
    ) -> dict:
        """EMP, profit curve, ROI — the business-oriented evaluation layer."""
        print("\n💰  BUSINESS METRICS AUDIT")
        print("=" * 60)

        cb         = cost_benefit_matrix(clv=self.clv, offer_cost=self.offer_cost)
        emp_result = expected_maximum_profit(y_true, y_prob, cb=cb)
        roi_result = roi_of_retention(
            y_true, y_prob,
            opt_threshold=emp_result["optimal_threshold"],
            cb=cb,
        )

        report = format_business_report(emp_result, roi_result)
        plain  = report.replace("**", "").replace("##", "").replace("|", " | ")
        print(plain)

        # Save profit curve
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, "audit_profit_curve.png")
        plot_profit_curve(
            emp_result["curve_df"],
            opt_threshold=emp_result["optimal_threshold"],
            emp=emp_result["emp"],
            save_path=plot_path,
        )
        print(f"  📈 Profit curve saved to: {plot_path}")

        return {"emp": emp_result, "roi": roi_result}

    def run_full_audit(self, save_dir: str = "saved_models") -> dict:
        """Run all three audit passes and return a combined results dict."""
        X, y_true, y_pred, y_prob, audit_df = self._prepare()

        fairness_results  = self.run_fairness_audit(audit_df, y_true, y_pred)
        geo_results       = self.run_geographic_audit(audit_df)
        business_results  = self.run_business_audit(y_true, y_prob, save_dir=save_dir)

        auc = roc_auc_score(y_true, y_prob)
        print(f"\n  AUC-ROC (full dataset): {auc:.4f}")

        return {
            "fairness":  fairness_results,
            "geography": geo_results,
            "business":  business_results,
            "auc_roc":   round(auc, 4),
        }

    # Keep the old method name for backwards compatibility
    def run_bias_and_fairness_audit(self):
        return self.run_full_audit()


if __name__ == "__main__":
    try:
        auditor = ModelAuditor()
        auditor.run_full_audit()
    except Exception as e:
        print(f"❌ Audit failed: {e}")
